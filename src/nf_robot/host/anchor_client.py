
import asyncio
import os
import signal
import websockets
import time
import json
import cv2
import av
import numpy as np
from functools import partial
import threading
import os
from collections import defaultdict, deque
from websockets.exceptions import ConnectionClosedError, InvalidURI, InvalidHandshake, ConnectionClosedOK
import copy
from scipy.spatial.transform import Rotation

from nf_robot.common.cv_common import *
from nf_robot.common.pose_functions  import *
from nf_robot.common.util import *
import nf_robot.common.definitions as model_constants
from nf_robot.ml.target_heatmap import HM_IMAGE_RES
from nf_robot.generated.nf import telemetry, common
from nf_robot.host.video_streamer import VideoStreamer

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'fast;1|fflags;nobuffer|flags;low_delay'

# number of origin detections to average
max_origin_detections = 12

# fastSAM parameters
# seconds between processing frames with fastSAM. there is no need need to run it on every frame, since 
# we are looking at a relatively static image.
sam_rate = 1.0 # per second
sam_confidence_cutoff = 0.75

# the genertic client for a raspberri pi based robot component
class ComponentClient:
    def __init__(self, address, port, datastore, ob, pool, stat, telemetry_env):
        self.address = address
        self.port = port
        self.origin_poses = defaultdict(lambda: deque(maxlen=max_origin_detections))
        self.datastore = datastore
        self.ob = ob # instance of observer. mocks only need the update_avg_named_pos and send_ui methods
        self.websocket = None
        self.connected = False  # status of connection to websocket
        self.receive_task = None  # Task for receiving messages from websocket
        self.video_task = None  # Task for streaming video
        self.stream_start_ts = None
        self.pool = pool
        self.stat = stat
        self.last_gantry_frame_coords = None
        self.ct = None # task to connect to websocket
        self.save_raw = False
        self.connection_established_event = None
        self.frame = None # last frame of video seen
        self.last_frame_cap_time = None
        self.heartbeat_receipt = asyncio.Event()
        self.safety_task = None
        self.local_udp_port = None
        self.telemetry_env = telemetry_env

        # things used by jpeg/resizing thread
        self.frame_lock = threading.Lock()
        # This condition variable signals the worker when a new frame is ready
        self.new_frame_condition = threading.Condition(self.frame_lock)
        self.last_frame_resized = None
        # The final, encoded bytes for lerobot. Atomic write, so no lock needed.
        self.lerobot_jpeg_bytes = None
        self.lerobot_mode = False # when false disables constant encoded to improve performance.
        self.calibrating_room_spin = False # set to true momentarily during auto calibration

        self.config = ob.config

        self.conn_status = None # subclass needs to set this in init

    def send_conn_status(self):
        self.ob.send_ui(component_conn_status=copy.deepcopy(self.conn_status))

    def receive_video(self, port):
        video_uri = f'tcp://{self.address}:{port}'
        print(f'Connecting to {video_uri}')
        self.conn_status.video_status = telemetry.ConnStatus.CONNECTING
        # cannot send here, not in event loop
        self.notify_video = True

        options = {
            'rtsp_transport': 'tcp',
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'fast': '1',
        }

        try:

            # attempt = 3 # sometimes it just takes longer than expected.
            # while attempt>0:
            #     try:
            container = av.open(video_uri, options=options, mode='r')
            # except av.error.ConnectionRefusedError:
            #     attempt-=1
            #     if attempt==0:
            #         raise
            #     time.sleep(1.5)

            stream = next(s for s in container.streams if s.type == 'video')
            stream.thread_type = "SLICE"

            # start thread for frame risize and forwarding
            encoder_thread = None
            components_to_stream = [None, *self.config.preferred_cameras]
            if self.anchor_num in components_to_stream:
                encoder_thread = threading.Thread(target=self.frame_resizer_loop, kwargs={"cam_num": components_to_stream.index(self.anchor_num)}, daemon=True)
                encoder_thread.start()

            print(f'video connection successful')
            self.conn_status.video_status = telemetry.ConnStatus.CONNECTED
            self.notify_video = True
            lastSam = time.time()
            last_time = time.time()

            def error_callback_func(error):
                print(f"Error in pool worker: {error}")

            for av_frame in container.decode(stream):
                if not self.connected:
                    break
                # determine the wall time when the frame was captured
                timestamp = self.stream_start_ts + av_frame.time
                self.last_frame_cap_time = timestamp

                fr = av_frame.to_ndarray(format='rgb24')
                with self.new_frame_condition:
                    self.frame = fr
                    self.new_frame_condition.notify()

                # save information about stream latency and framerate
                now = time.time()
                self.stat.latency.append(now - timestamp)
                fr = 1/(now - last_time)
                self.stat.framerate.append(fr)
                last_time = now

                # send frame to apriltag detector
                if self.anchor_num is not None:
                    try:
                        if self.stat.pending_frames_in_pool < 60:
                            self.stat.pending_frames_in_pool += 1
                            self.pool.apply_async(
                                locate_markers,
                                (self.frame, self.config.camera_cal),
                                callback=partial(self.handle_detections, timestamp=timestamp),
                                error_callback=error_callback_func)
                        else:
                            pass
                            # print(f'Dropping frame because there are already too many pending.')
                            # TODO record fraction of frames which are dropped in stat collector
                    except ValueError:
                        break # the pool is not running

                # sleep is mandatory or this thread could prevent self.handle_detections from running and fill up the pool with work.
                # handle_detections runs in this process, but in a thread managed by the pool.
                time.sleep(0.005)

            if encoder_thread is not None:
                encoder_thread.join()

        except (av.error.TimeoutError, av.error.ConnectionRefusedError):
            print('no video stream available')
            self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
            self.notify_video = True
            return

        finally:
            if 'container' in locals():
                container.close()

    def frame_resizer_loop(self, cam_num):
        """
        This runs in a dedicated thread. It waits for a signal that a new
        frame is available, resizes it, and stabilizes it in the gripper case.
        The the frame is written to an ffmpeg subprocess that is sending the video
        to the ui over UDP or RTMP depending on where it is.

        The purpose of this method is to have a frame ready to send as fast as possible,
        As well as to present resized frames for inference networks to use.

        For the sake of performance, the UIs are made to consume a resolution identcal to the models,
        but if they needed to be different, we could just to two different resize ops.

        Numpy functions such as those used by cv2.resize actually release the GIL
        which is why this is a thread not a task (main loop can run faster this way)

        cam_num identifies which of the preferred cameras this is. 0 is the gripper, 1 and 2 are the two overhead cams.
        """

        # TODO allow these to change when in a teleop mode
        if self.anchor_num is None:
            final_shape = SF_TARGET_SHAPE # resize for centering network input
            final_fps = 10
        else:   
            final_shape = HM_IMAGE_RES # resize for target heatmap network input
            final_fps = 10

        path = f'stringman/{self.config.robot_id}/{cam_num}'

        if self.telemetry_env is None:
            rtmp = None # when in LAN mode do not upload ANYTHING to the cloud.
        elif self.telemetry_env == "local":
            rtmp = f'rtmp://localhost:1935/{path}?user=user&pass=pass'
        elif self.telemetry_env == "staging":
            rtmp = f'rtmp://media.neufangled.com:1935/{path}?user=user&pass=pass'
        elif self.telemetry_env == "production":
            rtmp = f'rtmp://media.neufangled.com:1935/{path}?user=user&pass=pass'
        else:
            rtmp = None

        # this is basically an ffmpeg subprocess
        vs = VideoStreamer(width=final_shape[0], height=final_shape[1], fps=final_fps, rtmp_url=rtmp)
        vs.start()

        frames_sent = 0
        time_last_frame_taken = time.time()-1

        while self.connected:
            with self.new_frame_condition:
                # Wait until the main receive_video loop signals us.
                # The 'wait' call will timeout after 1 second to re-check
                # the self.connected flag, allowing the thread to exit gracefully.
                signaled = self.new_frame_condition.wait(timeout=1.0)
                if not signaled:
                    continue
                # only take every nth frame based on framerate target
                now = time.time()
                if now < (time_last_frame_taken + 1/final_fps):
                    continue
                time_last_frame_taken = now
                # We were woken up, so copy the frame pointer while we have the lock
                frame_to_encode = self.frame

            if frame_to_encode is None:
                print(f'no frame to encode {self}')
                continue

            # Do the actual work outside the lock
            # This lets the receive_video loop add the next frame without waiting for the encode.
            self.last_frame_resized = self.process_frame(frame_to_encode)

            # send self.last_frame_resized to the UI process
            vs.send_frame(self.last_frame_resized)
            frames_sent += 1
            if frames_sent == 20:
                # sending the notification on the 20th frame ensures that the mediamtx server has something to send before clients connect
                self.local_udp_port = vs.local_udp_port
                self.ob.send_ui(video_ready=telemetry.VideoReady(
                    is_gripper=self.anchor_num is None,
                    anchor_num=self.anchor_num,
                    local_uri=f'udp://127.0.0.1:{vs.local_udp_port}',
                    stream_path=path,
                ))

        vs.stop()

    def process_frame(self, frame_to_encode):
        """
        Subclasses should override this function to resize or stabilize frames based on specific hardware constants
        the returned frame will be what is used for inference and sent to any teleoperation pipelines.
        Runs in a seperate thread from the main client.
        """
        return frame_to_encode

    async def connect_websocket(self):
        # main client loop
        self.conn_status.websocket_status = telemetry.ConnStatus.CONNECTING
        self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
        self.conn_status.ip_address = self.address
        self.send_conn_status()

        self.abnormal_shutdown = False # indicating we had a connection and then lost it unexpectedly
        self.failed_to_connect = False # indicating we failed to ever make a connection
        ws_uri = f"ws://{self.address}:{self.port}"
        print(f"Connecting to {ws_uri}...")
        try:
            async with websockets.connect(ws_uri, max_size=None, open_timeout=10) as websocket:
                self.connected = True
                print(f"Connected to {ws_uri}.")
                # Set an event that the observer is waiting on.
                if self.connection_established_event is not None:
                    self.connection_established_event.set()
                await self.receive_loop(websocket)
        except (asyncio.exceptions.CancelledError, websockets.exceptions.ConnectionClosedOK):
            pass # normal close
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Component server anum={self.anchor_num} disconnected abnormally: {e}")
            self.abnormal_shutdown = True
        except (OSError, TimeoutError, InvalidURI, InvalidHandshake) as e:
            print(f"Component server anum={self.anchor_num}: {e}")
            self.failed_to_connect = True
        finally:
            self.connected = False
        self.conn_status.websocket_status = telemetry.ConnStatus.NOT_DETECTED
        self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
        self.send_conn_status()
        return self.abnormal_shutdown

    async def receive_loop(self, websocket):
        self.conn_status.websocket_status = telemetry.ConnStatus.CONNECTED
        self.send_conn_status()
        # loop of a single websocket connection.
        # save a reference to this for send_commands
        self.websocket = websocket
        self.notify_video = False
        # send configuration to robot component to override default.
        r = await self.send_config()
        # start task to watch heartbeat event
        self.safety_task = asyncio.create_task(self.safety_monitor())
        vid_thread = None
        # Loop until disconnected
        while self.connected:
            try:
                message = await websocket.recv()
                # print(f'received message of length {len(message)}')
                update = json.loads(message)
                if 'video_ready' in update:
                    port = int(update['video_ready'][0])
                    self.stream_start_ts = float(update['video_ready'][1])
                    print(f'stream_start_ts={self.stream_start_ts} ({time.time()-self.stream_start_ts}s ago)')
                    vid_thread = threading.Thread(target=self.receive_video, kwargs={"port": port}, daemon=True)
                    vid_thread.start()
                # this event is used to detect an un-responsive state.
                self.heartbeat_receipt.set() 
                await self.handle_update_from_ws(update)

                # do this here because we seemingly can't do it in receive_video
                if self.notify_video:
                    self.send_conn_status()
                    self.notify_video = False

            except Exception as e:
                # don't catch websockets.exceptions.ConnectionClosedOK here because we want it to trip the infinite generator in websockets.connect
                # so it will stop retrying. after it has the intended effect, websockets.connect will raise it again, so we catch it in 
                # connect_websocket
                print(f"Connection to {self.address} closed. {e}")
                self.connected = False
                self.websocket = None
                # self.conn_status.websocket_status = telemetry.ConnStatus.NOT_DETECTED
                # self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
                # self.send_conn_status()
                raise e # TODO figure out if this causes the abnormal shutdown return value in connect_websocket like it should
                break
        if vid_thread is not None:
            # vid_thread should stop because self.connected is False
            vid_thread.join()

    async def send_commands(self, update):
        if self.connected:
            x = json.dumps(update)
            # by trying to get the result out of the future, you force any exception in the task to be raised
            # since this could be a websockets.exceptions.ConnectionClosedError it's important not to let it disappar
            result = await self.websocket.send(x)

    async def slow_stop_spool(self):
        # spool will decelerate at the rate allowed by the config file.
        # tracking mode will switch to 'speed'
        result = await self.send_commands({'aim_speed': 0})

    async def startup(self):
        self.ct = asyncio.create_task(self.connect_websocket())
        return await self.ct

    async def shutdown(self):
        if self.safety_task is not None:
            self.safety_task.cancel()
            result = await self.safety_task
        if self.connected:
            self.connected = False
            if not self.abnormal_shutdown and self.websocket:
                result = await self.websocket.close()
        elif self.ct:
            self.ct.cancel()
        print(f"Finished client {self.anchor_num} shutdown")

    def shutdown_sync(self):
        # this might get called twice
        print("\nWait for client shutdown (sync)")
        if self.connected:
            self.connected = False
            if self.websocket:
                asyncio.create_task(self.websocket.close())
        elif self.ct:
            self.ct.cancel()

    async def safety_monitor(self):
        """Notifies observer if this anchor stops sending line record updates for some time"""
        TIMEOUT=1 # seconds
        last_update = time.time()
        while self.connected:
            try:
                result = await asyncio.wait_for(self.heartbeat_receipt.wait(), TIMEOUT)
                # if you see the event within the timeout, all is well, clear it and wait again
                self.heartbeat_receipt.clear()
                last_update = time.time()
            except TimeoutError:
                print(f'No update sent from {self.anchor_num} in {TIMEOUT} seconds. it may have gone offline. sending ping')
                try:
                    pong_future = await self.websocket.ping()
                    latency = await asyncio.wait_for(pong_future, TIMEOUT)
                    # some hiccup on the server raspi made it unable to send anything for some time but it's not down.
                    print(f'Pong received in {latency}s, must have been my imagination.')
                    continue
                except (ConnectionClosedError, TimeoutError):
                    # it's no longer running, either because it lost power, or the server crashed.
                    print(f'Anchor {self.anchor_num} confirmed down. hasn\'t been seen in {time.time() - last_update} seconds.')
                    self.connected = False
                    # immediately trigger the "abnormal shutdown" return from the connect_websocket task
                    # this is how the observer is actually notified. follow the control flow by looking at `if abnormal_close:` in observer.py
                    if self.websocket and self.websocket.transport:
                        self.websocket.transport.close()
                except ConnectionClosedOK:
                    return
            except asyncio.exceptions.CancelledError:
                return

CAL_MARKERS = set(['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3'])
OTHER_MARKERS = set(['gamepad', 'hamper', 'trash', 'gamepad_back', 'hamper_back', 'trash_back'])

class RaspiAnchorClient(ComponentClient):
    def __init__(self, address, port, anchor_num, datastore, ob, pool, stat, telemetry_env):
        super().__init__(address, port, datastore, ob, pool, stat, telemetry_env)
        self.anchor_num = anchor_num # which anchor are we connected to
        self.conn_status = telemetry.ComponentConnStatus(
            is_gripper=False,
            anchor_num=self.anchor_num,
            websocket_status=telemetry.ConnStatus.NOT_DETECTED,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
        )
        self.last_raw_encoder = None
        self.extratilt = 0
        self.raw_gant_poses = deque(maxlen=12)
        self.updatePose(poseProtoToTuple(self.config.anchors[anchor_num].pose))
        self.gantry_pos_sightings = deque(maxlen=100)
        self.gantry_pos_sightings_lock = threading.RLock()

    def updatePose(self, pose):
        self.anchor_pose = pose
        self.camera_pose = np.array(compose_poses([
            self.anchor_pose,
            model_constants.anchor_camera,
            (np.array([0,0,self.extratilt/180*np.pi], dtype=float), np.zeros(3, dtype=float)),
        ]))

    async def handle_update_from_ws(self, update):
        if 'line_record' in update:
            self.datastore.anchor_line_record[self.anchor_num].insertList(np.array(update['line_record']))

            # this is the event that is set when *any* anchor sends a line record.
            # used by the position estimator to immedately recalculate the hang point
            self.datastore.anchor_line_record_event.set()

        if 'last_raw_encoder' in update:
            self.last_raw_encoder = update['last_raw_encoder']

        if len(self.gantry_pos_sightings) > 0:
            with self.gantry_pos_sightings_lock:
                self.ob.send_ui(gantry_sightings=telemetry.GantrySightings(
                    sightings=[common.Vec3(*position) for position in self.gantry_pos_sightings]
                ))
                self.gantry_pos_sightings.clear()

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)

        for detection in detections:

            if detection['n'] in CAL_MARKERS:
                # save all the detections of the origin for later analysis
                self.origin_poses[detection['n']].append(detection['p'])

            if detection['n'] == 'gantry':
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.
                # you have the pose of gantry_front relative to a particular anchor camera
                # convert it to a pose relative to the origin
                pose = np.array(compose_poses([
                    self.anchor_pose, # obtained from calibration
                    model_constants.anchor_camera, # constant
                    detection['p'], # the pose obtained just now
                    gantry_april_inv, # constant
                ]))
                position = pose.reshape(6)[3:]
                self.datastore.gantry_pos.insert(np.concatenate([[timestamp], [self.anchor_num], position])) # take only the position
                # print(f'Inserted gantry pose ts={timestamp}, pose={pose}')
                self.datastore.gantry_pos_event.set()

                self.last_gantry_frame_coords = detection['p'][1] # second item in pose tuple is position
                with self.gantry_pos_sightings_lock:
                    self.gantry_pos_sightings.append(position)

                if self.save_raw:
                    self.raw_gant_poses.append(detection['p'])

            if detection['n'] in OTHER_MARKERS:
                offset = model_constants.basket_offset_inv if detection['n'].endswith('back') else model_constants.basket_offset
                pose = np.array(compose_poses([
                    self.anchor_pose,
                    model_constants.anchor_camera, # constant
                    detection['p'], # the pose obtained just now
                    offset, # the named location is out in front of the tag.
                ]))
                position = pose.reshape(6)[3:]
                # save the position of this object for use in various planning tasks.
                self.ob.update_avg_named_pos(detection['n'], position)

    async def send_config(self):
        anchor_config_vars = {
            "MAX_ACCEL": self.config.max_accel,
            "REC_MOD": self.config.rec_mod,
            "RUNNING_WS_DELAY": self.config.running_ws_delay,
        }
        if len(anchor_config_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': anchor_config_vars}))

    def process_frame(self, frame_to_encode):
        return cv2.resize(frame_to_encode, HM_IMAGE_RES, interpolation=cv2.INTER_AREA)