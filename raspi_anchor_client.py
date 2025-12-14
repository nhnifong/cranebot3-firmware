
import asyncio
import os
import signal
import websockets
import time
import json
from cv_common import *
import cv2
import av
import numpy as np
import model_constants
from functools import partial
import threading
from config_loader import *
from util import *
import os
from trainer.stringman_pilot import IMAGE_SHAPE
from collections import defaultdict, deque
from websockets.exceptions import ConnectionClosedError, InvalidURI, InvalidHandshake, ConnectionClosedOK
from generated.nf import telemetry, common
import copy
from video_streamer import VideoStreamer

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'fast;1|fflags;nobuffer|flags;low_delay'

# number of origin detections to average
max_origin_detections = 12

# fastSAM parameters
# seconds between processing frames with fastSAM. there is no need need to run it on every frame, since 
# we are looking at a relatively static image.
sam_rate = 1.0 # per second
sam_confidence_cutoff = 0.75

def pose_from_det(det):
    return (np.array(det['r'], dtype=float).reshape((3,)), np.array(det['t'], dtype=float).reshape((3,)))

# the genertic client for a raspberri pi based robot component
class ComponentClient:
    def __init__(self, address, port, datastore, ob, pool, stat):
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
            container = av.open(video_uri, options=options, mode='r')
            stream = next(s for s in container.streams if s.type == 'video')
            stream.thread_type = "SLICE"

            # start thread for frame risize and forwarding
            encoder_thread = None
            if self.anchor_num in [None, *self.config.preferred_cameras]:
                encoder_thread = threading.Thread(target=self.frame_resizer_loop, daemon=True)
                encoder_thread.start()

            print(f'video connection successful')
            self.conn_status.video_status = telemetry.ConnStatus.CONNECTED
            self.notify_video = True
            lastSam = time.time()
            last_time = time.time()

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
                            self.pool.apply_async(locate_markers, (self.frame,), callback=partial(self.handle_detections, timestamp=timestamp))
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

        except av.error.TimeoutError:
            print('no video stream available')
            self.conn_status.video_status = telemetry.ConnStatus.NOT_DETECTED
            self.notify_video = True
            return

        finally:
            if 'container' in locals():
                container.close()

    def frame_resizer_loop(self):
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
        """

        # TODO allow these to changes when in a teleop mode
        if self.anchor_num is None:
            final_shape = sf_target_shape # resize for centering network input
            final_fps = 10
        else:   
            final_shape = (IMAGE_SHAPE[1], IMAGE_SHAPE[0]) # resize for dobby network input
            final_fps = 10

        vs = VideoStreamer(width=final_shape[1], height=final_shape[0], fps=final_fps, rtmp_url=None)
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
                continue

            # Do the actual work outside the lock
            # This lets the receive_video loop add the next frame without waiting for the encode.
            if self.anchor_num is None:
                # gripper
                # stabilize and resize for centering network input
                temp_image = cv2.resize(frame_to_encode, sf_input_shape, interpolation=cv2.INTER_AREA)
                fudge_latency =  0.3
                gripper_quat = self.datastore.imu_quat.getClosest(self.last_frame_cap_time - fudge_latency)[1:]
                if self.calibrating_room_spin or self.config.gripper.frame_room_spin is None:
                    # roomspin = 15/180*np.pi
                    roomspin = 0
                else:
                    roomspin = self.config.gripper.frame_room_spin
                self.last_frame_resized = stabilize_frame(temp_image, gripper_quat, roomspin)
            else:
                # anchors
                self.last_frame_resized = cv2.resize(frame_to_encode, final_shape, interpolation=cv2.INTER_AREA)

            # send self.last_frame_resized to the UI process
            vs.send_frame(self.last_frame_resized)
            frames_sent += 1
            if frames_sent == 20:
                self.local_udp_port = vs.local_udp_port
                self.ob.send_ui(video_ready=telemetry.VideoReady(
                    is_gripper=self.anchor_num is None,
                    anchor_num=self.anchor_num,
                    local_uri=f'udp://127.0.0.1:{vs.local_udp_port}'
                ))

        vs.stop()

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
                print(f"Connection to {self.address} closed.")
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
                print(f'No line record update sent from {self.anchor_num} in {TIMEOUT} seconds. it may have gone offline. sending ping')
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

class RaspiAnchorClient(ComponentClient):
    def __init__(self, address, port, anchor_num, datastore, ob, pool, stat):
        super().__init__(address, port, datastore, ob, pool, stat)
        self.anchor_num = anchor_num # which anchor are we connected to
        self.conn_status = telemetry.ComponentConnStatus(
            is_gripper=False,
            anchor_num=self.anchor_num,
            websocket_status=telemetry.ConnStatus.NOT_DETECTED,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
        )
        self.last_raw_encoder = None
        self.raw_gant_poses = []
        self.anchor_pose = poseProtoToTuple(self.config.anchors[anchor_num].pose)
        self.camera_pose = np.array(compose_poses([
            self.anchor_pose,
            model_constants.anchor_camera,
        ]))
        self.gantry_pos_sightings = deque(maxlen=100)
        self.gantry_pos_sightings_lock = threading.RLock()


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

            if detection['n'] in ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3']:
                # save all the detections of the origin for later analysis
                self.origin_poses[detection['n']].append(pose_from_det(detection))

            if detection['n'] == 'gantry':
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.
                # you have the pose of gantry_front relative to a particular anchor camera
                # convert it to a pose relative to the origin
                pose = np.array(compose_poses([
                    self.anchor_pose, # obtained from calibration
                    model_constants.anchor_camera, # constant
                    pose_from_det(detection), # the pose obtained just now
                    gantry_april_inv, # constant
                ]))
                position = pose.reshape(6)[3:]
                self.datastore.gantry_pos.insert(np.concatenate([[timestamp], [self.anchor_num], position])) # take only the position
                # print(f'Inserted gantry pose ts={timestamp}, pose={pose}')
                self.datastore.gantry_pos_event.set()

                self.last_gantry_frame_coords = np.array(detection['t'], dtype=float)
                with self.gantry_pos_sightings_lock:
                    self.gantry_pos_sightings.append(position)

                if self.save_raw:
                    self.raw_gant_poses.append(pose_from_det(detection))

            if detection['n'] in ['gamepad', 'hamper', 'trash', 'gamepad-back', 'hamper-back', 'trash-back']:
                offset = model_constants.basket_offset_inv if detection['n'].endswith('back') else model_constants.basket_offset
                pose = np.array(compose_poses([
                    self.anchor_pose,
                    model_constants.anchor_camera, # constant
                    pose_from_det(detection), # the pose obtained just now
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