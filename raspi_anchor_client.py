
import asyncio
import os
import signal
import websockets
import time
import json
from cv_common import locate_markers, compose_poses, invert_pose, average_pose, gantry_april_inv
import cv2
import av
import numpy as np
import model_constants
from functools import partial
import threading
from config import Config
import os
from trainer.stringman_pilot import IMAGE_SHAPE

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'fast;1|fflags;nobuffer|flags;low_delay'

# number of origin detections to average
max_origin_detections = 12

# fastSAM parameters
# seconds between processing frames with fastSAM. there is no need need to run it on every frame, since 
# we are looking at a relatively static image.
sam_rate = 1.0 # per second
sam_confidence_cutoff = 0.75

def pose_from_det(det):
    return (np.array(det['r'], dtype=float), np.array(det['t'], dtype=float))

# the genertic client for a raspberri pi based robot component
class ComponentClient:
    def __init__(self, address, port, datastore, to_ui_q, to_ob_q, pool, stat):
        self.address = address
        self.port = port
        self.origin_poses = []
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_ob_q = to_ob_q
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

        # things used by jpeg thread for training mode
        self.frame_lock = threading.Lock()
        # This condition variable signals the worker when a new frame is ready
        self.new_frame_condition = threading.Condition(self.frame_lock)
        self.last_frame_resized = None
        # The final, encoded bytes for lerobot. Atomic write, so no lock needed.
        self.lerobot_jpeg_bytes = None

        # todo: receive a command in observer that will set this value
        self.sendPreviewToUi = False

    def receive_video(self, port):
        video_uri = f'tcp://{self.address}:{port}'
        print(f'Connecting to {video_uri}')
        self.conn_status['video'] = 'connecting'
        self.to_ui_q.put({'connection_status': self.conn_status})

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

            print(f'video connection successful')
            self.conn_status['video'] = 'connected'
            self.notify_video = True
            lastSam = time.time()
            last_time = time.time()

            for av_frame in container.decode(stream):
                if not self.connected:
                    break
                # determine the wall time when the frame was captured
                timestamp = self.stream_start_ts + av_frame.time

                fr = av_frame.to_ndarray(format='rgb24')
                with self.new_frame_condition:
                    self.frame = fr
                    self.new_frame_condition.notify()
                        
                if self.sendPreviewToUi:
                    # send frame to UI
                    preview = cv2.flip(cv2.resize(cv2.cvtColor(self.frame, cv2.COLOR_RGB2RGBA), None, fx=0.25, fy=0.25), 0)
                    self.to_ui_q.put({'preview_image': {'anchor_num':self.anchor_num, 'image':preview}})

                # save information about stream latency and framerate
                now = time.time()
                self.stat.latency.append(now - timestamp)
                fr = 1/(now - last_time)
                self.stat.framerate.append(fr)
                last_time = now

                # send frame to apriltag detector
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

        except av.error.TimeoutError:
            print('no video stream available')
            self.conn_status['video'] = 'none'
            self.notify_video = True
            return

        finally:
            if 'container' in locals():
                container.close()

    def jpeg_encoder_loop(self):
        """
        This runs in a dedicated thread. It waits for a signal that a new
        frame is available, then encodes it and stores the bytes.
        These bytes are intended to be returnd by the GRPC lerobot connects with
        so this only needs to be running when that is connected.

        The purpose of this method is to have a frame ready to return to lerobot as fast as possible.
        Numpy functions such as those used by cv2.resize actually release the GIL
        which is why this is a thread not a task (main loop can run faster this way)
        """
        while self.connected:
            with self.new_frame_condition:
                # Wait until the main receive_video loop signals us.
                # The 'wait' call will timeout after 1 second to re-check
                # the self.connected flag, allowing the thread to exit gracefully.
                signaled = self.new_frame_condition.wait(timeout=1.0)
                if not signaled:
                    continue
                # We were woken up, so copy the frame pointer while we have the lock
                frame_to_encode = self.frame

            # Do the actual work outside the lock
            # This lets the receive_video loop add the next frame without waiting for the encode.
            if frame_to_encode is not None:
                # return the size lerobot is expecting. it's faster to do this resize before encoding.
                dsize = (IMAGE_SHAPE[1], IMAGE_SHAPE[0])
                self.last_frame_resized = cv2.resize(frame_to_encode, dsize, interpolation=cv2.INTER_AREA)
                params = [int(cv2.IMWRITE_JPEG_QUALITY), 99]
                is_success, buffer = cv2.imencode(".jpg", self.last_frame_resized, params)

                # Store the result. This is an atomic operation in Python.
                if is_success:
                    self.lerobot_jpeg_bytes = buffer.tobytes()
        
        print("Encoder thread exiting.")

    async def connect_websocket(self):
        # main client loop
        self.conn_status['websocket'] = 'connecting'
        self.conn_status['video'] = 'none'
        self.conn_status['ip_address'] = self.address
        self.to_ui_q.put({'connection_status': self.conn_status})
        abnormal_shutdown = False
        ws_uri = f"ws://{self.address}:{self.port}"
        print(f"Connecting to {ws_uri}...")
        try:
            # connect() can be used as an infinite asynchronous iterator to reconnect automatically on errors
            # It re-raises any error which it would not retry on. Some are expected on normal disconnects.
            async for websocket in websockets.connect(ws_uri, max_size=None, open_timeout=10):
                self.connected = True
                print(f"Connected to {ws_uri}.")
                # TODO Set an event that the observer is waiting on.
                if self.connection_established_event is not None:
                    self.connection_established_event.set()
                await self.receive_loop(websocket)
        except asyncio.exceptions.CancelledError:
            print("Cancelling connection")
        except websockets.exceptions.ConnectionClosedOK:
            print("Client closing connection")
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Component server anum={self.anchor_num} disconnected abnormally: {e}")
            abnormal_shutdown = True
        finally:
            self.connected = False
        return abnormal_shutdown

    async def receive_loop(self, websocket):
        print('receive loop')
        self.conn_status['websocket'] = 'connected'
        self.to_ui_q.put({'connection_status': self.conn_status})
        # loop of a single websocket connection.
        # save a reference to this for send_commands
        self.websocket = websocket
        self.notify_video = False
        # start thread for lerobot jpeg encoding
        encoder_thread = threading.Thread(target=self.jpeg_encoder_loop, daemon=True)
        encoder_thread.start()
        # send configuration to robot component to override default.
        asyncio.create_task(self.send_config())
        vid_thread = None
        # Loop until disconnected
        while self.connected:
            try:
                message = await websocket.recv()
                # print(f'received message of length {len(message)}')
                update = json.loads(message)
                if 'video_ready' in update:
                    print(f'got a video ready update {update}')
                    port = int(update['video_ready'][0])
                    self.stream_start_ts = float(update['video_ready'][1])
                    print(f'stream_start_ts={self.stream_start_ts} ({time.time()-self.stream_start_ts}s ago)')
                    if self.anchor_num in [None,2,3]:
                        vid_thread = threading.Thread(target=self.receive_video, kwargs={"port": port})
                        vid_thread.start()
                self.handle_update_from_ws(update)

                # do this here because we seemingly can't do it in receive_video
                if self.notify_video:
                    self.to_ui_q.put({'connection_status': self.conn_status})
                    self.notify_video = False

            except Exception as e:
                # don't catch websockets.exceptions.ConnectionClosedOK here because we want it to trip the infinite generator in websockets.connect
                # so it will stop retrying. after it has the intended effect, websockets.connect will raise it again, so we catch it in 
                # connect_websocket
                print(f"Connection to {self.address} closed.")
                self.connected = False
                self.websocket = None
                self.conn_status['websocket'] = 'none'
                self.conn_status['video'] = 'none'
                self.to_ui_q.put({'connection_status': self.conn_status})
                raise e
                break
        if vid_thread is not None:
            # vid_thread should stop because self.connected is False
            vid_thread.join()
        if encoder_thread is not None:
            encoder_thread.join()

    async def send_commands(self, update):
        if self.connected:
            x = json.dumps(update)
            # print(f'send commands {x}')
            await self.websocket.send(x)
        # just discard the update if not connected.

    async def slow_stop_spool(self):
        # spool will decelerate at the rate allowed by the config file.
        # tracking mode will switch to 'speed'
        await self.send_commands({'aim_speed': 0})

    async def startup(self):
        self.ct = asyncio.create_task(self.connect_websocket())
        return await self.ct

    async def shutdown(self):
        print("\nWait for client shutdown")
        if self.connected:
            self.connected = False
            if self.websocket:
                await self.websocket.close()
        elif self.ct:
            self.ct.cancel()
        print("Finished client shutdown")

    def shutdown_sync(self):
        # this might get called twice
        print("\nWait for client shutdown (sync)")
        if self.connected:
            self.connected = False
            if self.websocket:
                asyncio.create_task(self.websocket.close())
        elif self.ct:
            self.ct.cancel()

class RaspiAnchorClient(ComponentClient):
    def __init__(self, address, port, anchor_num, datastore, to_ui_q, to_ob_q, pool, stat):
        super().__init__(address, port, datastore, to_ui_q, to_ob_q, pool, stat)
        self.anchor_num = anchor_num # which anchor are we connected to
        self.conn_status = {'anchor_num': self.anchor_num}
        self.last_raw_encoder = None
        self.raw_gant_poses = []
        self.line_record_receipt = asyncio.Event()

        config = Config()
        self.anchor_pose = config.anchors[anchor_num].pose
        self.to_ui_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})

    def handle_update_from_ws(self, update):
        # TODO if we are not regularly receiving line_record, display this as a server problem status
        if 'line_record' in update:
            self.datastore.anchor_line_record[self.anchor_num].insertList(np.array(update['line_record']))

            # this is the event that is set when *any* anchor sends a line record.
            # used by the position estimator to immedately recalculate the hang point
            self.datastore.anchor_line_record_event.set()

            # this event is set only for this specific anchor
            # it is used to detect an un-responsive state.
            self.line_record_receipt.set() 

        if 'last_raw_encoder' in update:
            self.last_raw_encoder = update['last_raw_encoder']

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)

        for detection in detections:

            if detection['n'] == "origin":
                # save all the detections of the origin for later analysis
                self.origin_poses.append(pose_from_det(detection))
                if len(self.origin_poses) > max_origin_detections:
                    self.origin_poses.pop(0)

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
                self.to_ui_q.put({'gantry_observation': position})
                if self.save_raw:
                    self.raw_gant_poses.append(pose_from_det(detection))

            if detection['n'] in ['gamepad', 'hamper', 'trash']:
                pose = np.array(compose_poses([
                    self.anchor_pose,
                    model_constants.anchor_camera, # constant
                    pose_from_det(detection), # the pose obtained just now
                ]))
                position = pose.reshape(6)[3:]
                # save the position of this object for use in various planning tasks.
                self.to_ob_q.put({'avg_named_pos': (detection['n'], position)})

    async def send_config(self):
        config = Config()
        anchor_config_vars = config.vars_for_anchor(self.anchor_num)
        if len(anchor_config_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': anchor_config_vars}))

        # Arm the unresponsiveness safety check
        self.safety_task = asyncio.create_task(self.safety_monitor())

    async def safety_monitor(self):
        """Notifies observer if this anchor stops sending line record updates for some time"""
        TIMEOUT=1 # seconds
        last_update = time.time()
        while self.connected:
            try:
                await asyncio.wait_for(self.line_record_receipt.wait(), TIMEOUT)
                # if you see the event within the timeout, all is well, clear it and wait again
                self.line_record_receipt.clear()
                last_update = time.time()
            except TimeoutError:
                try:
                    latency = await asyncio.wait_for(websocket.ping(), TIMEOUT)
                    # if the pong arrives, everything is fine, false alarm, resume the loop
                    # some hiccup on the server raspi made it unable to send anything for some time but it's not down.
                    continue
                except (ConnectionClosedError, TimeoutError):
                    # it's no longer running, either because it lost power, or the server crashed.
                    print(f'Anchor {self.anchor_num} has not sent a line record update in {time.time() - last_update} seconds.')
                    # immediately trigger the "abnormal shutdown" return from the connect_websocket task
                    await self.websocket.close(code=1101)


if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_ob_q = Queue()
    to_ui_q.cancel_join_thread()
    to_ob_q.cancel_join_thread()

    async def main():
        ac = RaspiAnchorClient("127.0.0.1", 0, datastore, to_ui_q, None)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), ac.shutdown_sync)
        await ac.startup()
    asyncio.run(main())
