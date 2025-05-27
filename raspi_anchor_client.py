
import asyncio
import os
import signal
import websockets
import time
import json
from cv_common import locate_markers, compose_poses, invert_pose, average_pose
import cv2
import numpy as np
import model_constants
from functools import partial
import threading
from config import Config

# number of origin detections to average
max_origin_detections = 25
video_port = 8888
websocket_port = 8765

# fastSAM parameters
# seconds between processing frames with fastSAM. there is no need need to run it on every frame, since 
# we are looking at a relatively static image.
sam_rate = 1.0 # per second
sam_confidence_cutoff = 0.75

def pose_from_det(det):
    return (np.array(det['r'], dtype=float), np.array(det['t'], dtype=float))

# the genertic client for a raspberri pi based robot component
class ComponentClient:
    def __init__(self, address, datastore, to_ui_q, to_ob_q, pool, stat):
        self.address = address
        self.origin_poses = []
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_ob_q = to_ob_q
        self.websocket = None
        self.connected = False  # status of connection to websocket
        self.receive_task = None  # Task for receiving messages from websocket
        self.video_task = None  # Task for streaming video
        self.frame_times = {}
        self.pool = pool
        self.stat = stat
        self.shape_tracker = None
        self.last_gantry_frame_coords = None
        self.ct = None # task to connect to websocket

        # todo: receive a command in observer that will set this value
        self.sendPreviewToUi = False

    def receive_video(self):
        video_uri = f'tcp://{self.address}:{video_port}'
        print(f'Connecting to {video_uri}')
        cap = cv2.VideoCapture(video_uri)
        if not cap.isOpened():
            print('no video stream available')
            self.conn_status['video'] = False
            self.to_ui_q.put({'connection_status': self.conn_status})
            return
        print(f'video connection successful {cap}')
        self.conn_status['video'] = True
        self.notify_video = True
        lastSam = time.time()
        while self.connected:
            last_time = time.time()
            ret, frame = cap.read() # blocking call, that's why we're in a thread.
            if ret:

                # send frame to shape tracker and ui
                if self.shape_tracker is not None and self.anchor_num is not None: # skip gripper for now:

                    # send one frame per second to the fastSAM model
                    if time.time() > lastSam + self.shape_tracker.preferred_delay:
                        lastSam = time.time()
                        if not self.connected:
                            return
                        # self.shape_tracker.processFrame(self.anchor_num, frame)
                        
                        if self.sendPreviewToUi:
                            # send frame to UI
                            preview = cv2.flip(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), None, fx=0.25, fy=0.25), 0)
                            self.to_ui_q.put({'preview_image': {'anchor_num':self.anchor_num, 'image':preview}})

                # determine the timestamp of when the frame was captured by looking it up in the self frame_times map
                fnum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                try:
                    timestamp = self.frame_times[fnum]
                    del self.frame_times[fnum]
                    now = time.time()
                    self.stat.latency.append(now - timestamp)
                    self.stat.framerate.append(1/(now - last_time))
                    last_time = now
                except KeyError:
                    # print(f'received a frame without knowing when it was captured')
                    continue

                # send frame to aruco detector
                try:
                    if not self.connected:
                        return
                    if self.stat.pending_frames_in_pool < 60:
                        self.stat.pending_frames_in_pool += 1
                        self.pool.apply_async(locate_markers, (frame,), callback=partial(self.handle_detections, timestamp=timestamp))
                    else:
                        print(f'Dropping frame because there are already too many pending.')
                except ValueError:
                    return # the pool is not running
                # self.handle_detections(locate_markers(frame), timestamp=timestamp)
            else:
                time.sleep(0.1)

    def handle_frame_times(self, frame_time_list):
        """
        Handle messages from the server contain information about when frames were captured.
        this info is not embedded in the video stream, we have to save and reassociate it later.
        """
        for ft in frame_time_list:
            # this item represents the time that rpicam-vid captured the frame with the given number.
            # we need to know this for when we get frames from the stream
            if len(self.frame_times) > 500:
                print('How did we miss 500 frames? Video task crashed?')
                raise RuntimeError("The websocket connection continued to receive info on frames, but the video thread is not consuming them, and got too far behind.")
            self.frame_times[int(ft['fnum'])] = float(ft['time'])

        # do this here because we seemingly can't do it in receive_video
        if self.notify_video:
            self.to_ui_q.put({'connection_status': self.conn_status})
            self.notify_video = False

    async def connect_websocket(self):
        # main client loop
        self.conn_status['websocket'] = 1
        self.conn_status['video'] = False
        self.to_ui_q.put({'connection_status': self.conn_status})
        ws_uri = f"ws://{self.address}:{websocket_port}"
        print(f"Connecting to {ws_uri}...")
        try:
            # connect() can be used as an infinite asynchronous iterator to reconnect automatically on errors
            # It re-raises any error which it would not retry on. Some are expected on normal disconnects.
            async for websocket in websockets.connect(ws_uri, max_size=None, open_timeout=10):
                self.connected = True
                print(f"Connected to {ws_uri}.")
                await self.receive_loop(websocket)
        except asyncio.exceptions.CancelledError:
            print("Cancelling connection")
            return
        except websockets.exceptions.ConnectionClosedOK:
            print("Client closing connection")
            return

    async def receive_loop(self, websocket):
        print('receive loop')
        self.conn_status['websocket'] = 2
        self.to_ui_q.put({'connection_status': self.conn_status})
        # loop of a single websocket connection.
        # save a reference to this for send_commands
        self.websocket = websocket
        self.notify_video = False
        # send configuration to robot component to override default.
        asyncio.create_task(self.send_config())
        vid_thread = None
        # Loop until disconnected
        while self.connected:
            try:
                message = await websocket.recv()
                # print(f'received message of length {len(message)}')
                update = json.loads(message)
                if 'frames' in update:
                    self.handle_frame_times(update['frames'])
                if 'video_ready' in update:
                    # TODO take another shot at making asyncio run this
                    vid_thread = threading.Thread(target=self.receive_video)
                    vid_thread.start()
                self.handle_update_from_ws(update)

            except Exception as e:
                # don't catch websockets.exceptions.ConnectionClosedOK here because we want it to trip the infinite generator in websockets.connect
                # so it will stop retrying. after it has the intended effect, websockets.connect will raise it again, so we catch it in 
                # connect_websocket
                print(f"Connection to {self.address} closed.")
                self.connected = False
                self.websocket = None
                self.conn_status['websocket'] = 0
                self.conn_status['video'] = False
                self.to_ui_q.put({'connection_status': self.conn_status})
                raise e
                break
        if vid_thread is not None:
            # vid_thread should stop because self.connected is False
            vid_thread.join()

    async def send_commands(self, update):
        if self.connected:
            x = json.dumps(update)
            print(f'send commands {x}')
            await self.websocket.send(x)
        # just discard the update if not connected.

    async def slow_stop_spool(self):
        # spool will decelerate at the rate allowed by the config file.
        # tracking mode will switch to 'speed'
        await self.send_commands({'aim_speed': 0})

    async def startup(self):
        self.ct = asyncio.create_task(self.connect_websocket())
        await self.ct

    async def shutdown(self):
        print("\nWait for client shutdown")
        if self.connected:
            self.connected = False
            if self.websocket:
                await self.websocket.close()
        elif self.ct:
            self.ct.cancel()

    def shutdown_sync(self):
        # this might get called twice
        print("\nWait for client shutdown")
        if self.connected:
            self.connected = False
            if self.websocket:
                asyncio.create_task(self.websocket.close())
        elif self.ct:
            self.ct.cancel()

class RaspiAnchorClient(ComponentClient):
    def __init__(self, address, anchor_num, datastore, to_ui_q, to_ob_q, pool, stat, shape_tracker):
        super().__init__(address, datastore, to_ui_q, to_ob_q, pool, stat)
        self.anchor_num = anchor_num # which anchor are we connected to
        self.conn_status = {'anchor_num': self.anchor_num}
        self.shape_tracker = shape_tracker

        config = Config()
        self.anchor_pose = config.anchors[anchor_num].pose
        if self.shape_tracker is not None:
            self.shape_tracker.setCameraPoses(self.anchor_num, compose_poses([self.anchor_pose, model_constants.anchor_camera]))
        self.to_ui_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})

    def handle_update_from_ws(self, update):
        # TODO if we are not regularly receiving line_record, display this as a server problem status
        if 'line_record' in update:
            self.datastore.anchor_line_record[self.anchor_num].insertList(update['line_record'])

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

            if detection['n'] == 'gantry_front':
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.
                # you have the pose of gantry_front relative to a particular anchor camera
                # convert it to a pose relative to the origin
                pose = np.array(compose_poses([
                    self.anchor_pose, # obtained from calibration
                    model_constants.anchor_camera, # constant
                    pose_from_det(detection), # the pose obtained just now
                    model_constants.gantry_aruco_front_inv, # constant
                ]))
                position = pose.reshape(6)[3:]
                self.datastore.gantry_pos.insert(np.concatenate([[timestamp], position])) # take only the position
                # print(f'Inserted gantry pose ts={timestamp}, pose={pose}')

                self.last_gantry_frame_coords = np.array(detection['t'], dtype=float)
                self.to_ui_q.put({'gantry_observation': position})
                    

    async def send_config(self):
        config = Config()
        anchor_config_vars = config.vars_for_anchor(self.anchor_num)
        if len(anchor_config_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': anchor_config_vars}))

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
