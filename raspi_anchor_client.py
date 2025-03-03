
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

# number of origin detections to average
max_origin_detections = 10
video_port = 8888
websocket_port = 8765

def pose_from_det(det):
    return (np.array(det['r'], dtype=float), np.array(det['t'], dtype=float))

# this client is designed for the raspberri pi based anchor
class RaspiAnchorClient:
    def __init__(self, address, anchor_num, datastore, to_ui_q, to_pe_q, pool):
        self.address = address
        self.origin_poses = []
        self.anchor_num = anchor_num # which anchor are we connected to
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
        self.websocket = None
        self.connected = False  # status of connection to websocket
        self.receive_task = None  # Task for receiving messages from websocket
        self.video_task = None  # Task for streaming video
        self.calibration_mode = False # true is pose calibration mode.
        self.frame_times = {}
        self.pool = pool
        try:
            # read calibration data from file
            saved_info = np.load('anchor_pose_%i.npz' % self.anchor_num)
            self.anchor_pose = tuple(saved_info['pose'].astype(float))
            print(f"Read pose of anchor {self.anchor_num} from file: {self.anchor_pose}")
            self.to_ui_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})
            self.to_pe_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})
        except FileNotFoundError:
            self.anchor_pose = (np.array([0,0,0]), np.array([0,0,0]))

        # to help with a loop that does the same thing four times in handle_detections
        # name, offset, datastore
        self.arucos = [
            ('gripper_front', model_constants.gripper_aruco_front_inv, datastore.gripper_pose),
            ('gripper_back', model_constants.gripper_aruco_back_inv, datastore.gripper_pose),
            ('gantry_front', model_constants.gantry_aruco_front_inv, datastore.gantry_pose),
            ('gantry_back', model_constants.gantry_aruco_back_inv, datastore.gantry_pose),
        ]

    def receive_video(self):
        # don't connect too early or you will be rejected
        time.sleep(6)
        video_uri = f'tcp://{self.address}:{video_port}'
        print(f'Connecting to {video_uri}')
        cap = cv2.VideoCapture(video_uri)
        print(cap)
        while self.connected:
            ret, frame = cap.read()
            if ret:
                fnum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                try:
                    timestamp = self.frame_times[fnum]
                    del self.frame_times[fnum]
                except KeyError:
                    print('received a frame without knowing when it was captured')
                    continue
                self.pool.apply_async(locate_markers, (frame,), callback=partial(self.handle_detections, timestamp=timestamp))

    def calibrate_pose(self):
        print(f"writing 'anchor_pose_{self.anchor_num}.npz'")
        np.savez(f'anchor_pose_{self.anchor_num}.npz', pose = self.anchor_pose)
        self.to_pe_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})

    def handle_frame_times(self, frame_time_list):
        for ft in frame_time_list:
            # this item represents the time that rpicam-vid captured the frame with the given number.
            # we need to know this for when we get frames from the stream
            if len(self.frame_times) > 500:
                print('How did we miss 500 frames? Video task crashed?')
                self.shutdown()
            self.frame_times[int(ft['fnum'])] = float(ft['time'])

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the server
        """
        if self.calibration_mode:
            for detection in detections:
                # print(f"Name: {detection['n']}")
                # print(f"Timestamp: {detection['s']}")
                # print(f"Rotation Vector: {detection['r']}")
                # print(f"Translation Vector: {detection['t']}")
                # sys.stdout.flush()

                if detection['n'] == "origin":
                    print(f'detected origin {detection}')
                    self.origin_poses.append(pose_from_det(detection))
                    if len(self.origin_poses) > max_origin_detections:
                        self.origin_poses.pop(0)

                    # recalculate the pose of the connected anchor from recent origin detections
                    # anchor_cam_pose = invert_pose(average_pose(self.origin_poses))
                    # self.anchor_pose = compose_poses([anchor_cam_pose, model_constants.anchor_cam_inv])

                    self.anchor_pose = invert_pose(compose_poses([model_constants.anchor_camera, average_pose(self.origin_poses)]))

                    # show real time updates of this process on the UI
                    self.to_ui_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})
        else:
            for detection in detections:
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.
                print(f'detection {detection}')
                for name, offset, dest  in self.arucos:
                    if detection['n'] == name:
                        # you have the pose of gripper_front relative to a particular anchor camera
                        # Anchor is relative to the origin
                        # anchor camera is relative to anchor
                        # gripper_front is relative to anchor camera
                        # gripper is relative to gripper_front
                        # gripper_grommet is relative to gripper
                        pose = np.array(compose_poses([
                            self.anchor_pose, # obtained from calibration
                            model_constants.anchor_camera, # constant
                            pose_from_det(detection), # the pose obtained just now
                            offset, # constant
                        ]))
                        dest.insert(np.concatenate([[timestamp], pose.reshape(6)]))
                        print(f'Inserted pose in datastore name={name} t={detection["s"]}, pose={pose}')

    async def connect_websocket(self):
        # main client loop
        ws_uri = f"ws://{self.address}:{websocket_port}"
        print(f"Connecting to anchor {self.anchor_num} at {ws_uri}...")
        try:
            # connect() can be used as an infinite asynchronous iterator to reconnect automatically on errors
            async for websocket in websockets.connect(ws_uri, max_size=None, open_timeout=10):
                # try:
                self.connected = True
                print(f"Connected to anchor {self.anchor_num} at {ws_uri}.")
                await self.receive_loop(websocket)
                # except websockets.exceptions.ConnectionClosed:
                #     print(f"Connection closed")
                #     self.connected = False
                #     continue
        except asyncio.exceptions.CancelledError:
            print("Cancelling connection")
            return

    async def receive_loop(self, websocket):
        self.to_ui_q.put({'connection_status': {
            'anchor_num': self.anchor_num,
            'status': 'Online', # user visible string
        }})
        # loop of a single websocket connection.
        # save a reference to this for send_anchor_commands_async
        self.websocket = websocket
        # just could not make asyncio deal with this, so I used threading. hey it works, go figure
        vid_thread = threading.Thread(target=self.receive_video)
        vid_thread.start()
        # Loop until disconnected
        while self.connected:
            try:
                message = await websocket.recv()
                # print(f'received message of length {len(message)}')
                update = json.loads(message)
                if 'line_record' in update and not self.calibration_mode:
                    self.datastore.anchor_line_record[self.anchor_num].insertList(update['line_record'])
                if 'frames' in update:
                    self.handle_frame_times(update['frames'])

            except Exception as e:
                # don't catch websockets.exceptions.ConnectionClosedOK because we want it to trip the infinite generator in websockets.connect
                # so it will stop retrying.
                print(f"Connection to anchor {self.anchor_num} closed.")
                self.connected = False
                self.websocket = None
                self.to_ui_q.put({'connection_status': {
                    'anchor_num': self.anchor_num,
                    'status': 'Offline', # user visible string
                }})
                raise e
                break
        vid_thread.join()

    async def send_commands(self, update):
        if self.connected:
            await self.websocket.send(json.dumps(update))
        # just discard the update if not connected.

    async def slow_stop_spool(self):
        await self.send_commands({'length_plan' : []})

    async def startup(self):
        self.ct = asyncio.create_task(self.connect_websocket())
        await self.ct

    def shutdown(self):
        # this might get called twice
        print("\nWait for client shutdown")
        if self.connected:
            self.connected = False
            if self.websocket:
                asyncio.create_task(self.websocket.close())
        else:
            self.ct.cancel()

if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    to_ui_q.cancel_join_thread()
    to_pe_q.cancel_join_thread()
    to_ob_q.cancel_join_thread()

    async def main():
        ac = RaspiAnchorClient("127.0.0.1", 0, datastore, to_ui_q, to_pe_q, None)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), ac.shutdown)
        await ac.startup()
    asyncio.run(main())
