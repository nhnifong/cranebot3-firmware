
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

# maximum number of origin detections to keep
max_origin_detections = 40
video_port = 8888
websocket_port = 8765

def pose_from_det(det):
    return (np.array(det['r']), np.array(det['t']))

# this client is designed for the raspberri pi based anchor
class RaspiAnchorClient:
    def __init__(self, address, anchor_num, datastore, to_ui_q, to_pe_q):
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
        self.calibration_mode = True
        try:
            # read calibration data from file
            saved_info = np.load('anchor_pose_%i' % self.anchor_num)
            self.anchor_pose = tuple(saved_info['pose'])
            self.to_ui_q.put({'anchor_pose': (self.anchor_num, pose)})
            self.to_pe_q.put({'anchor_pose': (self.anchor_num, pose)})
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

    def calibrate_pose(self):
        np.savez('anchor_pose_%i' % self.anchor_num, pose = self.anchor_pose)

    def handle_detections(self, detections):
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
                    self.origin_poses.append(pose_from_det(detection))
                    if len(self.origin_poses) > max_origin_detections:
                        self.origin_poses.pop(0)
                    print()

                    # recalculate the pose of the connected anchor from recent origin detections
                    anchor_cam_pose = invert_pose(average_pose(self.origin_poses))
                    self.anchor_pose = compose_poses([anchor_cam_pose, invert_pose(model_constants.gripper_camera)])
                    print(f'anchor {self.anchor_num} pose {pose_from_det(detection)}')
                    # show real time updates of this process on the UI
                    self.to_ui_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})
                    self.to_pe_q.put({'anchor_pose': (self.anchor_num, self.anchor_pose)})
        else:
            for detection in detections:
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.

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
                        dest.insert(np.concatenate([[det['s']], pose.reshape(6)]))
                        print(f'Inserted pose in datastore name={name} t={timestamp}, pose={pose}')

    async def connect_websocket(self):
        # main client loop
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), self.shutdown)

        ws_uri = f"ws://{self.address}:{websocket_port}"
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

    async def receive_loop(self, websocket):
        # loop of a single websocket connection.
        # save a reference to this for send_anchor_commands_async
        self.websocket = websocket
        # Loop until disconnected
        while self.connected:
            try:
                message = await websocket.recv()
                print(f'received message of length {len(message)}')
                data = json.loads(message)
                if 'line_record' in data and not self.calibration_mode:
                    self.datastore.anchor_line_record[self.anchor_num].insertList(data['line_record'])
                if 'detections' in data:
                    self.handle_detections(data['detections'])
            except Exception as e:
                # don't catch websockets.exceptions.ConnectionClosedOK because we want it to trip the infinite generator in websockets.connect
                # so it will stop retrying.
                print(f"Connection to anchor {self.anchor_num} closed.")
                self.connected = False
                self.websocket = None
                raise e
                break

    async def send_anchor_commands_async(self, update):
        if self.connected:
            await self.websocket.send(json.dumps(update))
        # just discard the update if not connected.

    def shutdown(self):
        # this might get called twice
        if self.connected:
            self.connected = False
            if self.websocket:
                asyncio.create_task(self.websocket.close())

if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    ac = RaspiAnchorClient("127.0.0.1", 0, datastore, to_ui_q, to_pe_q)
    asyncio.run(ac.connect_websocket())
