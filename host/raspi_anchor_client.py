
import asyncio
import websockets
import time
import json
from cv_common import locate_markers
import cv2
import numpy as np
from calibration import compose_poses, invert_pose, average_pose
import model_constants

# maximum number of origin detections to keep
max_origin_detections = 40
video_port = 8888
websocket_port = 8765


# this client is designed for the raspberri pi based anchor
class RaspiAnchorClient:
    def __init__(self, address, anchor_num, datastore, to_ui_q, to_pe_q):
        self.address = address
        self.origin_detections = []
        self.anchor_num = anchor_num # which anchor are we connected to
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
        self.websocket = None
        self.connected = False  # status of connection to websocket
        self.receive_task = None  # Task for receiving messages from websocket
        self.video_task = None  # Task for streaming video
        try:
            # read calibration data from file
            saved_info = np.load('anchor_pose_%i' % self.anchor_num)
            self.anchor_pose = tuple(saved_info['pose'])
        except FileNotFoundError:
            self.anchor_pose = (np.array([0,0,0]), np.array([0,0,0]))

        # to help with a loop that does the same thing four times in handle_image
        # name, offset, datastore
        self.arucos = [
            ('gripper_front', model_constants.gripper_aruco_front_inv, datastore.gripper_pose),
            ('gripper_back', model_constants.gripper_aruco_back_inv, datastore.gripper_pose),
            ('gantry_front', model_constants.gantry_aruco_front_inv, datastore.gantry_pose),
            ('gantry_back', model_constants.gantry_aruco_back_inv, datastore.gantry_pose),
        ]

    def calibrate_pose(self):
        # recalculate the pose of the connected anchor from recent origin detections
        anchor_cam_pose = [invert_pose(*average_pose(det)) for det in self.origin_detections]
        self.anchor_pose = compose_poses([anchor_cam_pose, invert_pose(model_constants.gripper_camera)])
        np.savez('anchor_pose_%i' % self.anchor_num, pose = pose)
        self.to_ui_q.put({'anchor_pose': (self.anchor_num, pose)})
        self.to_pe_q.put({'anchor_pose': (self.anchor_num, pose)})


    def handle_image(self, frame):
        """
        handle a single image from the stream
        """
        # We don't have a timestamp in the stream, so we have to assume the local time minus some for latency
        timestamp = time.time() - 0.2
        # within this scope, interpret the symbol calibration_mode as referring to the global calibration_mode
        global calibration_mode
        if calibration_mode:
            for detection in locate_markers(frame):
                print(f"Found board: {detection.name}")
                print(f"Timestamp: {timestamp}")
                print(f"Rotation Vector: {detection.rvec}")
                print(f"Translation Vector: {detection.tvec}")
                # sys.stdout.flush()

                if detection.name == "origin":
                    origin_detections.append(detection)
                    if len(origin_detections) > max_origin_detections:
                        origin_detections.pop(0)
        else:
            for detection in locate_markers(frame):
                # rotate and translate to where that object's origin would be
                # given the position and rotation of the camera that made this observation (relative to the origin)
                # store the time and that position in the appropriate measurement array in observer.

                for name, offset, dest  in self.arucos:
                    if detection.name == name:
                        # you have the pose of gripper_front relative to a particular anchor camera
                        # Anchor is relative to the origin
                        # anchor camera is relative to anchor
                        # gripper_front is relative to anchor camera
                        # gripper is relative to gripper_front
                        # gripper_grommet is relative to gripper
                        gripper_global_pose = np.array(compose_poses([
                            self.anchor_pose, # obtained from calibration
                            model_constants.anchor_camera, # constant
                            (detection.rotation, detection.translation), # the pose obtained just now
                            offset, # constant
                        ]))
                        dest.insert(np.concatenate([[timestamp], gripper_global_pose.reshape(6)]))

    def connect_video_stream(self, url):
        """
        Streams video from the raspberry pi video module 3
        blocking
        """
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"Error opening video stream: {stream_url}")
            return
        print("Connected to %s" % stream_url)
        while True:
            ret, frame = cap.read()
            if not ret:
                return
            if frame is not None:
                self.handle_image(frame)
        print("Stream ended from %s" % stream_url)
        cap.release()

    async def connect_websocket(self, ws_uri):
        try:
            self.websocket = await websockets.connect(ws_uri)
            self.connected = True
            print(f"Connected to anchor {self.anchor_num} at {ws_uri}.")
            self.receive_task = asyncio.create_task(self._receive_loop()) #Start receiving messages
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.websocket = None
            self.connected = False
            return False

    async def _receive_loop(self): #Private method for the receive loop
        while self.connected: #Loop until disconnected
            try:
                await self._send_anchor_commands_async({"foo": 123})
                await asyncio.sleep(1)
                message = await self.websocket.recv()
                data = json.loads(message)
                print(f"Received: {data}")
            except websockets.exceptions.ConnectionClosedOK:
                print("Connection closed by server.")
                self.connected = False
                break
            except Exception as e:
                print(f"Receive error: {e}")
                self.connected = False
                break

    def send_anchor_commands(self, something):
        """Sends commands to the server (non-blocking)."""
        if not self.connected:
            raise ConnectionError("Not connected to server.")
        asyncio.create_task(self._send_array_async(something))  # Send in a separate task

    async def _send_anchor_commands_async(self, something):
        try:
            await self.websocket.send(json.dumps(something))
        except Exception as e:
            print(f"Error sending anchor commands: {e}")
            self.connected = False #If there is an error, the connection is probably down

    def close_websocket(self):
        if self.connected:
            asyncio.create_task(self._close_websocket_async())

    async def close_all(self):
        if self.receive_task:
            self.receive_task.cancel()  # Stop receiving messages
            try:
                await self.receive_task #Wait for the task to finish cancelling
            except asyncio.CancelledError:
                pass #Expected
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.connected = False
        if self.video_task:
            self.video_task.cancel()  # Stop receiving messages
            try:
                await self.video_task
            except asyncio.CancelledError:
                pass #Expected
        print(f"Connection to anchor {self.anchor_num} at {ws_uri} closed.")

    async def connect_all(self):
        """
        Connect to both the video stream and the websocket of this anchor
        """
        # theres a different port for the video and the websocket
        # stream_url = f"tcp://{self.address}:{video_port}"  # Construct the URL
        # self.video_task = asyncio.create_task(asyncio.to_thread(self.connect_video_stream, args=(stream_url, )))
        # self.connect_video_stream(stream_url)
        # video_thread = threading.Thread(target=ac.connect_video_stream, args=(stream_url,), daemon=True)
        # video_thread.start()

        # a websocket connection for the control
        ws_uri = f"ws://{self.address}:{websocket_port}"
        asyncio.create_task(self.connect_websocket(ws_uri))