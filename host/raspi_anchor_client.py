
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

# this client is designed for the raspberri pi based anchor
class RaspiAnchorClient:
    def __init__(self, anchor_num, datastore, to_ui_q, to_pe_q):
        self.origin_detections = []
        self.anchor_num = anchor_num # which anchor are we connected to
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
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

    def connect_websocket(self, url):


# async def send_control(control_signal):
#     uri = "ws://<raspberry_pi_ip>:8765"  # Replace with Pi's IP
#     async with websockets.connect(uri) as websocket:
#         await websocket.send(json.dumps(control_signal))  # Send message
#         response = await websocket.recv()  # Receive response
#         print(f"Response: {response}")

# # Example usage:
# asyncio.run(send_control({"motor1": 255, "motor2": 128}))
# asyncio.run(send_control({"led_on": True}))