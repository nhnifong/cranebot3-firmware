"""
A Lerobot Robot subclass for Stringman (pilot launch version).
Connectes to the observer process and talks to it.
To keep concerns seperated I'm not making the AsyncObserver itself a subclass of Robot since it's already very complex
and uses service discovery to automatically connect to robot components.
"""

from functools import cached_property
from typing import Any
from dataclasses import dataclass, field

import numpy as np
import cv2
from lerobot.robots import Robot, RobotConfig
import grpc
import io
from .robot_control_service_pb2 import (
    GetObservationRequest, GetObservationResponse,
    TakeActionRequest, TakeActionResponse,
    GetGamepadActionRequest,
    GetEpisodeControlRequest, GetEpisodeControlResponse,
    Point3D,
)
from .robot_control_service_pb2_grpc import RobotControlServiceStub

@RobotConfig.register_subclass("stringman")
@dataclass
class StringmanConfig(RobotConfig):
    grpc_addr: str

IMAGE_SHAPE = (1080, 1920, 3)

def decode_image(jpeg_bytes):
    try:
        im = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert im is not None
        return im
    except:
        return np.zeros(IMAGE_SHAPE, dtype=np.uint8)

class StringmanPilotRobot(Robot):
    config_class = StringmanConfig
    name = "stringman"

    def __init__(self, config: StringmanConfig):
        super().__init__(config)
        self.channel_address = config.grpc_addr
        self.channel = None
        self.stub = None
        self.start_held = False

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        # lerobot assumes all features are either joints (float) or images (speicified as a tuple of width, height, channels)
        # here I have place all the properties we can command of the robot, even if they are not strictly motor joints.
        return { 
            "gantry_vel_x": float,
            "gantry_vel_y": float,
            "gantry_vel_z": float,
            "winch_line_speed": float,
            "finger_angle": float,
        }

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        # use only one anchor camera to keep latency high and training load lower.
        return {
            "anchor_camera": IMAGE_SHAPE,
            "gripper_camera": IMAGE_SHAPE,
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft,
            "gripper_imu_rot_x": float,
            "gripper_imu_rot_y": float,
            "gripper_imu_rot_z": float,
            "laser_rangefinder": float,
            "finger_pad_voltage": float,
        }

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.stub is not None

    def connect(self, calibrate: bool = True) -> None:
        print(f"Establishing gRPC connection to {self.channel_address}...")
        self.channel = grpc.insecure_channel(self.channel_address)
        self.stub = RobotControlServiceStub(self.channel)
        print("gRPC channel established and stub created.")

    def disconnect(self) -> None:
        if self.channel:
            print("Closing gRPC channel...")
            self.channel.close()
            self.channel = None
            self.stub = None
            print("gRPC channel closed.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        response: GetObservationResponse = self.stub.GetObservation(GetObservationRequest())
        obs_dict = {
            'gantry_vel_x': response.gantry_vel.x,
            'gantry_vel_y': response.gantry_vel.y,
            'gantry_vel_z': response.gantry_vel.z,
            "winch_line_speed": response.winch_line_speed,
            "finger_angle": response.finger_angle,
            "gripper_imu_rot_x": response.gripper_imu_rot.x,
            "gripper_imu_rot_y": response.gripper_imu_rot.y,
            "gripper_imu_rot_z": response.gripper_imu_rot.z,
            "laser_rangefinder": response.laser_rangefinder,
            "finger_pad_voltage": response.finger_pad_voltage,
            "anchor_camera": decode_image(response.anchor_camera),
            "gripper_camera": decode_image(response.gripper_camera),
        }
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        request = TakeActionRequest(
            gantry_vel=Point3D(x=action['gantry_vel_x'], y=action['gantry_vel_y'], z=action['gantry_vel_z']),
            winch_line_speed=action['winch_line_speed'],
            finger_angle=action['finger_angle'],
        )
        # Call the synchronous stub method
        response: TakeActionResponse = self.stub.TakeAction(request)

        # return the action that was actually taken
        return {
            "gantry_vel_x": float(response.gantry_vel.x),
            "gantry_vel_y": float(response.gantry_vel.y),
            "gantry_vel_z": float(response.gantry_vel.z),
            "winch_line_speed": float(response.winch_line_speed),
            "finger_angle": float(response.finger_angle),
        }

    def get_last_action(self):
        """
        Get the last action taken by the robot
        Not part of normal lerobot flow. I'm bypassing the teleoperator
        """
        response: TakeActionResponse = self.stub.GetGamepadAction(GetGamepadActionRequest())
        return {
            "gantry_vel_x": float(response.gantry_vel.x),
            "gantry_vel_y": float(response.gantry_vel.y),
            "gantry_vel_z": float(response.gantry_vel.z),
            "winch_line_speed": float(response.winch_line_speed),
            "finger_angle": float(response.finger_angle),
        }

    def get_episode_control_events(self):
        response: GetEpisodeControlResponse = self.stub.GetEpisodeControl(GetEpisodeControlRequest())
        events = {}
        for e in response.events:
            events[e] = True
        return events