"""
A Lerobot Robot subclass for Stringman.
Connectes to the observer process and talks to it.
To keep concerns seperated I'm not making the AsyncObserver itself a subclass of Robot since it's already very complex
and uses service discovery to automatically connect to robot components.
"""

from functools import cached_property
from typing import Any

from lerobot.robots import Robot
from stringman_pilot_config import StringmanConfig

class StringmanPilotRobot(Robot):
    config_class = StringmanConfig
    name = "stringman"

    def __init__(self, config: StringmanConfig):
        super().__init__(config)
        # init object needed to talk to observer process
        self.channel = None

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        # lerobot assumes all features are either joints (float) or images (speicified as a tuple of width, height, channels)
        # here I have place all the properties we can command of the robot, even if they are not strictly motor joints.
        return { 
            "gantry_pos_x": float,
            "gantry_pos_y": float,
            "gantry_pos_z": float,
            "winch_length": float,
            "finger_angle": float,
        }

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "anchor_0": (1920, 1080, 3),
            "anchor_1": (1920, 1080, 3),
            "anchor_2": (1920, 1080, 3),
            "anchor_3": (1920, 1080, 3),
            "gripper_camera": (1920, 1080, 3),
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
        return self.channel is not None

    def connect(self, calibrate: bool = True) -> None:
        self.channel = 1234
        self.configure()

    def disconnect(self) -> None:
        self.channel = None

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

        # Read all observables
        obs_dict = {}
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

        # return the action that was actually sent
        return action