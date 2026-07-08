"""
A Lerobot Robot subclass for Stringman robot with arp gripper.
Connects to the observer process for control and telemetry.
"""

from functools import cached_property
from typing import Any
from dataclasses import dataclass
import numpy as np
import cv2
import argparse
import threading
import websockets.exceptions
from websockets.sync.client import connect as websocket_connect_sync
import time
from urllib.parse import urlparse
import av
from huggingface_hub import repo_exists, get_token, whoami
from huggingface_hub.errors import HfHubHTTPError
import os
import shutil
import signal
import sys
import json
import importlib.metadata

from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common

from lerobot.robots import Robot, RobotConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
try:
    # lerobot 0.5.2
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
except ImportError:
    # lerobot 0.5.1
    from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import OBS_STR, ACTION
# from lerobot.utils.visualization_utils import log_rerun_data, init_rerun
# from lerobot.utils.utils import log_say

# --- Recording Configuration ---
EPISODE_MAX_TIME_SEC = 600
FPS = 30
TASK_DESCRIPTION = "Pick up the item"
NUM_BUFFERS = 3


IMG_RES = 384

CHECKPOINT_EVERY = 10

# feed_number -> observation key name
_FEED_NAMES: dict[int, str] = {
    0: "gripper_camera",
    1: "anchor_camera_0",
    2: "anchor_camera_1",
    3: "overhead_camera",
}

# camera_mode -> {feed_number: (width, height)}
_CAMERA_MODES: dict[str, dict[int, tuple[int, int]]] = {
    "gripper_224":       {0: (224, 224)},
    "gripper_384":       {0: (384, 384)},
    "gripper_floor_224": {0: (224, 224), 3: (224, 224)},
    "gripper_floor_384": {0: (384, 384), 3: (384, 384)},
    "gripper_anchors_384": {0: (384, 384), 1: (384, 384), 2: (384, 384)},
    "all_square":        {0: (384, 384), 3: (512, 512), 1: (960, 544), 2: (960, 544)},
    "all":               {0: (684, 384), 3: (512, 512), 1: (960, 544), 2: (960, 544)},
}

# action_space -> ordered list of action component names.
# "gripper_vel" is the original 5-dim action space used by older policies/datasets.
# "dual_vel_contact" additionally commands velocity in the room frame of reference
# (fused with the gripper-frame velocity at send_action time, see _CONTROL_ACTION_ROLES),
# and reserves slots for "eventual contact position" and "episode end" predictions.
# Those last three are not knowable live during recording/teleop and are recorded as
# zero placeholders; a future post-processing script can fill them in retroactively.
_ACTION_SPACES: dict[str, list[str]] = {
    "gripper_vel": ["vel_x", "vel_y", "vel_z", "wrist_speed", "finger_speed"],
    "gripper_vel_contact": [
        "vel_x", "vel_y", "vel_z",
        "wrist_speed", "finger_speed",
        "contact_vec_x", "contact_vec_y", "contact_vec_z",
        "episode_end",
    ],
    "dual_vel_contact": [
        "vel_x", "vel_y", "vel_z",
        "room_vel_x", "room_vel_y",
        "wrist_speed", "finger_speed",
        "contact_vec_x", "contact_vec_y", "contact_vec_z",
        "episode_end",
    ],
}
DEFAULT_ACTION_SPACE = "dual_vel_contact"

# action component name -> control role used by send_action to build a single CombinedMove.
# Action components not listed here (e.g. contact_vec_*, episode_end) are auxiliary
# prediction targets and are ignored for control purposes.
_CONTROL_ACTION_ROLES: dict[str, str] = {
    "vel_x": "gripper_x",
    "vel_y": "gripper_y",
    "vel_z": "z",
    "room_vel_x": "room_x",
    "room_vel_y": "room_y",
    "wrist_speed": "wrist_speed",
    "finger_speed": "finger_speed",
}

# Debugging switches: at eval time, force the gripper-frame or room-frame xy velocity
# components to zero so their individual contribution to visual servoing can be assessed.
IGNORE_GRIPPER_FRAME_VEL = False
IGNORE_ROOM_FRAME_VEL = False


def rotate_vector(vec, rad):
    """Rotates a 2D vector [x, y] by a given angle in radians."""
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    return np.array([
        vec[0] * cos_a - vec[1] * sin_a,
        vec[0] * sin_a + vec[1] * cos_a,
    ])


def camera_mode_from_features(features: dict) -> str:
    """Infer a camera_mode name from a dataset's feature dict (e.g. dataset.meta.features)."""
    found: dict[int, tuple[int, int]] = {}
    for feed_num, name in _FEED_NAMES.items():
        key = f"observation.images.{name}"
        if key in features:
            shape = features[key]["shape"]  # [height, width, channels]
            found[feed_num] = (shape[1], shape[0])

    for mode, spec in _CAMERA_MODES.items():
        if spec == found:
            return mode
    raise ValueError(f"No known camera_mode matches camera feature set {found}")


def action_space_from_features(features: dict) -> str:
    """Infer an action_space name from a dataset's feature dict (e.g. dataset.meta.features)."""
    names = list(features["action"]["names"])
    for space, action_names in _ACTION_SPACES.items():
        if action_names == names:
            return space
    raise ValueError(f"No known action_space matches action names {names}")


def describe_session_spaces(camera_mode: str, action_space: str, observation_features: dict, action_features: dict) -> str:
    """Human-readable description of the camera/action/observation configuration for a session."""
    lines = [
        f"camera_mode: {camera_mode} -> {_CAMERA_MODES[camera_mode]}",
        f"action_space: {action_space} -> {_ACTION_SPACES[action_space]}",
        f"observation_features ({len(observation_features)}): {list(observation_features)}",
        f"action_features ({len(action_features)}): {list(action_features)}",
    ]
    return "\n".join(lines)


@RobotConfig.register_subclass("stringman")
@dataclass
class StringmanConfig(RobotConfig):
    uri: str
    remote_stream_token: str | None = None
    camera_mode: str = "all"
    action_space: str = DEFAULT_ACTION_SPACE

def decode_image(jpeg_bytes):
    try:
        im = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert im is not None
        return im
    except:
        return np.zeros((IMG_RES, IMG_RES, 3), dtype=np.uint8)

class StringmanLeRobot(Robot):
    config_class = StringmanConfig
    name = "stringman"

    def __init__(self, config: StringmanConfig, events):
        super().__init__(config)
        self.address = config.uri

        # when connecting to remote streams, we will need the token of an authorized user in order to access the robot's telemetry
        self.remote_stream_token = config.remote_stream_token

        self.websocket = None
        
        self.last_commanded_vel = np.zeros(3)
        self.last_observed_vel = np.zeros(3)
        
        self.last_wrist_speed = 0.0
        self.last_finger_speed = 0.0

        self.last_finger_angle = 0.0
        self.last_pressure = 0.0
        self.last_range = 0.0
        self.last_wrist_angle = 0.0
        self.last_target_force = 0.0

        self.last_gripper_pos = np.zeros(3, dtype=float)
        self.last_gripper_rot_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)

        self.last_named_positions = {
            "hamper":           np.zeros(3),
            "toybox":           np.zeros(3),
            "trashcan":         np.zeros(3),
            "gamepad":          np.zeros(3),
            "parking_location": np.zeros(3),
        }
        self.last_swing_cancellation_on = 0.0
        self.last_tensions = np.zeros(4, dtype=float)
        self.last_gantry_pos = np.zeros(3, dtype=float)
        self.last_visual_pos = np.zeros(3, dtype=float)
        self.last_hang_pos = np.zeros(3, dtype=float)

        self.last_task_description = TASK_DESCRIPTION

        self.last_spin = 0.0

        self.events = events

        if config.camera_mode not in _CAMERA_MODES:
            raise ValueError(f"Unknown camera_mode '{config.camera_mode}'. Valid: {list(_CAMERA_MODES)}")
        # {feed_number: (width, height)}
        self.camera_specs: dict[int, tuple[int, int]] = _CAMERA_MODES[config.camera_mode]

        if config.action_space not in _ACTION_SPACES:
            raise ValueError(f"Unknown action_space '{config.action_space}'. Valid: {list(_ACTION_SPACES)}")
        self.action_space = config.action_space

        self.camera_locks = {f: threading.Lock() for f in self.camera_specs}
        self.video_threads = {}
        self.stop_video_events = {f: threading.Event() for f in self.camera_specs}

        self.last_images = {
            f: np.zeros((h, w, 3), dtype=np.uint8)
            for f, (w, h) in self.camera_specs.items()
        }

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "vel_x": float,
            "vel_y": float,
            "vel_z": float,
            "wrist_speed": float,
            "finger_speed": float,
        }

    @cached_property
    def _action_ft(self) -> dict[str, type]:
        return {name: float for name in _ACTION_SPACES[self.action_space]}

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            _FEED_NAMES[f]: (h, w, 3)
            for f, (w, h) in self.camera_specs.items()
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft,
            "gripper_pos_x": float,
            "gripper_pos_y": float,
            "gripper_pos_z": float,
            "gripper_rot_0": float,
            "gripper_rot_1": float,
            "gripper_rot_2": float,
            "gripper_rot_3": float,
            "gripper_rot_4": float,
            "gripper_rot_5": float,
            "spin": float,

            "finger_angle": float,
            "laser_rangefinder": float,
            "finger_pressure": float,
            "wrist_angle": float,
            "target_force": float,

            "hamper_bearing": float,
            "hamper_distance": float,
            "toybox_bearing": float,
            "toybox_distance": float,
            "trashcan_bearing": float,
            "trashcan_distance": float,
            "gamepad_bearing": float,
            "gamepad_distance": float,
            "parking_location_bearing": float,
            "parking_location_distance": float,

            "swing_cancellation_on": float,

            "tension_0": float,
            "tension_1": float,
            "tension_2": float,
            "tension_3": float,

            "gantry_position_x": float,
            "gantry_position_y": float,
            "gantry_position_z": float,
            "visual_pos_x": float,
            "visual_pos_y": float,
            "visual_pos_z": float,
            "hang_pos_x": float,
            "hang_pos_y": float,
            "hang_pos_z": float,
        }

    @cached_property
    def action_features(self) -> dict:
        return self._action_ft

    @property
    def is_connected(self) -> bool:
        return self.websocket is not None

    def connect(self, calibrate: bool = True) -> None:
        receive_updates_thread = threading.Thread(target=self.connect_thread, daemon=True)
        receive_updates_thread.start()
        give_up = time.time()+5
        while self.websocket is None and time.time() < give_up:
            time.sleep(0.1)

    def connect_thread(self):
        print(f'Connecting to {self.address}')
        with websocket_connect_sync(self.address) as websocket:
            self.websocket = websocket
            print(f'Connected')
            for message in websocket:
                batch = telemetry.TelemetryBatchUpdate().parse(message)
                for item in batch.updates:
                    self.process_update(item)

    def process_update(self, item: telemetry.TelemetryItem):
        if item.pos_estimate is not None:
            self._handle_pos_estimate(item.pos_estimate)
        if item.pos_factors_debug is not None:
            self._handle_pos_factors(item.pos_factors_debug)
        if item.grip_sensors is not None:
            self._handle_grip_sensors(item.grip_sensors)
        if item.video_ready is not None:
            self._handle_video_ready(item.video_ready)
        if item.raw_commanded_vel is not None:
            self._handle_raw_commanded_vel(item.raw_commanded_vel)
        if item.last_commanded_grip is not None:
            self._handle_last_commanded_grip(item.last_commanded_grip)
        if item.episode_control is not None:
            self._handle_episode_control(item.episode_control)
        if item.named_position is not None:
            self._handle_named_position(item.named_position)
        if item.swing_cancellation_state is not None:
            self._handle_swing_cancellation(item.swing_cancellation_state)

    def _handle_pos_estimate(self, item: telemetry.PositionEstimate):
        self.last_observed_vel = tonp(item.gantry_velocity)

        if item.gantry_position:
            self.last_gantry_pos = tonp(item.gantry_position)

        for i, t in enumerate(item.tension[:4]):
            self.last_tensions[i] = t

        if item.gripper_pose:
            if item.gripper_pose.position:
                self.last_gripper_pos = tonp(item.gripper_pose.position)
            
            if item.gripper_pose.rotation:
                rvec = tonp(item.gripper_pose.rotation)
                R, _ = cv2.Rodrigues(rvec)
                self.last_gripper_rot_6d = R[:, :2].flatten()

    def _handle_grip_sensors(self, item: telemetry.GripperSensors):
        if item.range is not None:
            self.last_range = item.range
        if item.angle is not None:
            self.last_finger_angle = item.angle
        if item.pressure is not None:
            self.last_pressure = item.pressure
        if item.wrist is not None:
            self.last_wrist_angle = item.wrist
        if item.target_force is not None:
            self.last_target_force = item.target_force

    def _handle_pos_factors(self, item: telemetry.PositionFactors):
        if item.visual_pos:
            self.last_visual_pos = tonp(item.visual_pos)
        if item.hanging_pos:
            self.last_hang_pos = tonp(item.hanging_pos)
        if item.spin:
            self.last_spin = item.spin

    def _handle_named_position(self, item: telemetry.NamedObjectPosition):
        if item.name in self.last_named_positions and item.position is not None:
            self.last_named_positions[item.name] = tonp(item.position)

    def _handle_swing_cancellation(self, item: telemetry.SwingCancellationState):
        self.last_swing_cancellation_on = 1.0 if item.enabled else 0.0

    def _handle_video_ready(self, item: telemetry.VideoReady):
        feed_num = item.feed_number

        if feed_num not in self.camera_specs:
            return

        # Return if this feed's stream is already alive
        if feed_num in self.video_threads and self.video_threads[feed_num].is_alive():
            return

        # Handle stream URL selection based on the remote_stream_token flag
        if self.remote_stream_token is not None and item.stream_path:
            staging = ''
            if 'localhost' in self.address:
                host = 'localhost:8554'
            elif 'host.docker.internal' in self.address:
                host = 'host.docker.internal:8554'
            elif 'nf-site-monolith-staging' in self.address:
                host = 'media.neufangled.com:8554'
                staging = '&staging=1'
            else:
                host = 'media.neufangled.com:8554'
            url = f"rtsp://{host}/{item.stream_path}?ticket={self.remote_stream_token}{staging}"
        elif item.local_uri is not None:
            url = item.local_uri

        if url:
            parsed = urlparse(url)
            hostname = parsed.hostname if parsed.hostname else "localhost"
            is_local = hostname in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]

            print(f"Video ready (feed {feed_num}). Connecting to stream: {url} (Local: {is_local})")
            self.stop_video_events[feed_num].clear()
            self.video_threads[feed_num] = threading.Thread(
                target=self._video_stream_loop, 
                args=(url, is_local, feed_num), 
                daemon=True
            )
            self.video_threads[feed_num].start()
        else:
            print(f"Received VideoReady event for feed {feed_num} but could not determine stream URL.")

    def _video_stream_loop(self, stream_url, is_local, feed_num):
        print(f"Opening PyAV stream for feed {feed_num}: {stream_url}")
        
        options = {
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'fast': '1',
        }

        if stream_url.startswith('rtsp'):
            options['rtsp_transport'] = 'udp' if is_local else 'tcp'
            options['stimeout'] = '5000000'
        elif stream_url.startswith('http'):
            options['timeout'] = '5000000'

        try:
            container = av.open(stream_url, options=options)
        except av.error.HTTPUnauthorizedError:
            self.disconnect()
        
        stream = next(s for s in container.streams if s.type == 'video')
        stream.thread_type = "SLICE"
        
        try:
            for av_frame in container.decode(stream):
                if self.stop_video_events[feed_num].is_set():
                    break

                frame = av_frame.to_ndarray(format='rgb24')
                target_w, target_h = self.camera_specs[feed_num]
                if frame.shape[0] != target_h or frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))
                    
                with self.camera_locks[feed_num]:
                    self.last_images[feed_num] = frame
        except av.error.TimeoutError:
            return
        finally:
            if 'container' in locals():
                container.close()
            print(f"Stream {stream_url} closed (feed {feed_num})")

    def _handle_raw_commanded_vel(self, item: telemetry.CommandedVelocity):
        # trained on commanded velocity in the gripper image frame of reference
        self.last_commanded_vel = tonp(item.velocity)

    def _handle_last_commanded_grip(self, item: telemetry.CommandedGrip):
        self.last_wrist_speed = item.wrist_speed
        self.last_finger_speed = item.finger_speed

    def _handle_episode_control(self, item: common.EpisodeControl):
        print(f'EpisodeControl received by lerobot session {item}')
        if item.command == common.EpCommand.ABANDON:
            self.events['episode_abandon'] = True
        if item.command == common.EpCommand.END_RECORDING:
            self.events['end_recording'] = True
        if item.command == common.EpCommand.EVAL_START:
            self.events['start'] = True
        if item.command == common.EpCommand.EVAL_STOP:
            self.events['stop'] = True
        if item.prompt is not None:
            self.last_task_description = item.prompt

    def disconnect(self) -> None:
        print("Disconnecting")
        for i in self.stop_video_events:
            self.stop_video_events[i].set()
        
        if self.websocket is not None:
            self.websocket.close()
        self.websocket = None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

    def get_observation(self) -> dict[str, Any]:
        images = {}
        for feed_num in self.camera_specs:
            with self.camera_locks[feed_num]:
                images[feed_num] = self.last_images[feed_num].copy()


        # calculate bearing and distance to every target in gripper frame of reference
        targets = {}
        for name, pos in self.last_named_positions.items():
            delta = pos[:2] - self.last_gripper_pos[:2]
            room_angle = np.arctan2(delta[0], delta[1])
            bearing = (room_angle - self.last_spin + np.pi) % (2 * np.pi) - np.pi
            targets[name] = (bearing, float(np.linalg.norm(delta)))

        return {
            'vel_x': float(self.last_observed_vel[0]),
            'vel_y': float(self.last_observed_vel[1]),
            'vel_z': float(self.last_observed_vel[2]),
            
            "wrist_speed": float(self.last_wrist_speed),
            "finger_speed": float(self.last_finger_speed),

            "gripper_pos_x": float(self.last_gripper_pos[0]),
            "gripper_pos_y": float(self.last_gripper_pos[1]),
            "gripper_pos_z": float(self.last_gripper_pos[2]),
            
            "gripper_rot_0": float(self.last_gripper_rot_6d[0]),
            "gripper_rot_1": float(self.last_gripper_rot_6d[1]),
            "gripper_rot_2": float(self.last_gripper_rot_6d[2]),
            "gripper_rot_3": float(self.last_gripper_rot_6d[3]),
            "gripper_rot_4": float(self.last_gripper_rot_6d[4]),
            "gripper_rot_5": float(self.last_gripper_rot_6d[5]),
            "spin": float(self.last_spin),

            "finger_angle": float(self.last_finger_angle),
            "laser_rangefinder": float(self.last_range),
            "finger_pressure": float(self.last_pressure),

            "wrist_angle": float(self.last_wrist_angle),
            "target_force": float(self.last_target_force),

            "hamper_bearing":   float(targets["hamper"][0]),
            "hamper_distance":  float(targets["hamper"][1]),
            "toybox_bearing":   float(targets["toybox"][0]),
            "toybox_distance":  float(targets["toybox"][1]),
            "trashcan_bearing": float(targets["trashcan"][0]),
            "trashcan_distance":float(targets["trashcan"][1]),
            "gamepad_bearing":  float(targets["gamepad"][0]),
            "gamepad_distance": float(targets["gamepad"][1]),
            "parking_location_bearing":  float(targets["parking_location"][0]),
            "parking_location_distance": float(targets["parking_location"][1]),

            "swing_cancellation_on": float(self.last_swing_cancellation_on),

            "tension_0": float(self.last_tensions[0]),
            "tension_1": float(self.last_tensions[1]),
            "tension_2": float(self.last_tensions[2]),
            "tension_3": float(self.last_tensions[3]),

            "gantry_position_x": float(self.last_gantry_pos[0]),
            "gantry_position_y": float(self.last_gantry_pos[1]),
            "gantry_position_z": float(self.last_gantry_pos[2]),
            "visual_pos_x": float(self.last_visual_pos[0]),
            "visual_pos_y": float(self.last_visual_pos[1]),
            "visual_pos_z": float(self.last_visual_pos[2]),
            "hang_pos_x": float(self.last_hang_pos[0]),
            "hang_pos_y": float(self.last_hang_pos[1]),
            "hang_pos_z": float(self.last_hang_pos[2]),

            **{_FEED_NAMES[f]: img for f, img in images.items()},
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:

        # action['wrist_speed'] *= 30
        # action['finger_speed'] *= 30
        GAIN = 1.0 # for act model, 1.5 is a little better.

        # Fuse whichever control-relevant action components are present into a single
        # gripper-frame CombinedMove. Components not in _CONTROL_ACTION_ROLES (e.g.
        # contact_vec_*, episode_end) are auxiliary predictions and are ignored here.
        gripper_xy = np.zeros(2)
        z = 0.0
        wrist_speed = 0.0
        finger_speed = 0.0

        if not IGNORE_GRIPPER_FRAME_VEL:
            gripper_xy += [action.get('vel_x', 0.0), action.get('vel_y', 0.0)]
        z += action.get('vel_z', 0.0)

        if not IGNORE_ROOM_FRAME_VEL:
            room_xy = np.array([action.get('room_vel_x', 0.0), action.get('room_vel_y', 0.0)])
            gripper_xy += rotate_vector(room_xy, self.last_spin)

        wrist_speed += action.get('wrist_speed', 0.0)
        finger_speed += action.get('finger_speed', 0.0)

        batch = control.ControlBatchUpdate(
            robot_id="0",
            updates=[control.ControlItem(move=control.CombinedMove(
                direction=common.Vec3(
                    x=gripper_xy[0]*GAIN,
                    y=gripper_xy[1]*GAIN,
                    z=z*GAIN,
                ),
                finger_speed=finger_speed*GAIN*GAIN,
                wrist_speed=wrist_speed*GAIN,
                # speed=0.12,
                direction_is_in_gripper_frame=True,
            ))]
        )
        to_send = bytes(batch)
        if self.websocket and to_send:
            self.websocket.send(to_send)
        return action

    def send_session_status(self, status: common.LerobotSessionStatus):
        batch = control.ControlBatchUpdate(
            robot_id="0",
            updates=[control.ControlItem(episode_control=common.EpisodeControl(
                status=status,
            ))]
        )
        to_send = bytes(batch)
        if self.websocket and to_send:
            try:
                self.websocket.send(to_send)
            except websockets.exceptions.WebSocketException as e:
                # Status updates are best-effort telemetry; a dropped connection
                # must not abort cleanup/finalization or mask the real error.
                print(f"Failed to send session status (connection closed?): {e}", flush=True)

    def get_last_action(self):
        # last_commanded_vel is in the gripper's frame of reference; convert to room
        # frame for room_vel_x/y (gripper -> room, see observer.py _handle_movement).
        room_vel = rotate_vector(self.last_commanded_vel[:2], -self.last_spin)

        values = {
            "vel_x": self.last_commanded_vel[0],
            "vel_y": self.last_commanded_vel[1],
            "vel_z": self.last_commanded_vel[2],
            "room_vel_x": room_vel[0],
            "room_vel_y": room_vel[1],
            "wrist_speed": self.last_wrist_speed,
            "finger_speed": self.last_finger_speed,
            # Not knowable live; placeholders for a future post-processing script.
            "contact_vec_x": 0.0,
            "contact_vec_y": 0.0,
            "contact_vec_z": 0.0,
            "episode_end": 0.0,
        }
        return {name: values[name] for name in _ACTION_SPACES[self.action_space]}

@safe_stop_image_writer
def record_episode(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    max_episode_duration: int | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    print('ep started')
    while timestamp < max_episode_duration:
        start_loop_t = time.perf_counter()
        
        if events['end_recording']:
            break
        if events["stop"]:
            print('ep complete')
            events["stop"] = False
            break
        if events["episode_abandon"]: 
            print('ep abandon')
            break

        obs = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        
        action_sent = robot.get_last_action()
        action_frame = build_dataset_frame(dataset.features, action_sent, prefix=ACTION)

        frame = {**observation_frame, **action_frame, "task": robot.last_task_description}
        dataset.add_frame(frame)

        # if display_data:
        #     log_rerun_data(observation=obs, action=action_sent)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time = 1 / fps - dt_s
        if sleep_time > 0:
            time.sleep(sleep_time)
        timestamp = time.perf_counter() - start_episode_t

    print('ep finished')

def append_episode_metadata(dataset: LeRobotDataset, robot_id: str):
    try:
        nf_robot_version = importlib.metadata.version("nf_robot")
    except importlib.metadata.PackageNotFoundError:
        nf_robot_version = "unknown"
    entry = {
        "episode_index": dataset.num_episodes - 1,
        "robot_id": robot_id,
        "nf_robot_version": nf_robot_version,
    }
    path = dataset.root / "meta" / "episodes_extra.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def ensure_hf_auth():
    """Fail fast with a clear message if HuggingFace auth isn't set up.

    A recording session resumes or creates a dataset under the user's HF
    namespace and (when uploading) pushes to the hub, all of which need a valid
    token.
    """
    token = get_token()
    if token is None:
        raise RuntimeError(
            "Not logged in to HuggingFace. Run `hf auth login` before recording."
        )
    try:
        user = whoami(token)
    except HfHubHTTPError as e:
        raise RuntimeError(
            "HuggingFace token is invalid or expired. Run `hf auth login` to re-authenticate."
        ) from e
    print(f"HuggingFace auth OK (logged in as {user['name']}).")


def record_until_disconnected(uri, hf_repo_id, robot_id, upload=True, remote_stream_token=None, camera_mode="all", action_space=DEFAULT_ACTION_SPACE):
    # Verify HF auth before doing anything else: dataset resume/create and upload all need it.
    ensure_hf_auth()

    class GracefulExit(Exception):
        pass

    def handle_shutdown_signal(signum, frame):
        print(f"Received termination signal ({signum}). Initiating graceful shutdown...", flush=True)
        raise GracefulExit()

    # Catch SIGTERM (from GCP Preemption) and SIGINT (Ctrl+C for local dev)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)

    # If resuming an existing dataset, match its existing camera_mode/action_space
    # rather than whatever was requested, so new episodes stay consistent with old ones.
    if repo_exists(hf_repo_id, repo_type="dataset"):
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        existing_meta = LeRobotDatasetMetadata(repo_id=hf_repo_id)
        camera_mode = camera_mode_from_features(existing_meta.features)
        action_space = action_space_from_features(existing_meta.features)

    events={
        'episode_abandon': False,
        'end_recording': False,
        'start': False,
        'stop': False,
    }

    # connect to the robot right away because it is our channel to send error messages back to the user.
    robot = StringmanLeRobot(StringmanConfig(uri, remote_stream_token=remote_stream_token, camera_mode=camera_mode, action_space=action_space), events)
    print(describe_session_spaces(camera_mode, action_space, robot.observation_features, robot.action_features))
    robot.connect()
    time.sleep(2)
    if not robot.is_connected:
        raise ConnectionError(f"Failed to connect to robot at {uri}")

    recorded_episodes = 0
    dataset = None

    try:
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        dataset_features = {**action_features, **obs_features}

        dsname = hf_repo_id.split('/')[1]
        root = f"datasets/{dsname}"
        os.makedirs('datasets', exist_ok=True)

        # Automatically determine how to initialize the dataset
        if repo_exists(hf_repo_id, repo_type="dataset"):
            print(f"Found existing dataset {hf_repo_id}. Resuming...")
            # resume requires a seperate root direc
            dataset = LeRobotDataset.resume(
                repo_id=hf_repo_id,
                root=root,
                # image_writer_threads=8,
                streaming_encoding=True,
            )
            # dataset.start_image_writer(num_threads=8)
        else:
            print(f"Creating new dataset {hf_repo_id}...")
            create_kwargs = dict(
                repo_id=hf_repo_id,
                root=root,
                # private=True, # coming soon?
                fps=FPS,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                # image_writer_threads=8,
                streaming_encoding=True,
            )
            try:
                dataset = LeRobotDataset.create(**create_kwargs)
            except FileExistsError:
                # A leftover local dir with no resumable dataset makes create()
                # refuse to overwrite. Since the HF repo doesn't exist, it's just trash;
                # remove it and retry.
                print(f"Local dir {root} exists but isn't resumable; removing and retrying...")
                shutil.rmtree(root)
                dataset = LeRobotDataset.create(**create_kwargs)

        # init_rerun(session_name="stringman_record")
        print("System ready. Press start.")
        robot.send_session_status(common.LerobotSessionStatus(
            status=common.LerobotStatus.REC_READY,
            session_ep_number=0,
            dataset_ep_count=dataset.num_episodes,
            dataset_repo_id=hf_repo_id,
            episodes_until_checkpoint=(CHECKPOINT_EVERY - recorded_episodes % CHECKPOINT_EVERY),
        ))

        while robot.is_connected and not events["end_recording"]:
            time.sleep(0.03)
        
            if events['end_recording']:
                events["end_recording"] = False
                break
            if not events['start']:
                continue 
            events['start'] = False 

            print(f"Recording episode {recorded_episodes + 1}")
            robot.send_session_status(common.LerobotSessionStatus(
                status=common.LerobotStatus.RECORDING,
                session_ep_number=recorded_episodes + 1
            ))

            record_episode(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                max_episode_duration=EPISODE_MAX_TIME_SEC,
                display_data=True,
            )

            if events["episode_abandon"] or not dataset.has_pending_frames():
                if events["episode_abandon"]:
                    print("Discarding episode.")
                else:
                    print("Episode contained no frames; discarding.")
                events["episode_abandon"] = False
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                robot.send_session_status(common.LerobotSessionStatus(status=common.LerobotStatus.REC_EP_ABANDONED))
                robot.send_session_status(common.LerobotSessionStatus(
                    status=common.LerobotStatus.REC_READY,
                    session_ep_number=recorded_episodes
                ))
                continue

            recorded_episodes += 1
            print(f"Episode {recorded_episodes} complete.")
            dataset.save_episode()
            append_episode_metadata(dataset, robot_id)
            print(f"Ready.")
            
            # Checkpoint: Upload data to Hugging Face every 10 episodes
            #if upload and recorded_episodes % CHECKPOINT_EVERY == 0:
            #    print(f"Checkpoint reached: Uploading dataset to Hugging Face ({recorded_episodes} episodes)...", flush=True)
            #    robot.send_session_status(common.LerobotSessionStatus(
            #        status=common.LerobotStatus.REC_CHECKPOINT,
            #        session_ep_number=recorded_episodes,
            #        episodes_until_checkpoint=CHECKPOINT_EVERY,
            #    ))
            #    dataset.push_to_hub()

            # Tf the user set the ep start event while processing was occuring, clear it. Unexpected episode starts lead to low quality data.
            events['start'] = False

            robot.send_session_status(common.LerobotSessionStatus(
                status=common.LerobotStatus.REC_READY,
                session_ep_number=recorded_episodes,
                episodes_until_checkpoint=(CHECKPOINT_EVERY - recorded_episodes % CHECKPOINT_EVERY),
            ))

    except GracefulExit:
        print("Shutdown signal caught. Abandoning any active episode.", flush=True)
        if 'dataset' in locals() and dataset is not None:
            dataset.clear_episode_buffer()

    except Exception as e:
        robot.send_session_status(common.LerobotSessionStatus(
            status=common.LerobotStatus.ERROR,
            error=str(e)
        ))
        raise

    finally:
        print("Recording stopped. Cleaning up.", flush=True)

        if recorded_episodes > 0:
            robot.send_session_status(common.LerobotSessionStatus(
                status=common.LerobotStatus.REC_PROCESSING,
                session_ep_number=recorded_episodes,
            ))
            print(f"{recorded_episodes} episodes collected. Encoding remaining video", flush=True)
            dataset.finalize()
            if upload:
                print("Encoding complete. Uploading to hugging face.", flush=True)
                dataset.push_to_hub()
                print("Upload complete.", flush=True)

        robot.send_session_status(common.LerobotSessionStatus(status=common.LerobotStatus.REC_ALL_COMPLETE))
        time.sleep(0.03)
        robot.disconnect()

def eval_episode(
    robot: Robot,
    policy,
    preprocessor,
    postprocessor,
    dataset_features: dict,
    events: dict,
    fps: int,
    max_episode_duration: int,
    display_data: bool = False,
):
    import torch
    from lerobot.policies.utils import get_device_from_parameters
    try:
        # lerobot 0.5.2
        from lerobot.common.control_utils import predict_action
    except ImportError:
        # lerobot 0.5.1
        from lerobot.utils.control_utils import predict_action

    timestamp = 0
    start_episode_t = time.perf_counter()
    robot.send_session_status(common.LerobotSessionStatus(status=common.LerobotStatus.EVAL_ACTIVE))
    
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    policy_device = get_device_from_parameters(policy)

    while timestamp < max_episode_duration:
        start_loop_t = time.perf_counter()

        if events['end_recording']:
            break
        if events["stop"]:
            events["stop"] = False
            break
        obs = robot.get_observation()
        observation_frame = build_dataset_frame(dataset_features, obs, prefix="observation")

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=policy_device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=robot.last_task_description,
            robot_type=robot.name,
        )

        action_vector = action_values.get('action', action_values) if isinstance(action_values, dict) else action_values
        action_vector = action_vector.cpu().numpy() if torch.is_tensor(action_vector) else action_vector
        action_vector = np.squeeze(action_vector)

        action_dict = dict(zip(dataset_features["action"]["names"], (float(v) for v in action_vector)))

        robot.send_action(action_dict)

        # if display_data:
        #     log_rerun_data(observation=obs, action=action_dict)

        dt_s = time.perf_counter() - start_loop_t
        time.sleep(max(0, 1 / fps - dt_s))
        timestamp = time.perf_counter() - start_episode_t

    robot.send_session_status(common.LerobotSessionStatus(status=common.LerobotStatus.EVAL_IDLE))
    robot.send_action({
        "vel_x": 0.0, "vel_y": 0.0, "vel_z": 0.0,
        "wrist_speed": 0.0,
        "finger_speed": 0.0
    })

def eval_until_disconnected(uri, policy_repo_id, robot_id, device="cuda", remote_stream_token=None, camera_mode=None):
    import torch
    import json
    from huggingface_hub import hf_hub_download
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    events = {
        'episode_abandon': False,
        'end_recording': False,
        'start': False,
        'stop': False,
    }

    # Read train_config.json directly to avoid draccus rejecting unknown fields
    # (e.g. return_uint8 present in the saved config but not in the installed DatasetConfig)
    _config_file = hf_hub_download(repo_id=policy_repo_id, filename="train_config.json")
    with open(_config_file) as _f:
        dataset_repo_id = json.load(_f)["dataset"]["repo_id"]

    print("Fetching training dataset to acquire metadata")
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        download_videos=False,
    )

    if camera_mode is None:
        camera_mode = camera_mode_from_features(dataset.meta.features)
    action_space = action_space_from_features(dataset.meta.features)

    print(f"Loading policy config from {policy_repo_id}...")
    cfg = PreTrainedConfig.from_pretrained(policy_repo_id)
    cfg.pretrained_path = policy_repo_id 

    # smoothly blend overlapping chunks
    cfg.temporal_ensemble_coeff = 0.001

    print("Instantiating processors and policy...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_repo_id,
        dataset_stats=dataset.meta.stats,
    )

    rm=None
    if policy_repo_id == "naavox/g224_smolvla":
        rm={"observation.images.gripper_camera": "observation.images.camera1"}
    elif policy_repo_id.startswith("naavox/jepa"):
        rm={"observation.images.gripper_camera": "observation.images.image", "observation.images.overhead_camera": "observation.images.image2"}

    policy = make_policy(
        cfg=cfg,
        ds_meta=dataset.meta,
        rename_map=rm
    )
    policy.eval()
    print("Policy loaded.")
    
    print(f"Connecting to robot...")
    robot = StringmanLeRobot(StringmanConfig(uri, remote_stream_token=remote_stream_token, camera_mode=camera_mode, action_space=action_space), events)
    print(describe_session_spaces(camera_mode, action_space, robot.observation_features, robot.action_features))
    robot.connect()

    # init_rerun(session_name="stringman_eval")
    print("Eval Ready. Waiting for start command.")
    robot.send_session_status(common.LerobotSessionStatus(
        status=common.LerobotStatus.EVAL_IDLE,
        policy_repo_id=policy_repo_id,
    ))

    while robot.is_connected:
        time.sleep(0.03)
        
        if events['end_recording']:
            events["end_recording"] = False
            break
        if not events['start']:
            continue
        
        events['start'] = False
        
        eval_episode(
            robot=robot,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset.meta.features,
            events=events,
            fps=FPS,
            max_episode_duration=EPISODE_MAX_TIME_SEC,
            display_data=True
        )
        
        print("Episode Complete. Waiting...")
            
    print("Eval process stopping.")
    robot.send_session_status(common.LerobotSessionStatus(status=common.LerobotStatus.EVAL_ALL_COMPLETE))
    robot.disconnect()

if __name__ == "__main__":
    """
    python -m nf_robot.ml.stringman_lerobot record --robot_id=simulated_robot_1 --server_address=ws://localhost:4245 --repo_id=naavox/grasping_dataset
    python -m nf_robot.ml.stringman_lerobot eval --robot_id=simulated_robot_1 --server_address=ws://localhost:4245 --policy_id=naavox/grasping_act_policy --dataset_id=naavox/grasping_dataset
    """
    parser = argparse.ArgumentParser(description="Stringman Lerobot Episode Recorder / Evaluator")
    subparsers = parser.add_subparsers(dest='command', required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--robot_id", default="simulated_robot_1", help="id of robot to record from")
    parent_parser.add_argument("--server_address", default="ws://localhost:4245", help="WebSocket server address (not including path)")
    parent_parser.add_argument("--remote_stream_token", help="Token of authorized user. must be supplied in order to access remote streams. When set, recording will only use using media.neufangled.com and ignore local URIs")

    camera_mode_choices = list(_CAMERA_MODES.keys())
    action_space_choices = list(_ACTION_SPACES.keys())

    record_parser = subparsers.add_parser('record', parents=[parent_parser], help="Record new episodes")
    record_parser.add_argument("--repo_id", default="naavox/grasping_dataset", help="repo id of dataset to append to")
    record_parser.add_argument("--upload", default=True, help="upload data to huggingface when complete")
    record_parser.add_argument("--camera_mode", default="all", choices=camera_mode_choices, help="which cameras to record")
    record_parser.add_argument("--action_space", default=DEFAULT_ACTION_SPACE, choices=action_space_choices, help="action space to record (ignored when resuming an existing dataset)")

    eval_parser = subparsers.add_parser('eval', parents=[parent_parser], help="Evaluate existing policy")
    eval_parser.add_argument("--policy_id", default="naavox/grasping_act_policy", help="repo id of policy to load")
    eval_parser.add_argument("--camera_mode", default=None, choices=camera_mode_choices, help="override the camera setup inferred from the policy's training dataset")

    args = parser.parse_args()
    uri = f'{args.server_address}/control/{args.robot_id}'
    if args.remote_stream_token:
        uri += f'?ticket={args.remote_stream_token}'
    print(f'Connecting to robot at {uri}')

    if args.command == 'eval':
        eval_until_disconnected(uri, args.policy_id, args.robot_id, remote_stream_token=args.remote_stream_token, camera_mode=args.camera_mode)
    else:
        record_until_disconnected(uri, args.repo_id, args.robot_id, args.upload, remote_stream_token=args.remote_stream_token, camera_mode=args.camera_mode, action_space=args.action_space)
