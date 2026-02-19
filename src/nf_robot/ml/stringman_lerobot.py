"""
A Lerobot Robot subclass for Stringman robot with arp gripper.
Connects to the observer process for control and telemetry.
"""

from functools import cached_property
from typing import Any
from dataclasses import dataclass, field
import numpy as np
import cv2
import argparse
import threading
from websockets.sync.client import connect as websocket_connect_sync
import time
from urllib.parse import urlparse
import av # pip install av
import torch

from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common

from lerobot.robots import Robot, RobotConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import OBS_STR, ACTION
from lerobot.utils.visualization_utils import log_rerun_data, init_rerun
from lerobot.utils.utils import log_say
from lerobot.policies.factory import make_policy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.envs.configs import EnvConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType

IMG_RES = 384  # Square resolution of post-stabilized gripper camera.

@dataclass
class SimplePolicyFeature:
    shape: tuple
    dtype: str
    type: FeatureType

@dataclass
class EvalEnvConfig(EnvConfig):
    @property
    def gym_kwargs(self) -> dict:
        return {}

@RobotConfig.register_subclass("stringman")
@dataclass
class StringmanConfig(RobotConfig):
    uri: str

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
        self.websocket = None
        
        # Telemetry state
        self.last_commanded_vel = common.Vec3(0,0,0)
        self.last_observed_vel = common.Vec3(0,0,0)
        
        # Action state (Speed)
        self.last_wrist_speed = 0.0
        self.last_finger_speed = 0.0

        # Sensor state (Position/Status)
        self.last_finger_angle = 0.0
        self.last_pressure = 0.0
        self.last_range = 0.0
        
        # Pose state (Cartesian)
        self.last_gripper_pos = np.zeros(3, dtype=float)
        # 6D Rotation representation (first 2 columns of rotation matrix flattened)
        # Default to identity matrix (no rotation) -> col1=[1,0,0], col2=[0,1,0]
        self.last_gripper_rot_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)

        self.events = events
        # Video state
        self.last_image_frame = np.zeros((IMG_RES, IMG_RES, 3), dtype=np.uint8)
        self.camera_lock = threading.Lock()
        self.video_thread = None
        self.stop_video_event = threading.Event()

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        # Defines the Action Space.
        return { 
            "vel_x": float, # meters per second
            "vel_y": float, 
            "vel_z": float, # up-down
            "wrist_speed": float, # degrees per second
            "finger_speed": float, # degrees per second
        }

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "gripper_camera": (IMG_RES, IMG_RES, 3),
        }

    @cached_property
    def observation_features(self) -> dict:
        # Observation space includes the motor features (actions), cameras,
        # gripper pose, and gripper sensors.
        return {**self._motors_ft, **self._cameras_ft,
            # Cartesian Pose (Replacing raw wrist angle)
            "gripper_pos_x": float,
            "gripper_pos_y": float,
            "gripper_pos_z": float,
            # 6D rotation features, supposedly easier to learn.
            "gripper_rot_0": float,
            "gripper_rot_1": float,
            "gripper_rot_2": float,
            "gripper_rot_3": float,
            "gripper_rot_4": float,
            "gripper_rot_5": float,
            
            "finger_angle": float, # Actual position
            "laser_rangefinder": float, # distance in meters
            "finger_pressure": float,
        }

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

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
        if item.pos_estimate:
            self._handle_pos_estimate(item.pos_estimate)
        if item.grip_sensors:
            self._handle_grip_sensors(item.grip_sensors)
        if item.video_ready:
            self._handle_video_ready(item.video_ready)
        if item.last_commanded_vel:
            self._handle_last_commanded_vel(item.last_commanded_vel)
        if item.last_commanded_grip:
            self._handle_last_commanded_grip(item.last_commanded_grip)
        if item.episode_control:
            self._handle_episode_control(item.episode_control)

    def _handle_pos_estimate(self, item: telemetry.PositionEstimate):
        self.last_observed_vel = item.gantry_velocity
        
        if item.gripper_pose:
            # Translation
            if item.gripper_pose.position:
                self.last_gripper_pos[0] = item.gripper_pose.position.x
                self.last_gripper_pos[1] = item.gripper_pose.position.y
                self.last_gripper_pos[2] = item.gripper_pose.position.z
            
            # Rotation: Convert Rodrigues vector (rvec) to 6D rotation representation
            if item.gripper_pose.rotation:
                rvec = np.array([
                    item.gripper_pose.rotation.x,
                    item.gripper_pose.rotation.y,
                    item.gripper_pose.rotation.z
                ], dtype=float)
                
                # Convert rvec to 3x3 rotation matrix
                # cv2.Rodrigues handles the conversion (theta * axis -> matrix)
                R, _ = cv2.Rodrigues(rvec)
                
                # Take first two columns for 6D representation (Zhou et al, 2019)
                # Flattened: [r11, r21, r31, r12, r22, r32]
                self.last_gripper_rot_6d = R[:, :2].flatten()

    def _handle_grip_sensors(self, item: telemetry.GripperSensors):
        if item.range is not None:
            self.last_range = item.range
        
        if item.angle is not None:
            self.last_finger_angle = item.angle
            
        if item.pressure is not None:
            self.last_pressure = item.pressure

    def _handle_video_ready(self, item: telemetry.VideoReady):
        if not item.is_gripper:
            return

        if self.video_thread is not None and self.video_thread.is_alive():
            return

        url = ""
        if item.local_uri:
            url = item.local_uri
        elif item.stream_path:
            url = f"rtsp://media.neufangled.com:8554/{item.stream_path}"

        if url:
            parsed = urlparse(url)
            hostname = parsed.hostname if parsed.hostname else "localhost"
            is_local = hostname in ["localhost", "127.0.0.1", "::1", "0.0.0.0"]

            print(f"Video ready. Connecting to stream: {url} (Local: {is_local})")
            self.stop_video_event.clear()
            self.video_thread = threading.Thread(
                target=self._video_stream_loop, 
                args=(url, is_local), 
                daemon=True
            )
            self.video_thread.start()
        else:
            print("Received VideoReady event but could not determine stream URL.")

    def _video_stream_loop(self, stream_url, is_local):
        print(f"Opening PyAV stream: {stream_url}")
        
        options = {
            'fflags': 'nobuffer',
            'flags': 'low_delay',
            'fast': '1',
        }

        if stream_url.startswith('rtsp'):
            transport = 'udp' if is_local else 'tcp'
            options['rtsp_transport'] = transport
            options['stimeout'] = '5000000'
        elif stream_url.startswith('http'):
            options['timeout'] = '5000000'

        container = av.open(stream_url, options=options)
        
        try:
            stream = next(s for s in container.streams if s.type == 'video')
            stream.thread_type = "SLICE"
        except StopIteration:
            print(f"No video stream found in {stream_url}")
            return

        try:
            for av_frame in container.decode(stream):
                if self.stop_video_event.is_set():
                    break
                frame = av_frame.to_ndarray(format='rgb24')
                if frame.shape[0] != IMG_RES or frame.shape[1] != IMG_RES:
                    frame = cv2.resize(frame, (IMG_RES, IMG_RES))
                with self.camera_lock:
                    self.last_image_frame = frame
        finally:
            if 'container' in locals():
                container.close()
            print(f"Stream {stream_url} closed")

    def _handle_last_commanded_vel(self, item: telemetry.CommandedVelocity):
        self.last_commanded_vel = tonp(item.velocity)

    def _handle_last_commanded_grip(self, item: telemetry.CommandedGrip):
        self.last_wrist_speed = item.wrist_speed
        self.last_finger_speed = item.finger_speed

    def _handle_episode_control(self, item: common.EpisodeControl):
        if item.command == common.EpCommand.START_OR_COMPLETE:
            self.events['episode_start_or_complete'] = True
        if item.command == common.EpCommand.ABANDON:
            self.events['episode_abandon'] = True
        if item.command == common.EpCommand.END_RECORDING:
            self.events['end_recording'] = True
        if item.command == common.EpCommand.EVAL_START:
            self.events['eval_start'] = True
        if item.command == common.EpCommand.EVAL_STOP:
            self.events['eval_stop'] = True

    def disconnect(self) -> None:
        print("Disconnecting")
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
        with self.camera_lock:
            img_copy = self.last_image_frame.copy()

        obs_dict = {
            'vel_x': float(self.last_observed_vel.x),
            'vel_y': float(self.last_observed_vel.y),
            'vel_z': float(self.last_observed_vel.z),
            
            # Action echo (Speed)
            "wrist_speed": float(self.last_wrist_speed),
            "finger_speed": float(self.last_finger_speed),

            # State Observation (Position/Status)
            "gripper_pos_x": float(self.last_gripper_pos[0]),
            "gripper_pos_y": float(self.last_gripper_pos[1]),
            "gripper_pos_z": float(self.last_gripper_pos[2]),
            
            # 6D Rotation (continuous representation)
            "gripper_rot_0": float(self.last_gripper_rot_6d[0]),
            "gripper_rot_1": float(self.last_gripper_rot_6d[1]),
            "gripper_rot_2": float(self.last_gripper_rot_6d[2]),
            "gripper_rot_3": float(self.last_gripper_rot_6d[3]),
            "gripper_rot_4": float(self.last_gripper_rot_6d[4]),
            "gripper_rot_5": float(self.last_gripper_rot_6d[5]),
            
            "finger_angle": float(self.last_finger_angle),
            "laser_rangefinder": float(self.last_range),
            "finger_pressure": float(self.last_pressure),
            "gripper_camera": img_copy,
        }
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Update internal state for echo
        self.last_wrist_speed = action.get('wrist_speed', 0.0)
        self.last_finger_speed = action.get('finger_speed', 0.0)

        batch = control.ControlBatchUpdate(
            robot_id="0",
            updates=[control.ControlItem(move=control.CombinedMove(
                direction=common.Vec3(
                    x=action['vel_x'],
                    y=action['vel_y'],
                    z=action['vel_z'],
                ),
                finger_speed=self.last_finger_speed,
                wrist_speed=self.last_wrist_speed,
            ))]
        )
        to_send = bytes(batch)
        if self.websocket and to_send:
            self.websocket.send(to_send)
        return action

    def get_last_action(self):
        """
        Get the last action taken by the robot.
        During passive recording, we rely on telemetry updates.
        """
        return {
            "vel_x": self.last_commanded_vel[0],
            "vel_y": self.last_commanded_vel[1],
            "vel_z": self.last_commanded_vel[2],
            "wrist_speed": self.last_wrist_speed,
            "finger_speed": self.last_finger_speed,
        }

# --- Recording Configuration ---
EPISODE_MAX_TIME_SEC = 600
FPS = 30
TASK_DESCRIPTION = "Pick up the item"
NUM_BUFFERS = 3

@safe_stop_image_writer
def record_episode(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset | None = None,
    max_episode_duration: int | None = None,
    task_description: str | None = None,
    display_data: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    print('ep started')
    while timestamp < max_episode_duration:
        start_loop_t = time.perf_counter()

        if events["episode_start_or_complete"]:
            print('ep complete')
            events["episode_start_or_complete"] = False
            break
        if events["episode_abandon"]: 
            print('ep abandon')
            break

        obs = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        
        action_sent = robot.get_last_action()
        action_frame = build_dataset_frame(dataset.features, action_sent, prefix=ACTION)

        frame = {**observation_frame, **action_frame, "task": task_description}
        dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs, action=action_sent)

        dt_s = time.perf_counter() - start_loop_t
        time.sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t

    print('ep finished')

def record_until_disconnected(uri, hf_repo_id):
    events={
        'episode_start_or_complete': False,
        'episode_abandon': False,
        'stop_recording': False,
        'eval_start': False,
        'eval_stop': False,
    }

    robot = StringmanLeRobot(StringmanConfig(uri), events)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    create_dataset = True
    dataset = None
    if create_dataset:
        dataset = LeRobotDataset.create(
            repo_id = hf_repo_id,
            fps = FPS,
            features = dataset_features,
            robot_type = robot.name,
            use_videos = True,
            image_writer_threads = 8,
        )
    else:
        dataset = LeRobotDataset(
            repo_id = hf_repo_id,
            download_videos = False,
            vcodec = 'h264',
        )
        dataset.start_image_writer(num_threads=8)
    
    recorded_episodes = 0 
    try:
        robot.connect()

        if not robot.is_connected:
            raise ConnectionError("Robot failed to connect!")
        init_rerun(session_name="stringman_record")
        log_say("System ready. Press start.")

        while robot.is_connected and not events["stop_recording"]:
            time.sleep(0.03)

            if not events['episode_start_or_complete']:
                continue 
            events['episode_start_or_complete'] = False 
            print(f'Recording episode events={events}')

            log_say(f"Recording episode {recorded_episodes + 1}")
            record_episode(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=dataset,
                max_episode_duration=EPISODE_MAX_TIME_SEC,
                task_description=TASK_DESCRIPTION,
                display_data=True,
            )

            if events["episode_abandon"]:
                log_say("Discarding episode.")
                events["episode_abandon"] = False
                dataset.clear_episode_buffer()
                continue

            log_say(f"Episode {recorded_episodes + 1} complete.")
            dataset.save_episode()
            log_say(f"Ready.")
            recorded_episodes += 1

    finally:
        log_say("Recording stopped. Cleaning up.")
        robot.disconnect()

        if recorded_episodes > 0:
            log_say(f"{recorded_episodes} episodes collected. Encoding remaining video")
            dataset.finalize()
            log_say("Encoding complete. Uploading to hugging face.")
            dataset.push_to_hub()
            log_say("Upload complete.")

def eval_until_disconnected(uri, repo_id, device="cuda"):
    events = {
        'episode_start_or_complete': False,
        'episode_abandon': False,
        'stop_recording': False,
        'eval_start': False,
        'eval_stop': False,
    }

    print(f"Connecting to robot to read features...")
    # 1. Instantiate Robot First to get hardware features
    robot = StringmanLeRobot(StringmanConfig(uri), events)
    
    try:
        robot.connect()
        if not robot.is_connected:
            raise ConnectionError("Robot failed to connect!")

        # 2. Construct EnvConfig from Robot Features (Factory requires EnvConfig or DatasetMetadata)
        print("Constructing environment configuration...")
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation")
        
        # Create an EnvConfig to satisfy make_policy's shape inference requirements
        env_cfg = EvalEnvConfig(fps=FPS)
        
        # Populate Action Features
        for k, v in action_features.items():
            env_cfg.features[k] = SimplePolicyFeature(
                shape=v['shape'], 
                dtype=v['dtype'],
                type=FeatureType.ACTION
            )
            # Explicitly map env keys to policy keys (Identity mapping)
            env_cfg.features_map[k] = k

        # Populate Observation Features
        for k, v in obs_features.items():
            # In LeRobot, images are VISUAL (3 dims), vectors/scalars are STATE
            if len(v['shape']) == 3:
                ft_type = FeatureType.VISUAL
            else:
                ft_type = FeatureType.STATE
                
            env_cfg.features[k] = SimplePolicyFeature(
                shape=v['shape'], 
                dtype=v['dtype'],
                type=ft_type
            )
            # Explicitly map env keys to policy keys (Identity mapping)
            env_cfg.features_map[k] = k

        # 3. Load Policy Configuration & Instantiate
        print(f"Loading policy config from {repo_id}...")
        cfg = PreTrainedConfig.from_pretrained(repo_id)
        cfg.pretrained_path = repo_id 

        print("Instantiating policy...")
        policy = make_policy(
            cfg=cfg,
            env_cfg=env_cfg,
        )
        policy.eval()
        print("Policy loaded.")
        
        init_rerun(session_name="stringman_eval")
        log_say("Eval Ready. Waiting for start command.")

        while robot.is_connected:
            time.sleep(0.03)
            
            if not events['eval_start']:
                continue
            
            events['eval_start'] = False
            log_say("Starting Evaluation Episode")
            
            eval_episode(
                robot=robot,
                policy=policy,
                events=events,
                fps=FPS,
                max_episode_duration=EPISODE_MAX_TIME_SEC,
                display_data=True
            )
            
            log_say("Episode Complete. Waiting...")
            
    finally:
        log_say("Eval process stopping.")
        robot.disconnect()

if __name__ == "__main__":
    """
    python -m nf_robot.ml.stringman_lerobot record
    python -m nf_robot.ml.stringman_lerobot eval
    """
    parser = argparse.ArgumentParser(description="Stringman Lerobot Episode Recorder / Evaluator")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--robot_id", default="simulated_robot_1", help="id of robot to record from")
    parent_parser.add_argument("--server_address", default="ws://localhost:4245", help="WebSocket server address")

    # Record command
    record_parser = subparsers.add_parser('record', parents=[parent_parser], help="Record new episodes")
    record_parser.add_argument("--repo_id", default="naavox/grasping_dataset", help="repo id of dataset to append to")

    # Eval command
    eval_parser = subparsers.add_parser('eval', parents=[parent_parser], help="Evaluate existing policy")
    eval_parser.add_argument("--policy_id", default="naavox/grasping_act_policy", help="repo id of policy to load")
    
    args = parser.parse_args()

    uri = f'{args.server_address}/telemetry/{args.robot_id}'

    if args.command == 'eval':
        eval_until_disconnected(uri, args.policy_id)
    else:
        record_until_disconnected(uri, args.repo_id)