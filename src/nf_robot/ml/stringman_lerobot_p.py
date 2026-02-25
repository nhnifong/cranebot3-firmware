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
from websockets.sync.client import connect as websocket_connect_sync
import time
from urllib.parse import urlparse
import av
import torch
from huggingface_hub import repo_exists

from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common

from lerobot.robots import Robot, RobotConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import OBS_STR, ACTION
from lerobot.utils.visualization_utils import log_rerun_data, init_rerun
from lerobot.utils.utils import log_say
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import get_device_from_parameters
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.control_utils import predict_action

IMG_RES = 384

@RobotConfig.register_subclass("stringman")
@dataclass
class StringmanConfig(RobotConfig):
    uri: str
    vision_only: bool = False

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
        self.vision_only = config.vision_only
        self.websocket = None
        
        self.last_commanded_vel = common.Vec3(0,0,0)
        self.last_observed_vel = common.Vec3(0,0,0)
        
        self.last_wrist_speed = 0.0
        self.last_finger_speed = 0.0

        self.last_finger_angle = 0.0
        self.last_wrist_angle = 0.0
        self.last_pressure = 0.0
        self.last_range = 0.0
        
        self.last_gripper_pos = np.zeros(3, dtype=float)
        self.episode_start_pos = np.zeros(3, dtype=float)
        self.last_gripper_rot_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float)
        
        # Track the last generated target action for the observation echo
        self.last_target_rel_pos = np.zeros(3, dtype=float)
        self.last_target_wrist = 0.0
        self.last_target_finger = 0.0

        self.events = events
        self.last_image_frame = np.zeros((IMG_RES, IMG_RES, 3), dtype=np.uint8)
        self.camera_lock = threading.Lock()
        self.video_thread = None
        self.stop_video_event = threading.Event()

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return { 
            "target_rel_pos_x": float,
            "target_rel_pos_y": float, 
            "target_rel_pos_z": float,
            "target_wrist_angle": float,
            "target_finger_angle": float,
        }

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            "gripper_camera": (IMG_RES, IMG_RES, 3),
        }

    @cached_property
    def observation_features(self) -> dict:
        obs = {**self._motors_ft, **self._cameras_ft,
            "gripper_rel_pos_x": float,
            "gripper_rel_pos_y": float,
            "gripper_rel_pos_z": float,
            "gripper_rot_0": float,
            "gripper_rot_1": float,
            "gripper_rot_2": float,
            "gripper_rot_3": float,
            "gripper_rot_4": float,
            "gripper_rot_5": float,
            "wrist_angle_sin": float,
            "wrist_angle_cos": float,
        }
        
        # Conditionally omit "shortcut" state features if running in vision-only mode
        if not self.vision_only:
            obs["finger_angle"] = float
            obs["laser_rangefinder"] = float
            obs["finger_pressure"] = float
            
        return obs

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
            if item.gripper_pose.position:
                self.last_gripper_pos[0] = item.gripper_pose.position.x
                self.last_gripper_pos[1] = item.gripper_pose.position.y
                self.last_gripper_pos[2] = item.gripper_pose.position.z
            
            if item.gripper_pose.rotation:
                rvec = np.array([
                    item.gripper_pose.rotation.x,
                    item.gripper_pose.rotation.y,
                    item.gripper_pose.rotation.z
                ], dtype=float)
                
                R, _ = cv2.Rodrigues(rvec)
                self.last_gripper_rot_6d = R[:, :2].flatten()

    def _handle_grip_sensors(self, item: telemetry.GripperSensors):
        if item.range is not None:
            self.last_range = item.range
        if hasattr(item, 'wrist') and item.wrist is not None:
            self.last_wrist_angle = item.wrist
        if item.angle is not None:
            self.last_finger_angle = item.angle
        if item.pressure is not None:
            self.last_pressure = item.pressure

    def _handle_video_ready(self, item: telemetry.VideoReady):
        if not item.is_gripper:
            return

        if self.video_thread is not None and self.video_thread.is_alive():
            return

        url = item.local_uri if item.local_uri else (f"rtsp://media.neufangled.com:8554/{item.stream_path}" if item.stream_path else "")

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
            options['rtsp_transport'] = 'udp' if is_local else 'tcp'
            options['stimeout'] = '5000000'
        elif stream_url.startswith('http'):
            options['timeout'] = '5000000'

        container = av.open(stream_url, options=options)
        stream = next(s for s in container.streams if s.type == 'video')
        stream.thread_type = "SLICE"

        for av_frame in container.decode(stream):
            if self.stop_video_event.is_set():
                break
            frame = av_frame.to_ndarray(format='rgb24')
            if frame.shape[0] != IMG_RES or frame.shape[1] != IMG_RES:
                frame = cv2.resize(frame, (IMG_RES, IMG_RES))
            with self.camera_lock:
                self.last_image_frame = frame
                
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
            # Establish the origin for relative movement tracking
            self.episode_start_pos = self.last_gripper_pos.copy()
        if item.command == common.EpCommand.ABANDON:
            self.events['episode_abandon'] = True
        if item.command == common.EpCommand.END_RECORDING:
            self.events['end_recording'] = True
        if item.command == common.EpCommand.EVAL_START:
            self.events['eval_start'] = True
            self.episode_start_pos = self.last_gripper_pos.copy()
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
            
        rel_pos = self.last_gripper_pos - self.episode_start_pos
        wrist_rads = np.radians(self.last_wrist_angle)

        obs = {
            "target_rel_pos_x": float(self.last_target_rel_pos[0]),
            "target_rel_pos_y": float(self.last_target_rel_pos[1]),
            "target_rel_pos_z": float(self.last_target_rel_pos[2]),
            "target_wrist_angle": float(self.last_target_wrist),
            "target_finger_angle": float(self.last_target_finger),

            "gripper_rel_pos_x": float(rel_pos[0]),
            "gripper_rel_pos_y": float(rel_pos[1]),
            "gripper_rel_pos_z": float(rel_pos[2]),
            
            "gripper_rot_0": float(self.last_gripper_rot_6d[0]),
            "gripper_rot_1": float(self.last_gripper_rot_6d[1]),
            "gripper_rot_2": float(self.last_gripper_rot_6d[2]),
            "gripper_rot_3": float(self.last_gripper_rot_6d[3]),
            "gripper_rot_4": float(self.last_gripper_rot_6d[4]),
            "gripper_rot_5": float(self.last_gripper_rot_6d[5]),
            
            "wrist_angle_sin": float(np.sin(wrist_rads)),
            "wrist_angle_cos": float(np.cos(wrist_rads)),
            "gripper_camera": img_copy,
        }
        
        if not self.vision_only:
            obs["finger_angle"] = float(self.last_finger_angle)
            obs["laser_rangefinder"] = float(self.last_range)
            obs["finger_pressure"] = float(self.last_pressure)
            
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Track the generated targets for the observation echo
        self.last_target_rel_pos = [action['target_rel_pos_x'], action['target_rel_pos_y'], action['target_rel_pos_z']]
        self.last_target_wrist = action['target_wrist_angle']
        self.last_target_finger = action['target_finger_angle']
        
        curr_rel_pos = self.last_gripper_pos - self.episode_start_pos
        
        # P-Controller converting target absolute positions into movement velocities
        Kp_pos = 5.0
        Kp_ang = 2.0
        
        vel_x = Kp_pos * (action['target_rel_pos_x'] - curr_rel_pos[0])
        vel_y = Kp_pos * (action['target_rel_pos_y'] - curr_rel_pos[1])
        vel_z = Kp_pos * (action['target_rel_pos_z'] - curr_rel_pos[2])
        
        wrist_speed = Kp_ang * (action['target_wrist_angle'] - self.last_wrist_angle)
        finger_speed = Kp_ang * (action['target_finger_angle'] - self.last_finger_angle)

        batch = control.ControlBatchUpdate(
            robot_id="0",
            updates=[control.ControlItem(move=control.CombinedMove(
                direction=common.Vec3(
                    x=vel_x,
                    y=vel_y,
                    z=vel_z,
                ),
                finger_speed=finger_speed,
                wrist_speed=wrist_speed,
            ))]
        )
        to_send = bytes(batch)
        if self.websocket and to_send:
            self.websocket.send(to_send)
        return action

    def get_last_action(self):
        # Create the training labels by extrapolating the current teleoperated velocity 
        # into a spatial waypoint for the P-controller to chase
        lookahead_dt = 0.1 
        rel_pos = self.last_gripper_pos - self.episode_start_pos
        
        return {
            "target_rel_pos_x": float(rel_pos[0] + self.last_commanded_vel[0] * lookahead_dt),
            "target_rel_pos_y": float(rel_pos[1] + self.last_commanded_vel[1] * lookahead_dt),
            "target_rel_pos_z": float(rel_pos[2] + self.last_commanded_vel[2] * lookahead_dt),
            "target_wrist_angle": float(self.last_wrist_angle + self.last_wrist_speed * lookahead_dt),
            "target_finger_angle": float(self.last_finger_angle + self.last_finger_speed * lookahead_dt),
        }

# --- Recording Configuration ---
EPISODE_MAX_TIME_SEC = 600
FPS = 30
TASK_DESCRIPTION = "Pick up the item"

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

def record_until_disconnected(uri, hf_repo_id, vision_only=False):
    events={
        'episode_start_or_complete': False,
        'episode_abandon': False,
        'stop_recording': False,
        'eval_start': False,
        'eval_stop': False,
    }

    robot = StringmanLeRobot(StringmanConfig(uri, vision_only=vision_only), events)

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Automatically determine how to initialize the dataset
    if repo_exists(hf_repo_id, repo_type="dataset"):
        print(f"Found existing dataset {hf_repo_id}. Resuming...")
        dataset = LeRobotDataset(
            repo_id=hf_repo_id,
            download_videos=False,
        )
        dataset.start_image_writer(num_threads=8)
    else:
        print(f"Creating new dataset {hf_repo_id}...")
        dataset = LeRobotDataset.create(
            repo_id=hf_repo_id,
            fps=FPS,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=8,
        )
    
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
    timestamp = 0
    start_episode_t = time.perf_counter()
    print('Eval episode started')
    
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    policy_device = get_device_from_parameters(policy)

    while timestamp < max_episode_duration:
        start_loop_t = time.perf_counter()

        if events["episode_start_or_complete"]:
            print('Eval stopped via command')
            events["episode_start_or_complete"] = False
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
            task=TASK_DESCRIPTION,
            robot_type=robot.name,
        )

        action_vector = action_values.get('action', action_values) if isinstance(action_values, dict) else action_values
        action_vector = action_vector.cpu().numpy() if torch.is_tensor(action_vector) else action_vector
        action_vector = np.squeeze(action_vector)
        
        action_dict = {
            "target_rel_pos_x": float(action_vector[0]),
            "target_rel_pos_y": float(action_vector[1]),
            "target_rel_pos_z": float(action_vector[2]),
            "target_wrist_angle": float(action_vector[3]),
            "target_finger_angle": float(action_vector[4]),
        }

        robot.send_action(action_dict)

        if display_data:
            # We must build a faux state to visualize the commanded action properly in rerun
            action_echo = {
                "vel_x": float(robot.last_commanded_vel.x),
                "vel_y": float(robot.last_commanded_vel.y),
                "vel_z": float(robot.last_commanded_vel.z),
                "wrist_speed": float(robot.last_wrist_speed),
                "finger_speed": float(robot.last_finger_speed),
            }
            log_rerun_data(observation=obs, action=action_echo)

        dt_s = time.perf_counter() - start_loop_t
        time.sleep(max(0, 1 / fps - dt_s))
        timestamp = time.perf_counter() - start_episode_t

    print('Eval episode finished')
    # Use the P-Controller structure to brake by setting targets to the current actual positions
    robot.send_action({
        "target_rel_pos_x": float(robot.last_gripper_pos[0] - robot.episode_start_pos[0]),
        "target_rel_pos_y": float(robot.last_gripper_pos[1] - robot.episode_start_pos[1]),
        "target_rel_pos_z": float(robot.last_gripper_pos[2] - robot.episode_start_pos[2]),
        "target_wrist_angle": float(robot.last_wrist_angle),
        "target_finger_angle": float(robot.last_finger_angle),
    })

def eval_until_disconnected(uri, policy_repo_id, dataset_repo_id, vision_only=False, device="cuda"):
    events = {
        'episode_start_or_complete': False,
        'episode_abandon': False,
        'stop_recording': False,
        'eval_start': False,
        'eval_stop': False,
    }

    print("Fetching training dataset to acquire metadata")
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        download_videos=False,
    )
    
    print(f"Loading policy config from {policy_repo_id}...")
    cfg = PreTrainedConfig.from_pretrained(policy_repo_id)
    cfg.pretrained_path = policy_repo_id 
    cfg.temporal_ensemble_coeff = 0.01

    print("Instantiating processors and policy...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_repo_id,
        dataset_stats=dataset.meta.stats,
    )

    policy = make_policy(
        cfg=cfg,
        ds_meta=dataset.meta,
    )
    policy.eval()
    print("Policy loaded.")
    
    print(f"Connecting to robot...")
    robot = StringmanLeRobot(StringmanConfig(uri, vision_only=vision_only), events)
    robot.connect()
    
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
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset.meta.features,
            events=events,
            fps=FPS,
            max_episode_duration=EPISODE_MAX_TIME_SEC,
            display_data=True
        )
        
        log_say("Episode Complete. Waiting...")
            
    log_say("Eval process stopping.")
    robot.disconnect()

if __name__ == "__main__":
    """
    python -m nf_robot.ml.stringman_lerobot record --robot_id=simulated_robot_1 --server_address=ws://localhost:4245 --repo_id=naavox/grasping_dataset --vision_only
    python -m nf_robot.ml.stringman_lerobot eval --robot_id=simulated_robot_1 --server_address=ws://localhost:4245 --policy_id=naavox/grasping_act_policy --dataset_id=naavox/grasping_dataset --vision_only
    """
    parser = argparse.ArgumentParser(description="Stringman Lerobot Episode Recorder / Evaluator")
    subparsers = parser.add_subparsers(dest='command', required=True)

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--robot_id", default="simulated_robot_1", help="id of robot to record from")
    parent_parser.add_argument("--server_address", default="ws://localhost:4245", help="WebSocket server address")
    parent_parser.add_argument("--vision_only", action="store_true", help="Omit shortcut sensors like the rangefinder to force purely visual behavioral cloning")

    record_parser = subparsers.add_parser('record', parents=[parent_parser], help="Record new episodes")
    record_parser.add_argument("--repo_id", default="naavox/grasping_dataset", help="repo id of dataset to append to")

    eval_parser = subparsers.add_parser('eval', parents=[parent_parser], help="Evaluate existing policy")
    eval_parser.add_argument("--policy_id", default="naavox/grasping_act_policy", help="repo id of policy to load")
    eval_parser.add_argument("--dataset_id", default="naavox/grasping_dataset", help="repo id of dataset for un-normalization stats")
    
    args = parser.parse_args()
    uri = f'{args.server_address}/telemetry/{args.robot_id}'

    if args.command == 'eval':
        eval_until_disconnected(uri, args.policy_id, args.dataset_id, vision_only=args.vision_only)
    else:
        record_until_disconnected(uri, args.repo_id, vision_only=args.vision_only)