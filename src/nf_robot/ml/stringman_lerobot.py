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

from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common

from lerobot.robots import Robot, RobotConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.constants import OBS_STR, ACTION
from lerobot.utils.visualization_utils import log_rerun_data, init_rerun
from lerobot.utils.utils import log_say

IMG_RES = 384  # Square resolution of post-stabilized gripper camera.

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

    def __init__(self, config: StringmanConfig):
        super().__init__(config)
        self.address = config.uri
        self.websocket = None
        self.last_commanded_vel = common.Vec3(0,0,0)
        self.last_observed_vel = common.Vec3(0,0,0)
        self.last_wrist_angle = 0
        self.last_finger_angle = 0
        self.last_pressure = 0
        self.last_range = 0
        self.events = {}

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        # lerobot assumes all features are either joints (float) or images (speicified as a tuple of width, height, channels)
        # here I have place all the properties we can command of the robot, even if they are not strictly motor joints.
        # Much of the kinematics of the robot are abstracted away so the model only learns how to perform a grasp.
        # It commands the velocity of the gripper and the wrist and finger angles.
        # the motion controller is responsible for moving the gripper to that position with swing cancellation and safety limits.
        return { 
            "vel_x": float, # meters per second
            "vel_y": float, 
            "vel_z": float, # up-down
            "wrist_angle": float, # radians away from a position aligned with the room y axis and camera y axis. can be positive or negative by multiple revolutions. 
            "finger_angle": float, # fully open (0) to fully closed (1)
        }

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        # use only one anchor camera to keep latency high and training load lower.
        return {
            "gripper_camera": (IMG_RES, IMG_RES, 3),
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft,
            "accel_x": float,
            "accel_y": float,
            "accel_z": float,
            "gyro_x": float,
            "gyro_y": float,
            "gyro_z": float,
            "laser_rangefinder": float, # distance in meters
            "finger_pressure": float, # no pressure (0) max pressure (1) 
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
        # block until completely connected.
        give_up = time.time()+5
        while self.websocket is None and time.time() < give_up:
            time.sleep(0.1)

    def connect_thread(self):
        # consume the telemetry stream from the robot like any other UI
        # saving the values relevant to this teleoperation system
        print(f'Connecting to {self.address}')
        with websocket_connect_sync(self.address) as websocket:
            self.websocket = websocket
            print(f'Connected')
            # iterator ends when websocket closes.
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
        if item.episode_control:
            self._handle_episode_control(item.episode_control)

    def _handle_pos_estimate(self, item: telemetry.PositionEstimate):
        self.last_observed_vel = item.gantry_velocity

    def _handle_grip_sensors(self, item: telemetry.GripperSensors):
        # item.range
        # item.angle
        # item.wrist
        # item.pressure
        if item.range is not None:
            self.last_range = item.range
        if item.angle is not None:
            self.last_finger_angle = item.angle
        if item.wrist is not None:
            self.last_wrist_angle = item.wrist
        if item.pressure is not None:
            self.last_pressure = item.pressure
        # TODO others

    def _handle_video_ready(self, item: telemetry.VideoReady):
        # for local robots connect at this address
        # udp:127.0.0.1:1234
        # item.local_uri

        # for remote robots construct a url using this stream path
        # for example http://localhost:8889/${streamPath}/whep
        # item.stream_path
        pass

    def _handle_last_commanded_vel(self, item: telemetry.CommandedVelocity):
        self.last_commanded_vel = item.velocity

    def _handle_episode_control(self, item: common.EpisodeControl):
        # these are forwarded from any UI connected to the robot.
        # if you receive one, store it in the events dict. it is up to the 
        # record_until_disconnected and record_episode to take action on them and clear them
        if item.command == common.EpCommand.EPCOMMAND_START_OR_COMPLETE:
            self.events['episode_start_or_complete'] = True
        if item.command == common.EpCommand.ABANDON:
            self.events['episode_abandon'] = True
        if item.command == common.EpCommand.END_RECORDING:
            self.events['end_recording'] = True

    def get_episode_control_events(self):
        return self.events

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
        obs_dict = {
            'vel_x': float(self.last_observed_vel.x),
            'vel_y': float(self.last_observed_vel.y),
            'vel_z': float(self.last_observed_vel.z),
            "wrist_angle": float(self.last_wrist_angle),
            "finger_angle": float(self.last_finger_angle),
            "accel_x": 0.0,
            "accel_y": 0.0,
            "accel_z": 0.0,
            "gyro_x": 0.0,
            "gyro_y": 0.0,
            "gyro_z": 0.0,
            "laser_rangefinder": float(self.last_range),
            "finger_pressure": float(self.last_pressure),
            "gripper_camera": float(self.last_image_frame),
        }
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # send action to robot
        batch = control.ControlBatchUpdate(
            robot_id="0",
            updates=[control.ControlItem(move=control.CombinedMove(
                direction=common.Vec3(
                    x=action['vel_x'],
                    y=action['vel_y'],
                    z=action['vel_z'],
                ),
                finger=action['finger_angle'],
                wrist=action['wrist_angle'],
            ))]
        )
        to_send = bytes(batch)
        # synchronous, and we're on a different thread, but websockets does this in thread safe way
        if self.websocket and to_send:
            self.websocket.send(to_send)
        # return the action that was actually taken
        return action

    def get_last_action(self):
        """
        Get the last action taken by the robot
        Not part of normal lerobot flow. I'm bypassing the teleoperator for the sake of latency.
        The robot's telemetry stream has already told os what the operator last did so we record that.
        """
        return {
            "vel_x": self.last_commanded_vel[0],
            "vel_y": self.last_commanded_vel[1],
            "vel_z": self.last_commanded_vel[2],
            "wrist_angle": self.last_wrist_angle, # todo differentiate between last commanded and last observed
            "finger_angle": self.last_finger_angle,
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
    """
    Record a single grasp. Starting from a position above an object.
    An episode should center on and grasp the object, retrying if necessary,
    And then raise the object off the floor before ending the episode.
    """
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < max_episode_duration:
        start_loop_t = time.perf_counter()

        if events["episode_start_or_complete"]:
            events["episode_start_or_complete"] = False
            break
        if events["episode_abandon"]: # gets cleared by record_until_disconnected
            break

        # Get robot observation
        obs = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, obs, prefix=OBS_STR)
        
        # get last action taken
        action_sent = robot.get_last_action()
        action_frame = build_dataset_frame(dataset.features, action_sent, prefix=ACTION)

        # Write to dataset
        frame = {**observation_frame, **action_frame, "task": task_description}
        dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs, action=action_sent)

        events.update(robot.get_episode_control_events())

        dt_s = time.perf_counter() - start_loop_t
        time.sleep(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t



def record_until_disconnected(uri, hf_repo_id):
    """
    Record episodes to the dataset as long as connected
    episode start and stop are signalled from the gamepad ultimately
    """

    # shared state used for returning early
    events={
        'episode_start_or_complete': False,
        'episode_abandon': False,
        'stop_recording': False,
    }

    # Initialize the robot connection
    robot = StringmanLeRobot(StringmanConfig(uri))

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # TODO determine if hf_repo_id exists already ussing hfapi
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
            # async_video_encoding = True,
            vcodec = 'h264',
        )
    else:
        dataset = LeRobotDataset(
            repo_id = hf_repo_id,
            download_videos = False,
            # async_video_encoding = True,
            vcodec = 'h264',
        )
        dataset.start_image_writer(num_threads=8)

        # TODO depend on my fork which has this function
        # dataset.start_async_video_encoder()
    
    recorded_episodes = 0 # number of new episodes recorded during this session
    try:
        # Connect to the robot
        robot.connect()

        # Initialize Rerun for visualization
        if not robot.is_connected:
            raise ConnectionError("Robot failed to connect!")
        init_rerun(session_name="stringman_record")
        log_say("System ready. Press start.")

        while robot.is_connected and not events["stop_recording"]:
            time.sleep(0.03)

            # wait for the signal to start an episode.
            events.update(robot.get_episode_control_events())
            if not events['episode_start_or_complete']:
                continue # while loop continues to run and process will stop if robot disconnects
            events['episode_start_or_complete'] = False # reset flag

            # Start of a new episode
            # TODO speak the episode number from the whole dataset, not just this session.
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
        # Cleanup and Upload
        log_say("Recording stopped. Cleaning up.")
        robot.disconnect()

        if recorded_episodes > 0:
            # log_say(f"{recorded_episodes} episodes collected. Encoding remaining video")
            # dataset.stop_async_video_encoder(wait=True)
            log_say("Encoding complete. Uploading to hugging face.")
            # dataset.push_to_hub()
            log_say("Upload complete.")

if __name__ == "__main__":
    """
    python -m nf_robot.ml.stringman_lerobot --robot_id=simulated_robot_1 --server_address=ws://localhost:8080 --repo_id=naavox/grasping_dataset
    """
    parser = argparse.ArgumentParser(description="Stringman Lerobot Episode Recorder")
    parser.add_argument("--server_address", help="WebSocket server address (ws://localhost:8080)")
    parser.add_argument("--robot_id", help="id of robot to record from")
    parser.add_argument("--repo_id", help="repo id of dataset to append to (naavox/grasping_dataset)")
    args = parser.parse_args()

    uri = f'{args.server_address}/telemetry/{args.robot_id}'

    # TODO add args for any useful settings.
    record_until_disconnected(uri, args.repo_id)