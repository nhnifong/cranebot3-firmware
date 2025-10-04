#!/usr/bin/env python

# currently this depends on lerobot main branch with the hebi import commented out (no ios support)
# this must be installed with pip install -e ~/lerobot

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun
from lerobot.processor import RobotProcessorPipeline, make_default_processors
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.teleoperators.phone.teleop_phone import Phone

from .stringman_pilot import StringmanPilotRobot
from .stringman_pilot_config import StringmanConfig
from .stringman_phone_processor import MapPhoneActionToStringmanAction
from .stringman_pilot import StringmanPilotRobot
from .stringman_pilot_config import StringmanConfig

import time
import os
import shutil

d = '/home/nhn/.cache/huggingface/lerobot/naavox/stringman-practice-dataset-2'
if os.path.exists(d):
    shutil.rmtree(d)

# --- Configuration ---
EPISODE_MAX_TIME_SEC = 600
FPS = 30
TASK_DESCRIPTION = "Pick up clutter from the floor and drop it in the bin."
HF_REPO_ID = "naavox/stringman-practice-dataset-2"
GRPC_ADDR = 'localhost:50051'


# Initialize keyboard listener and events to get a reference to the events dictionary
listener, events = init_keyboard_listener()

# Initialize the robot and teleoperatore
robot = StringmanPilotRobot(StringmanConfig(GRPC_ADDR))

phone_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
phone_teleop = Phone(phone_config)

phone_to_stringman_processor = RobotProcessorPipeline(
    steps=[MapPhoneActionToStringmanAction(platform=phone_config.phone_os)],
    # Use the default helper functions for standard data formatting.
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

# Create the dataset on the Hugging Face Hub
dataset = LeRobotDataset.create(
    repo_id=HF_REPO_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

recorded_episodes = 0

try:
    # Connect to the robot and teleoperator
    robot.connect()
    phone_teleop.connect()

    # Initialize Rerun for visualization
    init_rerun(session_name="stringman_record")

    if not robot.is_connected:
        raise ConnectionError("Robot or teleoperator failed to connect!")

    log_say("System ready.")

    while robot.is_connected and not events["stop_recording"]:

        # wait for the signal to start an episode
        # wand_teleoperator sets this whenever the button is pressed without knowing whether an episode is being recorded
        # so it serves two purposes here
        # if not events['exit_early']:
        #     time.sleep(0.1)
        #     continue
        # # reset flag
        # events['exit_early'] = False

        log_say(f"Recording episode {recorded_episodes + 1}")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            dataset=dataset,
            teleop=phone_teleop,
            control_time_s=EPISODE_MAX_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            teleop_action_processor=phone_to_stringman_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
        # stop moving at the end of the episode
        robot.send_action({
            "gantry_vel_x": 0,
            "gantry_vel_y": 0,
            "gantry_vel_z": 0,
            "winch_line_speed": 0,
            "finger_angle": -90, # user should be mindful not to end episodes while holding something. it will be dropped.
        })

        # Check if the user wants to stop or re-record before resetting
        if events["stop_recording"]:
            break

        if events["rerecord_episode"]:
            log_say("Re-recording episode.")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # Save the successfully recorded episode
        log_say(f"Episode {recorded_episodes + 1} complete. Encoding images to video file.")
        dataset.save_episode()
        recorded_episodes += 1


finally:
    # --- Cleanup and Upload ---
    log_say("Recording stopped. Cleaning up...")
    robot.disconnect()
    phone_teleop.disconnect()
    listener.stop()

    if recorded_episodes > 0:
        log_say(f"{recorded_episodes} episodes collected. Uploading to Hugging Face.")
        dataset.push_to_hub()
        log_say("Upload complete.")