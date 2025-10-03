#!/usr/bin/env python

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from stringman_pilot_robot import StringmanPilotRobot
from stringman_pilot_config import StringmanConfig
from wand_teleoperator import StringmanTrainingWand
from wand_teleoperator_config import WandConfig

# --- Configuration ---
NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 45
RESET_TIME_SEC = 10
TASK_DESCRIPTION = "Describe the task you are recording here."
# TODO: IMPORTANT! Replace with your Hugging Face username and desired dataset repo ID.
HF_REPO_ID = "naavox/stringman-practice-dataset-2"


# Initialize keyboard listener and events to get a reference to the events dictionary
listener, events = init_keyboard_listener()

# Initialize the robot and teleoperatore
robot = StringmanPilotRobot(StringmanConfig())
wand = StringmanTrainingWand(WandConfig(), events=events)

# Initialize default data processors
teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

# Configure the dataset features
action_features = hw_to_dataset_features(robot.action_features, ACTION)
obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
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

# Connect to the robot and teleoperator
robot.connect()
wand.connect()

# Initialize the keyboard listener for recording controls (e.g., 'q' to quit)
listener, events = init_keyboard_listener()

# Initialize Rerun for visualization
init_rerun(session_name="stringman_record")

if not robot.is_connected or not wand.is_connected:
    raise ConnectionError("Robot or teleoperator failed to connect!")

print("Starting record loop... Press 'q' then Enter to stop.")
recorded_episodes = 0
while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
    log_say(f"Recording episode {recorded_episodes + 1}/{NUM_EPISODES}")

    # Main record loop for the task
    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        dataset=dataset,
        teleop=[wand],
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
        robot_observation_processor=robot_observation_processor,
    )

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
    dataset.save_episode()
    recorded_episodes += 1

    # Reset the environment between episodes (if not the last one)
    if recorded_episodes < NUM_EPISODES:
        log_say("Reset the environment for the next episode.")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=[wand],
            control_time_s=RESET_TIME_SEC,
            single_task="Resetting the environment",
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )


# --- Cleanup and Upload ---
log_say("Recording finished. Cleaning up...")
robot.disconnect()
wand.disconnect()
listener.stop()

log_say(f"Uploading dataset to {HF_REPO_ID} on the Hugging Face Hub.")
dataset.push_to_hub()
log_say("Upload complete.")