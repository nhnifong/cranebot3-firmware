import time
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_STR, ACTION
from lerobot.utils.visualization_utils import log_rerun_data, init_rerun
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

from .stringman_pilot import StringmanPilotRobot, StringmanConfig

# --- Configuration ---
EPISODE_MAX_TIME_SEC = 600
FPS = 30
TASK_DESCRIPTION = "Pick up clutter from the floor and drop it in the bin."
HF_REPO_ID = "naavox/stringman-practice-dataset-2"
GRPC_ADDR = 'localhost:50051'
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
    while timestamp < max_episode_duration:
        start_loop_t = time.perf_counter()

        if events["episode_start_stop"]:
            events["episode_start_stop"] = False
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
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t

def record_until_disconnected():
    """
    Record episodes to the dataset as long as connected
    episode start and stop are signalled from the gamepad ultimately
    """

    # shared state used for returning early
    events={
        'episode_start_stop': False,
        'rerecord_episode': False,
        'stop_recording': False,
    }

    # Initialize the robot connection
    robot = StringmanPilotRobot(StringmanConfig(GRPC_ADDR))

    # Configure the dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # erase cached dataset from unfinished run
    # base_cache_dir = LeRobotDataset.get_default_cache_dir(HF_REPO_ID)
    base_cache_dir = '/home/nhn/.cache/huggingface/lerobot/naavox/stringman-practice-dataset-2'
    if os.path.exists(base_cache_dir):
        shutil.rmtree(base_cache_dir)
    base_cache_dir = Path(base_cache_dir)

    # Create N distinct dataset objects, each with its own cache directory.
    # they will be rotated between since saving episodes takes so long
    common_args = {
        "repo_id": HF_REPO_ID,
        "fps": FPS,
        "features": dataset_features,
        "robot_type": robot.name,
        "use_videos": True,
        "image_writer_threads": 8,
    }
    
    datasets = []
    for i in range(NUM_BUFFERS):
        cache_dir = base_cache_dir / f"buffer_{i}"
        dataset = LeRobotDataset.create(root=cache_dir, **common_args)
        datasets.append(dataset)
    
    save_futures = [Future() for _ in range(NUM_BUFFERS)]
    for future in save_futures:
        future.set_result(None) # Initialize all futures as "done"
    
    active_idx = 0
    save_executor = ThreadPoolExecutor(max_workers=1)
    recorded_episodes = 0
    try:
        # Connect to the robot
        robot.connect()

        # Initialize Rerun for visualization
        init_rerun(session_name="stringman_record")
        if not robot.is_connected:
            raise ConnectionError("Robot failed to connect!")
        log_say("System ready. Press start.")

        system_ready_logged = False
        while robot.is_connected and not events["stop_recording"]:
            time.sleep(0.1)

            buffer_is_ready = save_futures[active_idx].done()
            if buffer_is_ready and not system_ready_logged:
                log_say("System ready.")
                system_ready_logged = True # say it only once each time this flag is set

            # wait for the signal to start an episode.
            events.update(robot.get_episode_control_events())
            if not events['episode_start_stop']:
                continue # while loop continues to run and process will stop if robot disconnects
            # reset flag
            events['episode_start_stop'] = False

            if not buffer_is_ready:
                log_say("Please wait, the previous episode is still saving.")
                system_ready_logged = False # they only need to hear this if they have been asked to wait for it.
                continue # still waiting

            # Start of a new episode
            active_dataset = datasets[active_idx]
            log_say(f"Recording episode {recorded_episodes + 1}")
            record_episode(
                robot=robot,
                events=events,
                fps=FPS,
                dataset=active_dataset,
                max_episode_duration=EPISODE_MAX_TIME_SEC,
                task_description=TASK_DESCRIPTION,
                display_data=True,
            )

            if events["rerecord_episode"]:
                log_say("Discarding episode.")
                events["rerecord_episode"] = False
                active_dataset.clear_episode_buffer()
                continue

            # Submit save job to background thread
            log_say(f"Episode {recorded_episodes + 1} complete.")
            future = save_executor.submit(active_dataset.save_episode) # takes about 20 seconds
            save_futures[active_idx] = future
            recorded_episodes += 1

            # cycle to the next dataset for the next episode.
            active_idx = (active_idx + 1) % NUM_BUFFERS

    except grpc._channel._InactiveRpcError as e:
        # Silence the specific error from one of our RPC calls detecting that the server has shut down or exited training mode.
        if e.code() != grpc.StatusCode.CANCELLED:
            raise

    finally:
        # Cleanup and Upload
        log_say("Recording stopped. Cleaning up.")
        robot.disconnect()

        if recorded_episodes > 0:
            log_say(f"{recorded_episodes} episodes collected. Waiting for last video encoding to finish.")
            save_executor.shutdown(wait=True)
            log_say("Uploading to hugging face.")
            for i, dataset in enumerate(datasets):
                # Only push datasets that actually contain data
                if len(dataset) > 0:
                    dataset.push_to_hub()
            log_say("Upload complete.")

if __name__ == "__main__":
    record_until_disconnected()