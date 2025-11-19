import time
import grpc

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.utils.control_utils import predict_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import Robot
from lerobot.utils.utils import (
    get_safe_torch_device,
    log_say,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.constants import OBS_STR, ACTION
from lerobot.utils.visualization_utils import log_rerun_data, init_rerun
from .stringman_pilot import StringmanPilotRobot, StringmanConfig
from pprint import pprint

import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
)
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

EPISODE_MAX_TIME_SEC = 600
FPS = 30
TASK_DESCRIPTION = "Pick up laundry from the floor and drop it in the metal basket."
GRPC_ADDR = 'localhost:50051'
DATASET_REPO_ID = "naavox/merged-4"
POLICY_REPO_ID = "naavox/act_19"

def act_one_episode(
    robot: Robot,
    events: dict,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    features,
    fps: int = FPS,
    max_episode_duration: int = EPISODE_MAX_TIME_SEC,
    task_description: str | None = None,
):

    policy.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < start_episode_t + EPISODE_MAX_TIME_SEC:
        start_loop_t = time.perf_counter()

        if events["episode_start_stop"]:
            events["episode_start_stop"] = False
            break

        obs = robot.get_observation()
        observation_frame = build_dataset_frame(features, obs, prefix=OBS_STR)

        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task_description,
            robot_type=robot.robot_type,
        )

        act_processed_policy = make_robot_action(action_values, features)
        pprint(act_processed_policy)
        sent_action = robot.send_action(act_processed_policy)
        # sent_action = act_processed_policy


        # if display_data:
        log_rerun_data(observation=obs, action=sent_action)

        events.update(robot.get_episode_control_events())

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)
        timestamp = time.perf_counter() - start_episode_t

    # stop moving
    # robot.send_action({
    #     'gantry_pos_x': 0,
    #     'gantry_pos_y': 0,
    #     'gantry_pos_z': 0,
    #     'winch_line_length': 0,
    #     'finger_angle': 0,
    # })

def run_until_disconnected():
    robot = StringmanPilotRobot(StringmanConfig(GRPC_ADDR))
    try:
        # currently some data is required from the dataset in order to load the policy
        dataset = LeRobotDataset(DATASET_REPO_ID)

        # applies only to smolvla
        rename_map = {
            "observation.images.gripper_camera": "observation.images.camera1",
            "observation.images.anchor_camera_0": "observation.images.camera2",
            "observation.images.anchor_camera_1": "observation.images.camera3"
        }


        # policy_path_or_id = POLICY_REPO_ID
        policy_id = POLICY_REPO_ID
        path = "/home/nhn/lerobot/outputs/train/act_18/checkpoints/020000/pretrained_model/"
        # policy_cfg = PreTrainedConfig.from_pretrained(path)
        policy_cfg = PreTrainedConfig.from_pretrained(policy_id)
        print(policy_cfg)
        
        # policy_cfg.empty_cameras=1
        policy = make_policy(
            policy_cfg,
            ds_meta=dataset.meta,
            # rename_map=rename_map,
        )

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            # dataset_stats=rename_stats(dataset.meta.stats, rename_map),
            dataset_stats=dataset.meta.stats,
            preprocessor_overrides={
                # "rename_observations_processor": {"rename_map": rename_map},
            },
        )

        # Connect to the robot
        robot.connect()

        # Initialize Rerun for visualization
        init_rerun(session_name="stringman_eval")

        # shared state used for returning early
        events={
            'episode_start_stop': False,
            'rerecord_episode': False,
            'stop_recording': False,
        }

        # like dataset recording. act out one episode each time start is pressed on the controller.
        while robot.is_connected and not events["stop_recording"]:
            time.sleep(0.1)
            # wait for the signal to start an episode.
            events.update(robot.get_episode_control_events())
            if not events['episode_start_stop']:
                continue # while loop continues to run and process will stop if robot disconnects
            # reset flag
            events['episode_start_stop'] = False

            log_say("Start")
            act_one_episode(
                robot=robot,
                events=events,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                features=dataset.features,
                task_description=TASK_DESCRIPTION,
            )
            log_say("End")
    finally:
        # robot.send_action({
        #     'gantry_vel_x': 0,
        #     'gantry_vel_y': 0,
        #     'gantry_vel_z': 0,
        #     'winch_line_speed': 0,
        #     'finger_angle': 0,
        # })
        robot.disconnect()

if __name__ == "__main__":
    run_until_disconnected()