import time
import grpc

from lerobot.robots import Robot
from lerobot.utils.utils import (
    get_safe_torch_device,
    log_say,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import log_rerun_data, init_rerun
from .stringman_pilot import StringmanPilotRobot, StringmanConfig
from pprint import pprint
from .ghost import MultiCameraResNet, MODEL_PATH
import torch
import numpy as np

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
DATASET_REPO_ID = "naavox/merged-5"

def prepare_image(img_array):
            # Convert to tensor
            tensor = torch.from_numpy(img_array)
            
            # If image is (H, W, C), permute to (C, H, W)
            if tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1)
                
            # Normalize to [0, 1] if it's uint8 (0-255)
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            elif tensor.dtype == torch.float32 and tensor.max() > 1.0:
                 # Just in case it's float but 0-255
                 tensor = tensor / 255.0
                 
            # Add batch dimension (1, C, H, W) and move to GPU
            return tensor.unsqueeze(0).cuda()

def act_one_episode(
    robot: Robot,
    events: dict,
    model,
):

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < start_episode_t + EPISODE_MAX_TIME_SEC:
        start_loop_t = time.perf_counter()

        if events["episode_start_stop"]:
            events["episode_start_stop"] = False
            break

        obs = robot.get_observation()

        # Add a batch dimension (B, C, H, W) to all 3 images
        img0_batch = prepare_image(obs['anchor_camera_0'])
        img1_batch = prepare_image(obs['anchor_camera_1'])
        img2_batch = prepare_image(obs['gripper_camera'])
        state = torch.tensor([
            obs['gantry_pos_x'],
            obs['gantry_pos_y'],
            obs['gantry_pos_z'],
            obs['winch_line_length'],
            obs['finger_angle'],
            obs['gripper_imu_rot_x'],
            obs['gripper_imu_rot_y'],
            obs['gripper_imu_rot_z'],
            obs['laser_rangefinder'],
            obs['finger_pad_voltage'],
        ]).unsqueeze(0).cuda()

        # Get the model's prediction
        prediction_tensor = model(img0_batch, img1_batch, img2_batch, state)
        
        # Move prediction to CPU and remove batch dimension
        predicted_coords = prediction_tensor.squeeze().cpu().numpy()

        action = { 
            "gantry_pos_x": predicted_coords[0],
            "gantry_pos_y": predicted_coords[1],
            "gantry_pos_z": predicted_coords[2],
            "winch_line_length": 0.6,#predicted_coords[3],
            "finger_angle": predicted_coords[4],
        }

        pprint(action)
        sent_action = robot.send_action(action)

        # if display_data:
        log_rerun_data(observation=obs, action=sent_action)

        events.update(robot.get_episode_control_events())

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / FPS - dt_s)
        timestamp = time.perf_counter() - start_episode_t

def run_until_disconnected():
    robot = StringmanPilotRobot(StringmanConfig(GRPC_ADDR))
    try:
        model = MultiCameraResNet().cuda()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()  # Set model to evaluation mode

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

        with torch.no_grad():
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
                    model=model,
                )
                log_say("End")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    run_until_disconnected()