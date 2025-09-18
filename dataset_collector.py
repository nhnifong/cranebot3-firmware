import numpy as np
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

image_size = (480, 270)

class LeRobotDatasetCollector:
    def __init__(self, root_path="last_dataset", datastore, posi):
        self.datastore = datastore
        self.pe = posi
        self.is_recording = False
        self.dataset = None
        self.root_path = root_path
        self.current_task_description = None
        
        # define dataset features
        self.observation_features = {
            "gantry_position": {"shape": (3,), "dtype": "float32"},
            "gripper_imu_rot": {"shape": (3,), "dtype": "float32"},
            "laser_rangefinder": {"shape": (1,), "dtype": "float32"},
            "wrist_camera": {"shape": (image_size[0], image_size[1], 3), "dtype": "uint8"}, # 1/4 of native camera resolution
            "finger_pad_voltage": {"shape": (1,), "dtype": "float32"},
            "winch_length": {"shape": (1,), "dtype": "float32"},
        }
        self.action_features = {
            "gantry_velocity": {"shape": (3,), "dtype": "float32"},
            "winch_speed": {"shape": (1,), "dtype": "float32"},
            "finger_angle": {"shape": (1,), "dtype": "float32"},
        }


    def start_recording(self, repo_id: str, fps: int = 30):
        """Call this once to initialize the dataset file on disk."""
        if self.dataset:
            print("Warning: Recording is already active.")
            return

        obs_features = hw_to_dataset_features(self.observation_features, "observation", use_videos=True)
        act_features = hw_to_dataset_features(self.action_features, "action", use_videos=True)
        dataset_features = {**obs_features, **act_features}

        # create lerobot dataset instance
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=self.root_path,
            fps=fps,
            features=dataset_features,
            use_videos=True,
        )
        print(f"Dataset '{repo_id}' created successfully.")

    def start_episode(self, task_description: str):
        """
        Starts a new episode for a specific, described task.
        
        Args:
            task_description: A string describing the goal, e.g., "pick up the t-shirt".
        """
        if not self.dataset:
            print("Error: Must call start_recording() before starting an episode.")
            return
        self.is_recording = True
        self.current_task_description = task_description
        print(f"Episode started for task: '{self.current_task_description}'")

    def stop_episode(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.dataset.save_episode()
        self.current_task_description = None
        print("Episode saved.")

    def handle_training_vid_frame(self, timestamp, frame):
        """
        handle a timestamped frame of video by finding the latest recorded datapoint from every other sensor
        and storing them together as a single dataset frame.
        """

        if not self.is_recording or self.current_task_description is None:
            return

        winch = self.datastore.winch_line_record.getLast()
        finger = self.datastore.finger.getLast()
        imu = self.datastore.imu_rotvec.getLast()[1:]
        laser = self.datastore.range_record.getLast()[1:]

        # resize image.
        # TODO do this on pool and recover frame order in a consumer
        sized_frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_LINEAR)

        observation = {
            "gantry_position": pe.gant_pos, # treat kalman filter output as sensor
            "gripper_imu_rot": imu,
            "laser_rangefinder": laser,
            "wrist_camera": sized_frame,
            "finger_pad_voltage": finger[2],
            "winch_length": winch[1],
        }

        action  = {
            "gantry_velocity": pe.gant_vel,
            "winch_speed": winch[2],
            "finger_angle": finger[1],
        }

        # Build frames
        observation_frame = build_dataset_frame(self.dataset.features, observation, "observation")
        action_frame = build_dataset_frame(self.dataset.features, action, "action")
        
        # add the combined frame to the dataset, tagged with the current task
        frame = {**observation_frame, **action_frame}
        self.dataset.add_frame(frame, task=self.current_task_description)