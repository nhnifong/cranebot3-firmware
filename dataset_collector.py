import numpy as np
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features

class LeRobotDatasetCollector:
    def __init__(self, root_path="last_dataset", datastore):
        
        self.datastore = datastore # reference to the class containing circular buffers of recent robot data
        self.is_recording = False
        self.dataset = None
        self.root_path = root_path
        self.current_task_description = None
        
        # 1. DEFINE YOUR HARDWARE FEATURES
        self.observation_features = {
            # The absolute position of the gantry is a crucial piece of state information.
            "gantry_position": {"shape": (3,), "dtype": "float32"},
            "imu": {"shape": (3,), "dtype": "float32"},
            "laser_rangefinder": {"shape": (1,), "dtype": "float32"},
            "wrist_camera": {"shape": (576, 324, 3), "dtype": "uint8"}, # 1/4 of native camera resolution
            "finger_pressure": {"shape": (1,), "dtype": "float32"},
            "winch_length": {"shape": (1,), "dtype": "float32"},
        }
        self.action_features = {
            # The relative change in position is the action the policy will learn to output.
            "relative_gantry_movement": {"shape": (3,), "dtype": "float32"},
            "winch_speed": {"shape": (1,), "dtype": "float32"},
            "commanded_finger_angle": {"shape": (1,), "dtype": "float32"},
        }

        # This buffer will hold the *most recent* (timestamp, data) tuple for each sensor
        self.latest_observations = {}
        self.last_gantry_position = None

    def start_recording(self, repo_id: str, fps: int = 30):
        """Call this once to initialize the dataset file on disk."""
        if self.dataset:
            print("Warning: Recording is already active.")
            return

        # 2. CREATE THE FEATURES DICTIONARY
        obs_features = hw_to_dataset_features(self.observation_features, "observation", use_videos=True)
        act_features = hw_to_dataset_features(self.action_features, "action", use_videos=True)
        dataset_features = {**obs_features, **act_features}

        # 3. CREATE THE LEROBOT DATASET INSTANCE
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
        self.latest_observations.clear()
        self.last_gantry_position = None
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
        winch = self.datastore.winch_line_record.getLast()

        # resize image
        sized_frame = frame
        gantry_position = pe.gant_pos

        observation = {
            "wrist_camera": sized_frame,
            "gantry_position": pe.gant_pos,
            "relative_gantry_movement": pe.gant_vel,
            "winch_length": winch[1],
            "winch_speed": winch[2],
            "imu": self.datastore.imu_rotvec.getLast()[1:],
            "laser_rangefinder": self.datastore.range_record.getLast()[1:],
            "finger_pressure": self.datastore.finger_pressure.getLast()[1:],
            "commanded_finger_angle": self.datastore.commanded_finger_angle.getLast()[1:],
        }

        last_gantry_position = gantry_position




    def add_observation(self, source: str, data):
        """
        Updates the most recent value for a given sensor, along with the
        current timestamp for staleness checking.
        """
        if not self.is_recording:
            return
        self.latest_observations[source] = (time.time(), data)

    def add_timestep(self, wrist_camera_image, relative_gantry_movement, commanded_finger_angle, gripper_winch_speed):
        """
        This is called every time you get a new camera frame.
        It checks for data staleness, stores the relative gantry movement,
        and adds one complete, aligned frame to the dataset.
        """
        STALE_DATA_THRESHOLD_S = 0.5 # seconds

        if not self.is_recording or self.current_task_description is None:
            return

        now = time.time()
        
        # --- Staleness Check ---
        required_sources = ["gantry_position", "imu", "laser_rangefinder", "sensed_finger_pressure"]
        for source in required_sources:
            if source not in self.latest_observations:
                # Skip frame if we haven't received any data from this source yet
                return
            
            timestamp, _ = self.latest_observations[source]
            if now - timestamp > STALE_DATA_THRESHOLD_S:
                print(f"Warning: Discarding frame due to stale '{source}' data.")
                return

        # --- Build and Save Frame ---
        # Prepare the observation dictionary for this timestep
        observation = {
            "wrist_camera": wrist_camera_image,
            "gantry_position": self.latest_observations.get("gantry_position", (0, np.zeros(3)))[1],
            "imu": self.latest_observations.get("imu", (0, np.zeros(6)))[1],
            "laser_rangefinder": self.latest_observations.get("laser_rangefinder", (0, 0.0))[1],
            "sensed_finger_pressure": self.latest_observations.get("sensed_finger_pressure", (0, 0.0))[1],
        }
        
        # Prepare the action dictionary from the arguments
        action = {
            "relative_gantry_movement": relative_gantry_movement,
            "gripper_winch_speed": gripper_winch_speed,
            "commanded_finger_angle": commanded_finger_angle,
        }

        # 4. BUILD THE FRAMES
        observation_frame = build_dataset_frame(self.dataset.features, observation, "observation")
        action_frame = build_dataset_frame(self.dataset.features, action, "action")
        
        # 5. ADD THE COMBINED FRAME TO THE DATASET, TAGGED WITH THE CURRENT TASK
        frame = {**observation_frame, **action_frame}
        self.dataset.add_frame(frame, task=self.current_task_description)
