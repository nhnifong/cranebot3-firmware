import argparse
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
import sys
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

VOLTAGE_THRESHOLD = 0.4

def get_feature_indices(features):
    """Helper function to get the indices of our key features from the dataset metadata."""
    try:
        state_names = features["observation.state"]["names"]
        voltage_idx = state_names.index("finger_pad_voltage")

        action_names = features["action"]["names"]
        gantry_x_idx = action_names.index("gantry_pos_x")
        gantry_y_idx = action_names.index("gantry_pos_y")
        gantry_z_idx = action_names.index("gantry_pos_z")
        winch_idx = action_names.index("winch_line_length")
        finger_idx = action_names.index("finger_angle")

        # Gantry/winch indices (the ones we will overwrite)
        goal_action_indices = [gantry_x_idx, gantry_y_idx, gantry_z_idx, winch_idx]
        
        return voltage_idx, goal_action_indices, finger_idx
    except (ValueError, KeyError) as e:
        logger.error(f"FATAL: A required feature is missing from the dataset: {e}")
        logger.error(f"State features found: {features.get('observation.state', {}).get('names')}")
        logger.error(f"Action features found: {features.get('action', {}).get('names')}")
        sys.exit(1)


def process_dataset(input_repo_id, output_repo_id):
    """
    Loads an existing LeRobot dataset, relabels actions based on goals,
    and creates a new dataset.
    """
    
    # 1. Load the input dataset
    logger.info(f"Loading input dataset: {input_repo_id}")
    try:
        in_dataset = LeRobotDataset(input_repo_id)
    except Exception as e:
        logger.error(f"Failed to load dataset '{input_repo_id}'. Make sure it exists and you are logged in.")
        logger.error(e)
        return

    logger.info("Dataset loaded. Extracting metadata...")
    features = in_dataset.features
    camera_keys = in_dataset.meta.camera_keys # <-- ADD THIS
    
    # Get the indices for the features we need to read and write
    voltage_idx, goal_action_indices, finger_idx = get_feature_indices(features)
    
    # 2. Create the output dataset
    logger.info(f"Creating new dataset repo: {output_repo_id}")
    try:
        # Ensure the repo exists
        api = HfApi()
        create_repo(output_repo_id, repo_type="dataset", private=True, exist_ok=True)
        
        out_dataset = LeRobotDataset.create(
            repo_id=output_repo_id,
            features=features,
            fps=in_dataset.fps,
            robot_type='stringhman',
            use_videos=True,
            vcodec='libsvtav1',
        )
        out_dataset.start_image_writer()
        out_dataset.start_async_video_encoder()
    except Exception as e:
        logger.error(f"Failed to create output dataset '{output_repo_id}'.")
        logger.error(e)
        return

    logger.info("Starting episode processing...")
    
    # --- FIX 1: Correct attribute name ---
    if not hasattr(in_dataset, 'meta') or not hasattr(in_dataset.meta, 'episodes') or in_dataset.meta.episodes is None:
        logger.error("FATAL: Dataset object is missing '.meta.episodes' attribute.")
        logger.error("This might be an error in loading. Halting.")
        return

    # --- FIX 2: Correct iteration logic ---
    # in_dataset.meta.episodes is a list-like object (e.g., loaded from parquet)
    # with metadata for each episode.
    episode_metas = in_dataset.meta.episodes
    original_episode_count = len(episode_metas)
    skipped_episodes = 0

    # 3. Iterate through each episode in the original dataset
    for i in tqdm(range(original_episode_count), desc="Processing episodes"):
        # Get the metadata for this specific episode
        ep_meta = episode_metas[i]
        
        # Get the global start and end frame index for this episode
        # "dataset_from_index" is inclusive
        # "dataset_to_index" is exclusive (it's the index *after* the last frame)
        start_idx = ep_meta["dataset_from_index"]
        end_idx = ep_meta["dataset_to_index"] # This is exclusive

        # --- FIX: Load episode frame-by-frame ---
        # We must build the episode_frames list manually because __getitem__ doesn't take slices
        episode_frames_list = []
        for frame_idx in range(start_idx, end_idx):
            episode_frames_list.append(in_dataset[frame_idx])

        # --- Convert list of dicts to dict of tensors (like slicing would have done) ---
        # This is what the rest of the script expects for finding goals
        all_keys = episode_frames_list[0].keys()
        episode_frames = {}
        for key in all_keys:
            # Stack all tensors for this key
            # This check is needed because some items like 'task' are strings
            if isinstance(episode_frames_list[0][key], torch.Tensor):
                episode_frames[key] = torch.stack([frame[key] for frame in episode_frames_list])
            else:
                # Handle non-tensor data like 'task' string
                episode_frames[key] = [frame[key] for frame in episode_frames_list]

        
        # --- Find Goals ---
        finger_pad_voltage = episode_frames["observation.state"][:, voltage_idx]
        
        # Find the first frame where grasp happens
        grasp_indices = torch.where(finger_pad_voltage > VOLTAGE_THRESHOLD)[0]

        if len(grasp_indices) == 0:
            # This episode is "bad" (no grasp), skip it.
            skipped_episodes += 1
            continue
            
        grasp_idx = grasp_indices[0].item()
        
        # Goal 1: The action vector from the grasp frame
        # We get this from the *original* actions we just loaded
        original_actions = episode_frames["action"]
        over_sock_goal_action = original_actions[grasp_idx]
        
        # Goal 2: The action vector from the last frame
        over_bin_goal_action = original_actions[-1]

        # --- Write New Frames ---
        num_frames = len(episode_frames_list)
        
        for t in range(num_frames):
            # 1. Get the original frame from our list
            new_frame = episode_frames_list[t]
            
            # 2. Get a clone of the original action (to preserve finger_angle)
            modified_action = new_frame["action"].clone()
            
            # 3. Decide which goal to use
            voltage = finger_pad_voltage[t]
            
            if voltage < 1.0:
                # Use "over_sock" goal
                target_goal = over_sock_goal_action
            else:
                # Use "over_bin" goal
                target_goal = over_bin_goal_action
                
            # 4. Overwrite the gantry/winch actions with the chosen goal
            modified_action[goal_action_indices] = target_goal[goal_action_indices]
            
            # 5. Put the modified action back into the frame
            new_frame["action"] = modified_action
            
            # --- FIX: Pre-process frame before add_frame ---
            
            # FIX A: Remove metadata keys that add_frame creates automatically
            # These keys are returned by __getitem__ but are not expected by add_frame
            metadata_keys = ['index', 'task_index', 'frame_index', 'timestamp', 'episode_index']
            for key in metadata_keys:
                if key in new_frame:
                    del new_frame[key]
                    
            # FIX B: Permute images from (C, H, W) to (H, W, C) and convert to numpy
            # __getitem__ returns torch tensors (C,H,W), but add_frame expects (H,W,C)
            for key in camera_keys:
                if key in new_frame:
                    # Convert to (H, W, C) and to numpy array
                    new_frame[key] = new_frame[key].permute(1, 2, 0).numpy()

            # 6. Add the relabeled frame to the new dataset
            out_dataset.add_frame(new_frame)
            
        # Save the newly processed episode
        out_dataset.save_episode()

    logger.info("Processing complete.")
    logger.info(f"Total episodes processed: {original_episode_count - skipped_episodes}")
    logger.info(f"Total episodes skipped (no grasp detected): {skipped_episodes}")

    # 4. Finalize and upload
    logger.info("Stopping video encoder and pushing to hub...")
    out_dataset.stop_async_video_encoder(wait=True)
    out_dataset.push_to_hub()
    logger.info(f"Successfully created and uploaded dataset to {output_repo_id}")

def main():
    parser = argparse.ArgumentParser(
        description="Relabel a LeRobot dataset with goal-conditioned actions."
    )
    parser.add_argument(
        "--input_repo_id",
        type=str,
        required=True,
        help="The Hugging Face repo ID of the source dataset (e.g., 'naavox/merged-3').",
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        required=True,
        help="The Hugging Face repo ID for the new relabeled dataset (e.g., 'naavox/merged-3-relabeled').",
    )
    args = parser.parse_args()

    process_dataset(args.input_repo_id, args.output_repo_id)

if __name__ == "__main__":
    main()