import os
import shutil
import argparse
import json
import numpy as np
import glob
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, HfApi

# Indices based on the StringmanLeRobot observation and action feature definitions
# actions: [vel_x, vel_y, vel_z, wrist_speed, finger_speed]
FINGER_SPEED_IDX = 4
# observation.state: [vel_x, vel_y, vel_z, wrist_speed, finger_speed, pos(3), rot(6), finger_angle, ...]
FINGER_ANGLE_IDX = 14
FPS = 60.0
OFFSET = 6

def moving_average(arr, window_size=8):
    """Applies a simple moving average with edge padding to avoid drop-offs."""
    pad_width = window_size // 2
    padded = np.pad(arr, (pad_width, pad_width), mode='edge')
    return np.convolve(padded, np.ones(window_size)/window_size, mode='valid')

def fix_dataset(input_repo: str, output_repo: str, local_dir: str):
    print(f"Downloading dataset repository '{input_repo}' to '{local_dir}'...")
    # 1. Download the entire repo. This pulls the videos (LFS), meta files, and original parquets.
    # local_dir_use_symlinks=False is CRITICAL so we get real files to upload, not ignored symlinks.
    snapshot_download(
        repo_id=input_repo,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print("Modifying Parquet files directly in-place to preserve all original schema and data...")
    data_dir = os.path.join(local_dir, "data")
    parquet_files = glob.glob(os.path.join(data_dir, "**/*.parquet"), recursive=True)

    if not parquet_files:
        print("No parquet files found! Check the downloaded repository structure.")
        return

    all_actions_list = []

    # 2. Process each parquet file directly using PyArrow
    for pq_file in parquet_files:
        print(f"Processing {os.path.basename(pq_file)}...")
        table = pq.read_table(pq_file)

        # Extract only the necessary columns into numpy arrays
        actions = np.array(table.column('action').to_pylist(), dtype=np.float32)
        obs_states = np.array(table.column('observation.state').to_pylist(), dtype=np.float32)
        ep_indices = np.array(table.column('episode_index').to_pylist())

        unique_eps = np.unique(ep_indices)

        for ep in unique_eps:
            # Find all frames for this specific episode
            idx = np.where(ep_indices == ep)[0]

            # Extract finger angles and apply smoothing
            raw_finger_angles = obs_states[idx, FINGER_ANGLE_IDX]
            finger_angles = moving_average(raw_finger_angles, window_size=5)

            # Compute finger speed (rate of change per second)
            speeds = np.zeros(len(idx), dtype=np.float32)
            speeds[1:] = (finger_angles[1:] - finger_angles[:-1]) * FPS

            # Zero out original bugged finger speeds for the episode
            actions[idx, FINGER_SPEED_IDX] = 0.0

            # Overwrite finger speed 6 frames earlier
            for t in range(len(idx)):
                target_t = t - OFFSET
                if target_t >= 0:
                    actions[idx[target_t], FINGER_SPEED_IDX] = speeds[t]

        # Save modified actions for global stats computation
        all_actions_list.append(actions)

        # Create a new PyArrow array with the modified data, preserving the original type
        action_col_type = table.schema.field('action').type
        new_action_array = pa.array(actions.tolist(), type=action_col_type)

        # Replace the old action column in the PyArrow Table
        action_idx = table.schema.get_field_index('action')
        new_table = table.set_column(action_idx, table.schema.field('action'), new_action_array)

        # Write back the table in-place, preserving exactly the original schema
        pq.write_table(new_table, pq_file)

    print("Updating dataset statistics in meta/stats.json...")
    stats_file = os.path.join(local_dir, "meta", "stats.json")
    if os.path.exists(stats_file) and all_actions_list:
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            # Combine all processed actions from all chunks
            all_actions = np.vstack(all_actions_list)

            if "action" in stats:
                stats["action"]["min"] = np.min(all_actions, axis=0).tolist()
                stats["action"]["max"] = np.max(all_actions, axis=0).tolist()
                stats["action"]["mean"] = np.mean(all_actions, axis=0).tolist()
                stats["action"]["std"] = np.std(all_actions, axis=0, ddof=0).tolist()

                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=4)
                print("Successfully updated action stats.")
        except Exception as e:
            print(f"Warning: Failed to update stats.json automatically. Error: {e}")
    else:
        print("No meta/stats.json found, skipping stats update.")

    print(f"\nRepair complete! The modified dataset sits locally in '{local_dir}'.")

    # 4. Prompt for Hugging Face upload
    response = input(f"Do you want to upload this fixed dataset to '{output_repo}' on the Hub? (y/n): ")
    if response.lower().strip() in ['y', 'yes']:
        print(f"Uploading to {output_repo}...")
        api = HfApi()
        try:
            api.create_repo(repo_id=output_repo, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=local_dir,
                repo_id=output_repo,
                repo_type="dataset",
                commit_message="Fix finger_speed telemetry bug via angle delta offset"
            )
            print(f"Upload complete! URL: https://huggingface.co/datasets/{output_repo}")
        except Exception as e:
            print(f"Failed to upload to HF: {e}")
    else:
        print("Upload skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair LeRobot dataset finger speed telemetry")
    parser.add_argument("--input_repo", default="naavox/grasping_dataset_eggs", help="Original dataset HF repo")
    parser.add_argument("--output_repo", default="naavox/grasping_dataset_eggs_fix", help="Fixed dataset HF repo")
    parser.add_argument("--local_dir", default="./fixed_dataset_workspace", help="Local workspace folder")
    args = parser.parse_args()

    fix_dataset(args.input_repo, args.output_repo, args.local_dir)