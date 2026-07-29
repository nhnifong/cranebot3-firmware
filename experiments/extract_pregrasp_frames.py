#!/usr/bin/env python

"""Extract one frame per episode from a LeRobot dataset, taken a fixed time
before the gripper first makes contact with an object.

"Contact" is the first frame in the episode whose observation.state
finger_pressure exceeds --pressure_threshold (same convention as
lerobot_label_contact_actions.py). Episodes that never reach that threshold are
skipped.

Usage:
    python experiments/extract_pregrasp_frames.py \
        --repo_id naavox/move_clutter_combined_384_2 \
        --camera observation.images.gripper_camera \
        --seconds_before 1.0 \
        --output_dir ./pregrasp_frames
"""

import argparse
import os

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def extract(repo_id, camera_key, seconds_before, output_dir, root=None,
            pressure_threshold=0.1, ext="png"):
    print(f"Loading lerobot dataset {repo_id}...")
    dataset = LeRobotDataset(repo_id=repo_id, root=root)

    if camera_key not in dataset.meta.video_keys:
        raise ValueError(
            f"'{camera_key}' is not a video feature of {repo_id}. "
            f"Available: {dataset.meta.video_keys}"
        )

    obs_names = dataset.meta.features["observation.state"]["names"]
    if "finger_pressure" not in obs_names:
        raise ValueError(f"observation.state has no 'finger_pressure' component: {obs_names}")
    pressure_idx = obs_names.index("finger_pressure")

    os.makedirs(output_dir, exist_ok=True)

    # Pull just the columns needed to locate contact, avoiding video decoding.
    cols = dataset.hf_dataset.select_columns(["observation.state", "timestamp"])

    saved = 0
    skipped = []
    for ep_idx in range(dataset.meta.total_episodes):
        ep = dataset.meta.episodes[ep_idx]
        ep_from, ep_to = ep["dataset_from_index"], ep["dataset_to_index"]
        if ep_to <= ep_from:
            continue

        rows = cols[ep_from:ep_to]
        pressure = np.asarray(rows["observation.state"], dtype=np.float32)[:, pressure_idx]
        timestamps = np.asarray(rows["timestamp"], dtype=np.float64)

        contact = np.flatnonzero(pressure > pressure_threshold)
        if len(contact) == 0:
            skipped.append(ep_idx)
            continue
        contact_i = int(contact[0])

        # Last frame at or before the target time; clamped to the episode start
        # so short lead-ins still yield a frame.
        target_ts = timestamps[contact_i] - seconds_before
        offset = max(int(np.searchsorted(timestamps, target_ts, side="right")) - 1, 0)

        item = dataset[ep_from + offset]
        rgb = (item[camera_key].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        fn = os.path.join(output_dir, f"episode_{ep_idx:04d}.{ext}")
        cv2.imwrite(fn, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        saved += 1
        print(f"episode {ep_idx}: contact at t={timestamps[contact_i]:.2f}s, "
              f"saved frame {offset} (t={timestamps[offset]:.2f}s) -> {fn}")

    print(f"\nSaved {saved} frame(s) to {output_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} episode(s) with no finger_pressure > "
              f"{pressure_threshold}: {skipped}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="dataset repo id, e.g. naavox/move_clutter_combined_384_2")
    parser.add_argument("--camera", required=True, help="video feature key, e.g. observation.images.gripper_camera")
    parser.add_argument("--seconds_before", type=float, required=True,
                        help="seconds before first finger pressure to grab the frame")
    parser.add_argument("--output_dir", required=True, help="directory to write frames into")
    parser.add_argument("--root", help="local root of the dataset (default: HF cache, downloading if needed)")
    parser.add_argument("--pressure_threshold", type=float, default=0.1,
                        help="finger_pressure threshold marking contact")
    parser.add_argument("--ext", default="png", choices=["png", "jpg"], help="output image format")
    args = parser.parse_args()

    extract(args.repo_id, args.camera, args.seconds_before, args.output_dir,
            root=args.root, pressure_threshold=args.pressure_threshold, ext=args.ext)


if __name__ == "__main__":
    main()
