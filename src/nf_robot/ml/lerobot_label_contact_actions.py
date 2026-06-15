#!/usr/bin/env python

"""Post-process a recorded dataset to populate the "episode_end" and "contact_vec_*"
action components, which are not knowable live during recording/teleop.

For each episode:
  - "episode_end" is set to 1.0 for frames within `--episode_end_seconds` of the end
    of the episode, else 0.0.
  - The "eventual contact position" is taken to be observation.state's gripper_pos_*
    at the first frame where finger_pressure exceeds `--pressure_threshold`. For every
    frame, "contact_vec_*" is the vector (in room-frame x/y/z) from that frame's
    gripper position to the eventual contact position. If no frame in an episode
    exceeds the pressure threshold, contact_vec is left as zeros for that episode.
  - If `--rotate_contact_vec` is passed, contact_vec's x/y components are additionally
    rotated into the gripper's frame of reference using that frame's "spin"
    (observation.state's spin, room->gripper rotation about the vertical axis); z is
    left unrotated. This requires "spin" to be present in observation.state - if it
    isn't, the script fails fast rather than silently leaving contact_vec in room frame.

If the dataset's action space is "gripper_vel" (5 dims), it is extended to
"gripper_vel_contact" (9 dims) by appending these 4 new components. If the action
space already contains contact_vec_*/episode_end (e.g. "dual_vel_contact", recorded
with zero placeholders), those columns are overwritten in place.

The script is idempotent: contact_vec_*/episode_end are recomputed from
observation.state (gripper_pos_*, gripper_rot_*, finger_pressure) and timestamp, none
of which this script modifies, so re-running it reproduces the same values (up to
float rounding).

Usage:
    python src/nf_robot/ml/lerobot_label_contact_actions.py \
        --repo_id naavox/simple_grasp_224 \
        [--new_repo_id naavox/simple_grasp_224_contact --new_root datasets/simple_grasp_224_contact] \
        [--rotate_contact_vec] \
        [--push_to_hub]
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nf_robot.ml.stringman_lerobot import _ACTION_SPACES, rotate_vector

CONTACT_ACTION_NAMES = ["contact_vec_x", "contact_vec_y", "contact_vec_z", "episode_end"]


def label_dataset(root: Path, pressure_threshold: float, episode_end_seconds: float, rotate_contact_vec: bool) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    action_feat = info["features"]["action"]
    obs_names = info["features"]["observation.state"]["names"]
    src_names = list(action_feat["names"])

    if rotate_contact_vec and "spin" not in obs_names:
        raise ValueError(
            "--rotate_contact_vec requires 'spin' to be present in observation.state features, "
            f"but it is not in {obs_names}"
        )

    obs_idx = {name: i for i, name in enumerate(obs_names)}
    gripper_pos_idx = [obs_idx[f"gripper_pos_{a}"] for a in "xyz"]
    pressure_idx = obs_idx["finger_pressure"]
    spin_idx = obs_idx["spin"] if rotate_contact_vec else None

    if all(name in src_names for name in CONTACT_ACTION_NAMES):
        dst_names = src_names
    elif src_names == _ACTION_SPACES["gripper_vel"]:
        dst_names = _ACTION_SPACES["gripper_vel_contact"]
    else:
        raise ValueError(
            f"Don't know how to add contact/episode_end labels to action space {src_names}"
        )
    contact_dst_idx = {name: dst_names.index(name) for name in CONTACT_ACTION_NAMES}

    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files found under {root}/data")

    # Pass 1: read everything we need to compute per-episode contact positions/durations.
    tables = {}
    episodes: dict[int, list[dict]] = {}
    for f in data_files:
        table = pq.read_table(f)
        tables[f] = table

        episode_indices = table.column("episode_index").to_pylist()
        frame_indices = table.column("frame_index").to_pylist()
        timestamps = table.column("timestamp").to_pylist()
        obs_states = table.column("observation.state").to_pylist()

        for row, (ep, fi, ts, state) in enumerate(zip(episode_indices, frame_indices, timestamps, obs_states)):
            episodes.setdefault(ep, []).append({
                "file": f,
                "row": row,
                "frame_index": fi,
                "timestamp": ts,
                "gripper_pos": np.array([state[i] for i in gripper_pos_idx], dtype=np.float64),
                "spin": state[spin_idx] if spin_idx is not None else None,
                "pressure": state[pressure_idx],
            })

    # Pass 2: compute new action component values for every row.
    new_values: dict[tuple[Path, int], dict[str, float]] = {}
    episodes_without_contact = 0
    for ep, rows in episodes.items():
        rows.sort(key=lambda r: r["frame_index"])

        contact_row = next((r for r in rows if r["pressure"] > pressure_threshold), None)
        if contact_row is None:
            episodes_without_contact += 1
        contact_pos = contact_row["gripper_pos"] if contact_row is not None else None

        episode_duration = rows[-1]["timestamp"]
        for r in rows:
            if contact_pos is None:
                contact_vec = np.zeros(3)
            else:
                contact_vec = contact_pos - r["gripper_pos"]
                if rotate_contact_vec:
                    contact_vec = contact_vec.copy()
                    contact_vec[:2] = rotate_vector(contact_vec[:2], r["spin"])

            episode_end = 1.0 if (episode_duration - r["timestamp"]) <= episode_end_seconds + 1e-6 else 0.0

            new_values[(r["file"], r["row"])] = {
                "contact_vec_x": float(contact_vec[0]),
                "contact_vec_y": float(contact_vec[1]),
                "contact_vec_z": float(contact_vec[2]),
                "episode_end": episode_end,
            }

    if episodes_without_contact:
        logging.warning(
            f"{episodes_without_contact}/{len(episodes)} episodes never exceeded "
            f"finger_pressure > {pressure_threshold}; contact_vec set to zero for those episodes."
        )

    # Pass 3: write the updated action column back to each file.
    action_type = pa.list_(pa.float32(), len(dst_names))
    all_new_actions = []
    for f, table in tables.items():
        actions = table.column("action").to_pylist()
        new_actions = []
        for row, a in enumerate(actions):
            extra = new_values[(f, row)]
            if dst_names is src_names:
                new_a = list(a)
                for name, val in extra.items():
                    new_a[contact_dst_idx[name]] = val
            else:
                new_a = list(a) + [extra[name] for name in CONTACT_ACTION_NAMES]
            new_actions.append(new_a)
        all_new_actions.extend(new_actions)

        action_col_idx = table.schema.get_field_index("action")
        table = table.set_column(action_col_idx, "action", pa.array(new_actions, type=action_type))
        pq.write_table(table, f)

    # Update info.json
    action_feat["shape"] = [len(dst_names)]
    action_feat["names"] = dst_names
    info_path.write_text(json.dumps(info, indent=4))

    # Update stats.json for the action feature
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    arr = np.array(all_new_actions, dtype=np.float64)
    stats["action"] = {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [arr.shape[0]],
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q10": np.quantile(arr, 0.10, axis=0).tolist(),
        "q50": np.quantile(arr, 0.50, axis=0).tolist(),
        "q90": np.quantile(arr, 0.90, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
    }
    stats_path.write_text(json.dumps(stats, indent=4))

    logging.info(f"Updated action feature: {src_names} -> {dst_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="repo id of the source dataset")
    parser.add_argument("--root", help="local root of the source dataset (default datasets/<name>)")
    parser.add_argument("--new_repo_id", help="repo id for the labeled dataset (default: edit --repo_id in place)")
    parser.add_argument("--new_root", help="local root for the labeled dataset (default datasets/<new name>)")
    parser.add_argument("--pressure_threshold", type=float, default=0.1, help="finger_pressure threshold marking contact")
    parser.add_argument("--episode_end_seconds", type=float, default=1.0, help="duration of the 'episode end' window")
    parser.add_argument("--rotate_contact_vec", action="store_true", help="rotate contact_vec x/y into the gripper's frame using observation.state's spin (requires 'spin' to be present)")
    parser.add_argument("--push_to_hub", action="store_true", help="upload the labeled dataset to the Hugging Face Hub")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    root = Path(args.root) if args.root else Path(f"datasets/{args.repo_id.split('/')[1]}")

    if args.new_repo_id or args.new_root:
        new_root = Path(args.new_root) if args.new_root else Path(f"datasets/{args.new_repo_id.split('/')[1]}")
        new_repo_id = args.new_repo_id or args.repo_id
        if new_root.exists():
            raise FileExistsError(f"{new_root} already exists")
        logging.info(f"Copying {root} -> {new_root}")
        shutil.copytree(root, new_root)
        work_root = new_root
        if new_repo_id != args.repo_id:
            info_path = work_root / "meta" / "info.json"
            info = json.loads(info_path.read_text())
            info["repo_id"] = new_repo_id
            info_path.write_text(json.dumps(info, indent=4))
    else:
        new_repo_id = args.repo_id
        work_root = root

    label_dataset(work_root, args.pressure_threshold, args.episode_end_seconds, args.rotate_contact_vec)

    if args.push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        logging.info(f"Pushing '{new_repo_id}' to the Hugging Face Hub")
        LeRobotDataset(repo_id=new_repo_id, root=work_root).push_to_hub()


if __name__ == "__main__":
    main()
