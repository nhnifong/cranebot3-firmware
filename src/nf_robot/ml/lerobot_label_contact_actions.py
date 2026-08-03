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


def contact_blend_alphas(timestamps, pressures, pressure_threshold: float, blend_seconds: float):
    """Per-frame blend from the contact target toward the episode-end target.

    Returns (contact_index, alphas) with alpha 0 up to contact and ramping to 1 over
    blend_seconds after it, or (None, None) when the episode never reaches the pressure
    threshold. Callers apply the alphas to whatever they are blending - the gripper
    position here, the wrist angle in camera_goal.py - so the two cannot disagree about
    when contact happened.
    """
    contact_index = next((i for i, p in enumerate(pressures) if p > pressure_threshold), None)
    if contact_index is None:
        return None, None
    contact_ts = timestamps[contact_index]
    alphas = []
    for ts in timestamps:
        t_after = max(0.0, ts - contact_ts)
        alphas.append(min(t_after / blend_seconds, 1.0) if blend_seconds > 0 else float(t_after > 0))
    return contact_index, alphas


# A stop counts as a waypoint only if the gantry was slower than this for at least
# this long: brief dips through zero while reversing are not places it went to.
# Tuned against naavox/nick-aug3-2: operators slow down rather than fully stop, so a
# threshold near zero finds almost nothing. 0.06 m/s sits on a plateau where 0.05-0.2s
# of rest all recover the same six stops in that episode.
REST_SPEED_MPS = 0.06
MIN_REST_S = 0.1


def motion_waypoints(timestamps, speeds, rest_speed_mps=REST_SPEED_MPS, min_rest_s=MIN_REST_S):
    """Frame indices where the gantry left a rest, i.e. where it had been holding a position.

    The operator drives to somewhere, pauses, then drives on; the position held during
    each pause is a place they meant to reach. Those are the indices returned, which are
    the frames a target should point at while the gantry is on its way there.
    """
    moving = [s > rest_speed_mps for s in speeds]
    waypoints = []
    for i in range(1, len(moving)):
        if not (moving[i] and not moving[i - 1]):
            continue
        start = i - 1
        while start > 0 and not moving[start - 1]:
            start -= 1
        if timestamps[i - 1] - timestamps[start] >= min_rest_s:
            waypoints.append(i - 1)
    return waypoints


def label_dataset(root: Path, pressure_threshold: float, episode_end_seconds: float, rotate_contact_vec: bool,
                  blend_seconds: float = 0.5, mode: str = "contact",
                  rest_speed_mps: float = REST_SPEED_MPS, min_rest_s: float = MIN_REST_S) -> None:
    """Fill in the contact_vec_*/episode_end action components.

    mode="contact" points contact_vec at the grasp position, then blends to the episode's
    final position over blend_seconds - two waypoints per episode.

    mode="waypoints" points it at the next place the gantry actually stopped: every
    position it held before setting off again, plus the grasp position and the episode's
    final position. The target is piecewise constant and steps as each is reached, which
    suits an action space that predicts where to go rather than how fast to move. No
    blending is applied, since the steps are the signal.
    """
    if mode not in ("contact", "waypoints"):
        raise ValueError(f"unknown labelling mode {mode!r}; expected 'contact' or 'waypoints'")
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
    # the recorded gantry velocity, used by mode="waypoints" to find where it stopped
    velocity_idx = (
        [obs_idx[f"vel_{a}"] for a in "xyz"]
        if all(f"vel_{a}" in obs_idx for a in "xyz") else None
    )
    if mode == "waypoints" and velocity_idx is None:
        raise ValueError("mode='waypoints' needs vel_x/vel_y/vel_z in observation.state")

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
                "speed": (
                    float(np.linalg.norm([state[i] for i in velocity_idx]))
                    if velocity_idx is not None else None
                ),
            })

    # Pass 2: compute new action component values for every row.
    new_values: dict[tuple[Path, int], dict[str, float]] = {}
    episodes_without_contact = 0
    waypoints_used: list[int] = []
    for ep, rows in episodes.items():
        rows.sort(key=lambda r: r["frame_index"])

        contact_index, alphas = contact_blend_alphas(
            [r["timestamp"] for r in rows], [r["pressure"] for r in rows],
            pressure_threshold, blend_seconds,
        )
        if contact_index is None:
            episodes_without_contact += 1
        contact_pos = rows[contact_index]["gripper_pos"] if contact_index is not None else None
        episode_end_pos = rows[-1]["gripper_pos"]

        targets = None
        if mode == "waypoints":
            # Every position the gantry held before setting off again, plus the grasp and
            # the episode's last position. Sorted so each frame can look ahead to the next.
            stops = motion_waypoints(
                [r["timestamp"] for r in rows], [r["speed"] for r in rows],
                rest_speed_mps, min_rest_s,
            )
            indices = sorted({*stops, *( [contact_index] if contact_index is not None else [] ), len(rows) - 1})
            waypoints_used.append(len(indices))
            targets = []
            nxt = 0
            for i in range(len(rows)):
                while nxt < len(indices) - 1 and indices[nxt] < i:
                    nxt += 1
                targets.append(rows[indices[nxt]]["gripper_pos"])

        episode_duration = rows[-1]["timestamp"]
        for i, r in enumerate(rows):
            if targets is not None:
                target_pos = targets[i]
            elif contact_pos is None:
                contact_vec = np.zeros(3)
                target_pos = None
            else:
                # Before contact: point toward contact position.
                # After contact: blend toward the episode-end position over blend_seconds,
                # so the model is guided through pick-up and to the final resting location.
                alpha = alphas[i]
                target_pos = (1.0 - alpha) * contact_pos + alpha * episode_end_pos

            if target_pos is not None:
                contact_vec = target_pos - r["gripper_pos"]
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

    if waypoints_used:
        logging.info(
            f"mode='waypoints': {np.mean(waypoints_used):.1f} waypoints per episode "
            f"(min {min(waypoints_used)}, max {max(waypoints_used)})"
        )
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
    parser.add_argument("--mode", default="contact", choices=["contact", "waypoints"],
                        help="'contact' targets the grasp then the episode end; 'waypoints' targets "
                             "the next position the gantry actually stopped at")
    parser.add_argument("--rest_speed_mps", type=float, default=REST_SPEED_MPS,
                        help="gantry speed below which it counts as stopped (mode='waypoints')")
    parser.add_argument("--min_rest_s", type=float, default=MIN_REST_S,
                        help="how long it must stay stopped to count as a waypoint (mode='waypoints')")
    parser.add_argument("--blend_seconds", type=float, default=0.5, help="seconds after contact over which contact_vec blends from pointing at contact position to pointing at episode-end position")
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

    label_dataset(work_root, args.pressure_threshold, args.episode_end_seconds, args.rotate_contact_vec,
                  args.blend_seconds, mode=args.mode,
                  rest_speed_mps=args.rest_speed_mps, min_rest_s=args.min_rest_s)

    if args.push_to_hub:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        logging.info(f"Pushing '{new_repo_id}' to the Hugging Face Hub")
        LeRobotDataset(repo_id=new_repo_id, root=work_root).push_to_hub()


if __name__ == "__main__":
    main()
