#!/usr/bin/env python

"""Detect which robot recorded each episode of a dataset, from the recorded spin.

get_spin() computes `spin = radians(wrist_angle) + (frameRoomSpin - pi)`, and both
`spin` and `wrist_angle` are stored in observation.state, so every frame carries the
frameRoomSpin that was configured when it was recorded:

    frameRoomSpin = spin - radians(wrist_angle) + pi

It is constant to ~1e-6 rad within a recording session, so a dataset that spans two
robots - or one robot recalibrated partway through - shows up as a step. That makes
this a cheap way to find where to split a dataset: only meta/ and data/ are
downloaded, no video.

Values are matched against the configs in a calibrations directory, but a dataset
whose value matches nothing is normal: frameRoomSpin changes every time spin
calibration is run, so a config file is only correct for the sessions recorded
between one calibration and the next.

Usage:
    python experiments/check_dataset_robots.py --recipe src/nf_robot/ml/recipes/move_clutter_camera_goal.yaml
    python experiments/check_dataset_robots.py --repo_ids naavox/move_clutter justink04/laundry-in-hamper-8-1-26
"""

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

# Two sessions are the same robot+calibration if their constants agree this closely.
SAME_SESSION_RAD = 0.005


def recovered_constants(repo_id):
    """(per-episode frameRoomSpin, episode indices) for a dataset."""
    root = snapshot_download(repo_id=repo_id, repo_type="dataset", allow_patterns=["data/**", "meta/**"])
    info = json.loads(open(os.path.join(root, "meta/info.json")).read())
    names = info["features"]["observation.state"]["names"]
    if "spin" not in names or "wrist_angle" not in names:
        raise ValueError("observation.state has no spin/wrist_angle")

    states, episodes = [], []
    for f in sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True)):
        table = pq.read_table(f, columns=["observation.state", "episode_index"])
        states.append(np.array(table.column("observation.state").to_pylist(), dtype=np.float64))
        episodes.append(np.array(table.column("episode_index").to_pylist()))
    states = np.concatenate(states)
    episodes = np.concatenate(episodes)

    spin = states[:, names.index("spin")]
    wrist = states[:, names.index("wrist_angle")]
    per_frame = (spin - np.radians(wrist) + 2 * np.pi) % (2 * np.pi) - np.pi
    order = sorted(np.unique(episodes))
    return np.array([np.median(per_frame[episodes == e]) for e in order]), order


def runs_of_same_value(per_episode, episodes):
    """Consecutive episodes sharing a constant, as [(first, last, value), ...]."""
    runs, start = [], 0
    for i in range(1, len(per_episode) + 1):
        if i == len(per_episode) or abs(per_episode[i] - per_episode[start]) > SAME_SESSION_RAD:
            runs.append((episodes[start], episodes[i - 1], float(np.median(per_episode[start:i]))))
            start = i
    return runs


def load_configs(directory):
    """frameRoomSpin of every config under a directory, searched recursively."""
    configs = {}
    for path in sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True)):
        try:
            value = json.loads(open(path).read())["gripper"]["frameRoomSpin"]
        except (KeyError, ValueError):
            continue
        configs[os.path.relpath(path, directory)] = float(value)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", help="build recipe whose merge sources are checked")
    parser.add_argument("--repo_ids", nargs="+", help="datasets to check instead of a recipe")
    parser.add_argument("--calibrations", default="src/nf_robot/ml/calibrations")
    args = parser.parse_args()

    if args.recipe:
        import yaml
        merge = yaml.safe_load(open(args.recipe).read())["merge"]
        repo_ids = list(dict.fromkeys(e if isinstance(e, str) else e["repo_id"] for e in merge))
    elif args.repo_ids:
        repo_ids = args.repo_ids
    else:
        parser.error("give --recipe or --repo_ids")

    configs = load_configs(args.calibrations)
    print(f"known configs: " + ", ".join(f"{k}={v:+.4f}" for k, v in configs.items()) + "\n")

    split = []
    for repo_id in repo_ids:
        try:
            per_episode, episodes = recovered_constants(repo_id)
        except Exception as e:
            print(f"{repo_id}\n   ERROR {type(e).__name__}: {e}")
            continue
        runs = runs_of_same_value(per_episode, episodes)
        flag = "  <-- MORE THAN ONE ROBOT/CALIBRATION" if len(runs) > 1 else ""
        print(f"{repo_id}  ({len(episodes)} episodes, {len(runs)} session(s)){flag}")
        for first, last, value in runs:
            match = min(configs, key=lambda k: abs(configs[k] - value)) if configs else None
            error = abs(configs[match] - value) if match else float("inf")
            named = f"{match} (off by {error:.4f})" if error < SAME_SESSION_RAD else \
                    f"no config matches, nearest {match} off by {error:.3f}" if match else "no configs"
            print(f"   episodes {first:>4}-{last:<4} frameRoomSpin {value:+.6f}   {named}")
        if len(runs) > 1:
            split.append(repo_id)
        print()

    if split:
        print("Datasets needing to be listed once per session in a recipe:")
        for repo_id in split:
            print(f"  {repo_id}")


if __name__ == "__main__":
    main()
