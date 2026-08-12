#!/usr/bin/env python

"""Predict where the robot should reach next, in the ortho floor view's pixel space.

A separate approach to nf_robot.ml.target_heatmap, which learns from hand-labelled
anchor camera frames. Here the labels come from teleop recordings instead: wherever
an operator actually grasped something is by construction a place worth reaching
for, so every episode donates one label for free.

Two stages, because the intermediate is a LeRobot dataset and the result is not:

  1. recipes/ortho_target.yaml merges the teleop datasets down to the ortho feed
     alone and runs contact labelling:

       python src/nf_robot/ml/lerobot_build_dataset.py \
           --recipe src/nf_robot/ml/recipes/ortho_target.yaml \
           --temp_dir /home/nhn/data_scratch \
           --output_root /home/nhn/data_scratch/move_clutter_ortho

  2. `distill` reduces that to one sample per episode - the episode's first ortho
     frame and the ortho pixel where contact eventually happened - then uploads the
     result, which is a few hundred MB rather than a few hundred GB:

       python -m nf_robot.ml.ortho_target distill \
           --repo_id naavox/move_clutter_ortho \
           --root /home/nhn/data_scratch/move_clutter_ortho \
           --upload

The ortho view is an orthographic projection of the floor plane (host/floor_view.py),
so room metres map to its pixels analytically - no camera pose is involved, unlike
camera_goal.py's per-anchor projection.

The distilled layout matches target_heatmap's image folder (train/eval split, a
metadata.jsonl of {"file_name", "points"} rows), so both datasets can feed the same
kind of heatmap model.

Caveats worth knowing before trusting the labels:

  - The projection assumes contact happens on the floor plane. An object grasped at
    height z appears displaced in the ortho view by the anchor cameras' parallax, so
    the label drifts outward from the room centre as z grows. contact_m carries the
    full 3D position, so samples can be filtered on z later.
  - Contact position is recomputed here from observation.state rather than read from
    the labelled contact_vec_* action components. Both use the same definition (see
    contact_blend_alphas), but an episode that never reaches the pressure threshold
    gets a zeroed contact_vec, which is indistinguishable from "contact at the
    starting position". Reading the pressure directly is what makes those episodes
    skippable instead of silently mislabelled.
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from nf_robot.ml.lerobot_label_contact_actions import contact_blend_alphas
from nf_robot.ml.stringman_lerobot import _FEED_NAMES

ORTHO_FEED = 3
ORTHO_KEY = f"observation.images.{_FEED_NAMES[ORTHO_FEED]}"
ORTHO_CAMERA_MODE = "ortho_512"

# Metres of floor the square ortho map spans, per side, centred on the room origin.
# host/floor_view.py's EXTENT_M and the map_extent_meters observer._ortho_worker
# renders with; a recording made with a different extent would need its own value.
ORTHO_EXTENT_M = 5.0

DEFAULT_SOURCE_REPO_ID = "naavox/move_clutter_ortho"
DEFAULT_DATASET_ID = "naavox/ortho-target-dataset"
LOCAL_DATASET_ROOT = "ortho_target_data"

STATE_COMPONENTS = ("gripper_pos_x", "gripper_pos_y", "gripper_pos_z", "finger_pressure")


def room_to_ortho_px(x_m, y_m, width, height, extent_m=ORTHO_EXTENT_M):
    """Room-frame metres -> ortho pixel coordinates.

    Mirrors the M matrix in floor_view.generate_orthographic_floor_maps: the room
    origin is the image centre, x grows rightward, y grows upward (so v is flipped),
    and the image covers extent_m per side whatever resolution it was stored at.
    """
    u = x_m * (width / extent_m) + width / 2.0
    v = -y_m * (height / extent_m) + height / 2.0
    return u, v


def frame_to_bgr(frame):
    """A LeRobot video frame (CHW float RGB, or HWC uint8) as an HWC uint8 BGR array."""
    arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def coverage_fraction(bgr):
    """Fraction of pixels any camera actually painted.

    generate_orthographic_floor_maps leaves uncovered floor at zero, so a frame
    recorded before the anchor cameras came up is almost entirely black and carries
    no scene for a model to key on.
    """
    return float(np.count_nonzero(bgr.any(axis=2))) / float(bgr.shape[0] * bgr.shape[1])


def scan_episode_states(root: Path) -> dict[int, list[dict]]:
    """Per-episode gripper position, pressure and timestamp, ordered by frame index.

    Read straight from the parquets rather than through LeRobotDataset: only four of
    the 44 state components are wanted, for every frame of every episode, and pulling
    them as columns skips decoding a video frame per row.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    index_of = {name: i for i, name in enumerate(names)}

    missing = [name for name in STATE_COMPONENTS if name not in index_of]
    if missing:
        raise ValueError(
            f"{root} has no observation.state component(s) {missing}; the contact "
            f"position cannot be recovered without them. Present: {names}"
        )
    pos_idx = [index_of[f"gripper_pos_{axis}"] for axis in "xyz"]
    pressure_idx = index_of["finger_pressure"]

    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files found under {root}/data")

    episodes: dict[int, list[dict]] = {}
    for path in data_files:
        table = pq.read_table(
            path, columns=["episode_index", "frame_index", "timestamp", "observation.state"]
        )
        for ep, frame_index, timestamp, state in zip(
            table.column("episode_index").to_pylist(),
            table.column("frame_index").to_pylist(),
            table.column("timestamp").to_pylist(),
            table.column("observation.state").to_pylist(),
        ):
            episodes.setdefault(ep, []).append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "gripper_pos": [state[i] for i in pos_idx],
                "pressure": state[pressure_idx],
            })

    for rows in episodes.values():
        rows.sort(key=lambda r: r["frame_index"])
    return episodes


def episode_starts(dataset) -> dict[int, dict]:
    """Each episode's global start index, length and task string."""
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError(f"{dataset.repo_id} has no episode metadata loaded")
    columns = set(episodes.column_names)
    required = {"episode_index", "dataset_from_index", "length"}
    if not required <= columns:
        raise ValueError(
            f"episode metadata is missing {sorted(required - columns)}; this tool needs "
            f"LeRobot v3 episode metadata. Present: {sorted(columns)}"
        )

    tasks = episodes["tasks"] if "tasks" in columns else [None] * len(episodes["episode_index"])
    out = {}
    for ep, start, length, task in zip(
        episodes["episode_index"], episodes["dataset_from_index"], episodes["length"], tasks
    ):
        if isinstance(task, list):
            task = task[0] if task else None
        out[int(ep)] = {"start": int(start), "length": int(length), "task": task}
    return out


def annotate(bgr, u, v):
    """Copy of the frame with the label drawn on it, for eyeballing the projection."""
    out = bgr.copy()
    x, y = int(round(u)), int(round(v))
    cv2.drawMarker(out, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(out, (x, y), 12, (0, 255, 0), 2)
    return out


def build_samples(dataset, pressure_threshold, frame_offset, min_coverage, limit, annotate_dir):
    """One sample per episode: its early ortho frame and the contact pixel.

    Returns (samples, images, skipped) where `images` maps file name -> BGR frame and
    `skipped` counts why episodes were dropped.
    """
    root = Path(dataset.root)
    states = scan_episode_states(root)
    starts = episode_starts(dataset)

    if annotate_dir:
        Path(annotate_dir).mkdir(parents=True, exist_ok=True)

    samples, images = [], {}
    skipped = {"no_contact": 0, "too_short": 0, "blank_frame": 0, "off_map": 0}

    for ep in sorted(states):
        if limit and len(samples) >= limit:
            break
        rows = states[ep]
        meta = starts.get(ep)
        if meta is None:
            raise ValueError(f"episode {ep} has frames but no episode metadata")

        contact_index, _ = contact_blend_alphas(
            [r["timestamp"] for r in rows], [r["pressure"] for r in rows],
            pressure_threshold, blend_seconds=0.0,
        )
        if contact_index is None:
            skipped["no_contact"] += 1
            continue
        if frame_offset >= len(rows):
            skipped["too_short"] += 1
            continue

        contact = rows[contact_index]
        bgr = frame_to_bgr(dataset[meta["start"] + frame_offset][ORTHO_KEY])
        height, width = bgr.shape[:2]

        if coverage_fraction(bgr) < min_coverage:
            skipped["blank_frame"] += 1
            continue

        x_m, y_m, z_m = contact["gripper_pos"]
        u, v = room_to_ortho_px(x_m, y_m, width, height)
        if not (0 <= u < width and 0 <= v < height):
            skipped["off_map"] += 1
            continue

        file_name = f"ep{ep:06d}.jpg"
        images[file_name] = bgr
        samples.append({
            "file_name": file_name,
            # "points" (a list, of one) rather than a bare pair, so this dataset has
            # the same row shape as the hand-labelled target_heatmap one
            "points": [[round(u, 2), round(v, 2)]],
            "contact_m": [round(c, 4) for c in (x_m, y_m, z_m)],
            "episode_index": ep,
            "contact_frame_index": contact["frame_index"],
            "contact_time_s": round(contact["timestamp"], 3),
            "task": meta["task"],
        })

        if annotate_dir:
            cv2.imwrite(os.path.join(annotate_dir, file_name), annotate(bgr, u, v))

    return samples, images, skipped


def write_dataset(output_root: Path, samples, images, eval_fraction, seed):
    """Write the image folder: train/eval jpgs, metadata.jsonl each, and a README."""
    if output_root.exists():
        shutil.rmtree(output_root)

    ordered = sorted(samples, key=lambda s: s["file_name"])
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    cut = len(shuffled) - int(len(shuffled) * eval_fraction)
    split_of = {s["file_name"]: ("train" if i < cut else "eval") for i, s in enumerate(shuffled)}

    counts = {"train": 0, "eval": 0}
    for split in ("train", "eval"):
        split_dir = output_root / split
        split_dir.mkdir(parents=True)
        with open(split_dir / "metadata.jsonl", "w") as f:
            for sample in ordered:
                if split_of[sample["file_name"]] != split:
                    continue
                cv2.imwrite(str(split_dir / sample["file_name"]), images[sample["file_name"]])
                f.write(json.dumps(sample) + "\n")
                counts[split] += 1

    (output_root / "README.md").write_text(
        "---\nconfigs:\n- config_name: default\n  data_files:\n"
        "  - split: train\n    path: train/metadata.jsonl\n"
        "  - split: test\n    path: eval/metadata.jsonl\n---\n"
    )
    return counts


def upload_dataset(output_root: Path, dataset_id: str):
    """Replace the hub copy with this one.

    Unlike the hand-labelled target_heatmap dataset, every sample here is derived
    from the teleop datasets by a deterministic rule, so a full replacement loses
    nothing that cannot be regenerated - and pruning is what keeps samples from
    episodes since dropped by the recipe out of the dataset.
    """
    from huggingface_hub import HfApi, create_repo

    create_repo(dataset_id, repo_type="dataset", exist_ok=True)
    HfApi().upload_folder(
        folder_path=str(output_root),
        repo_id=dataset_id,
        repo_type="dataset",
        delete_patterns=["*.jpg"],
    )
    logging.info(f"Uploaded to {dataset_id}")


def distill(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root) if args.root else None
    dataset = LeRobotDataset(repo_id=args.repo_id, root=root)
    if ORTHO_KEY not in dataset.meta.video_keys:
        raise ValueError(
            f"'{args.repo_id}' has no '{ORTHO_KEY}' feature, so it carries no ortho view. "
            f"Build it with recipes/ortho_target.yaml. Present: {list(dataset.meta.video_keys)}"
        )

    logging.info(
        f"Distilling {dataset.meta.total_episodes} episode(s) of '{args.repo_id}' "
        f"({dataset.meta.total_frames} frames)"
    )
    samples, images, skipped = build_samples(
        dataset,
        pressure_threshold=args.pressure_threshold,
        frame_offset=args.frame_offset,
        min_coverage=args.min_coverage,
        limit=args.limit,
        annotate_dir=args.annotate_dir,
    )
    if not samples:
        raise ValueError(f"No usable episodes found. Skipped: {skipped}")

    output_root = Path(args.output)
    counts = write_dataset(output_root, samples, images, args.eval_fraction, args.seed)

    logging.info(f"Wrote {counts['train']} train + {counts['eval']} eval sample(s) to {output_root}")
    dropped = ", ".join(f"{reason}={n}" for reason, n in skipped.items() if n)
    logging.info(f"Dropped episodes: {dropped or 'none'}")
    if args.annotate_dir:
        logging.info(f"Annotated previews in {args.annotate_dir}")

    if args.upload:
        upload_dataset(output_root, args.dataset_id)
    else:
        logging.info(f"Not uploading; pass --upload to push to {args.dataset_id}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    distill_parser = subparsers.add_parser(
        "distill", help="reduce an ortho LeRobot dataset to one image + contact point per episode"
    )
    distill_parser.add_argument("--repo_id", default=DEFAULT_SOURCE_REPO_ID,
                                help="LeRobot dataset built by recipes/ortho_target.yaml")
    distill_parser.add_argument("--root", default=None,
                                help="local root of that dataset (default: the HF cache, downloading if needed)")
    distill_parser.add_argument("--output", default=LOCAL_DATASET_ROOT,
                                help="directory to write the distilled dataset to (replaced if it exists)")
    distill_parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID,
                                help="hub repo to upload the distilled dataset to")
    distill_parser.add_argument("--upload", action="store_true",
                                help="upload the result, replacing the hub copy")
    distill_parser.add_argument("--pressure_threshold", type=float, default=0.1,
                                help="finger_pressure above which an episode counts as having made contact")
    distill_parser.add_argument("--frame_offset", type=int, default=0,
                                help="which frame of each episode to keep, counted from its start")
    distill_parser.add_argument("--min_coverage", type=float, default=0.02,
                                help="skip frames where less than this fraction of the ortho map was "
                                     "painted by any camera")
    distill_parser.add_argument("--eval_fraction", type=float, default=0.1,
                                help="fraction of samples held out as the eval split")
    distill_parser.add_argument("--seed", type=int, default=0, help="seed for the train/eval split")
    distill_parser.add_argument("--limit", type=int, default=0,
                                help="stop after this many samples, for a quick trial run")
    distill_parser.add_argument("--annotate_dir", default=None,
                                help="also write copies with the label drawn on them, to check the projection")

    args = parser.parse_args()
    if args.command == "distill":
        distill(args)


if __name__ == "__main__":
    main()
