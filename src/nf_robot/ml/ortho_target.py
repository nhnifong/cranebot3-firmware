#!/usr/bin/env python

"""Predict everywhere the robot could reach next, in the ortho floor view's pixel space.

The map is one independent objectness per cell, so a frame holding four graspable things
can report four of them and the scores mean the same in every frame; see OrthoTargetNet
and objectness_loss, which carries the consequence of labelling only one of those four.

The labels come from teleop recordings: wherever an operator actually grasped something
is by construction a place worth reaching for, so every episode donates one label for
free, with no hand labelling.

Build, distill, train, then ship. The first two are separate because the intermediate
is a LeRobot dataset and the result is not:

  1. The recipe merges the teleop datasets, keeping the ortho feed this model needs
     and the gripper feed visual_servoing/mine_teleop.py needs, and runs contact
     labelling. One dataset serves both models because the expensive part - sourcing,
     excluding episodes and re-encoding video - is identical for each:

       python src/nf_robot/ml/lerobot_build_dataset.py \
           --recipe src/nf_robot/ml/recipes/combined_targets.yaml \
           --temp_dir /home/nhn/data_scratch \
           --output_root /home/nhn/data_scratch/combined_targets

     Then hold an eval set out of it. The two halves come out named after their splits
     - naavox/combined_targets_train and naavox/combined_targets_eval - in the LeRobot
     home, so everything below finds them by repo id alone:

       python src/nf_robot/ml/lerobot_split_dataset.py \
           --repo_id naavox/combined_targets \
           --root /home/nhn/data_scratch/combined_targets

     --eval_fraction defaults to 0.1 and --seed to 0, so the same command twice is the
     same split - which is what lets the two halves be distilled separately without
     overlapping. See the note on the eval split below for what it measures.

  2. `distill` reduces that to one sample per episode - the episode's first ortho
     frame and the ortho pixel where contact eventually happened - which is a few
     hundred MB rather than a few hundred GB. One run per split, and --upload needs
     both splits present locally because it prunes hub files that are absent - so
     distill the eval split first, or upload only on the second run:

       python -m nf_robot.ml.ortho_target distill \
           --repo_id naavox/combined_targets_eval --split eval

       python -m nf_robot.ml.ortho_target distill \
           --repo_id naavox/combined_targets_train --split train --upload

  3. `train` fits the model, saving the best checkpoint by top5@20cm to
     models/ortho_target.pth:

       python -m nf_robot.ml.ortho_target train

     The dataset resizes to whatever --image_size asks for, and the checkpoint records
     the backbone id and image size, so `evaluate` needs no flags at all. See the DINOv3
     footnote at the bottom for the backbone this used to default to.

  4. `evaluate` scores that checkpoint - or the published one, downloaded, if training
     has not run on this machine - against the held-out room, and --preview_dir
     draws what it actually predicted: the label in green, the ranked candidates in
     red. Numbers say whether it is right, the previews say whether it is right for
     the right reason, which is the check worth doing before a model reaches a robot:

       python -m nf_robot.ml.ortho_target evaluate --tta --preview_dir previews

  5. Try it on a robot before publishing. --local_models makes the observer load
     models/ortho_target.pth - where training just wrote it - instead of the hub
     copy. This is the only target model; the UI's targeting switch loads it:

       stringman-headless --local_models

  6. Publish for stringman users. Until this is done, anyone without --local_models
     is still on the previously published checkpoint:

       hf upload naavox/targeting models/ortho_target.pth ortho_target.pth

Labels can also come from the UI instead of a recording. These are the only frames where
every target is marked, so the rows of one submission are loaded back as a single sample
carrying all of them rather than as one sample each - which is the shape the head wants
and the teleop labels cannot supply. The RUN menu's "Add targets to dataset" saves the
ortho frame the robot is looking at and every target placed on it by hand, one row per
target in exactly the format step 2 writes, into
ortho_target_user_labels/ - relative to the directory stringman was started from. Nothing
trains on them until they are merged, which is the point of keeping them apart: nobody
reached for these, so nothing confirms the object was really there or could be picked up.

  Back them up, or gather several robots' worth in one place, on your own account. The
  repo is created private, since these are pictures of your floor:

      python -m nf_robot.ml.ortho_target upload_labels

  Fold them into the distilled dataset - from that directory, or from any hub repo full
  of them - and train on the result by pointing --data_root at it rather than letting it
  download the published copy:

      python -m nf_robot.ml.ortho_target merge_labels
      python -m nf_robot.ml.ortho_target merge_labels --repo_id you/ortho-target-user-labels

      python -m nf_robot.ml.ortho_target train --data_root ortho_target_data

  Merged files keep the names they arrived under, so `rm ortho_target_data/train/user-*.parquet`
  undoes it and merging twice overwrites rather than duplicates. Step 2 rewrites a split
  directory from scratch, so a re-distill drops them and the merge has to run again after
  it. They join the train split unless --split eval says otherwise; see merge_user_labels
  for why that is a different measurement rather than a bigger one.

Train with the backbone frozen, which is what step 3 does by default. --unfreeze_backbone
exists but is not the supported path, for three reasons that all point the same way: a
few thousand samples is far too few to move an 86M-parameter trunk without memorising
the floors it saw; frozen is what lets the observer serve this model and the visual
servoing one from a single shared trunk (ml/dino_trunk.py) rather than two copies of
the same 327MB - they default to the same backbone id, which is what makes that sharing
happen; and a frozen trunk is recoverable from backbone_id and a download, so
it stays out of the checkpoint and ortho_target.pth carries heads alone. A checkpoint
trained unfrozen records "freeze": False, loads a private trunk and shares nothing -
the model still runs, it just costs what it used to.

The ortho view is an orthographic projection of the floor plane (host/floor_view.py),
so room metres map to its pixels analytically - no camera pose is involved, unlike
camera_goal.py's per-anchor projection.

The eval split is a random sample of episodes, which is what lerobot_split_dataset.py
makes it. Worth knowing what that measures: consecutive episodes in one recording
session share a floor and usually most of an object layout - the operator clearing one
pile item by item - so a random cut lands near duplicates on both sides, and the score
says how well the model does on more of the rooms it has seen rather than on a room it
has not. This model used to get the harder measurement from a second recipe holding the
79west room whole; that recipe is gone and its sources are in combined_targets.yaml with
everything else.

Note that lerobot-edit-dataset's own fractional split is not a random sample - it takes
a contiguous range, the last tenth in order, which after a merge is whichever sources
sit at the bottom of the recipe. lerobot_split_dataset.py shuffles the indices and
passes them explicitly, which is the whole reason it exists.

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

Footnote: DINOv3.

The backbone was facebook/dinov3-vitb16-pretrain-lvd1689m at 512 until 2026-08-23. It is
gated - an approved access request plus an HF_TOKEN on every machine that trains or runs
the model - and DINOv2 with registers scores the same on the visual servoing task, so
there is no reason to prefer it. To train against it anyway:

    python -m nf_robot.ml.ortho_target train \
        --backbone facebook/dinov3-vitb16-pretrain-lvd1689m --image_size 512

The image size moves with the backbone because the trunk's patch size has to divide it,
and the grid has to stay a power-of-two multiple of the token grid: 512/16 and 448/14
both give 32x32 tokens and a 128 grid, so nothing downstream of the trunk changes.

A checkpoint carries its own backbone_id and image_size, so anything already trained
keeps loading and running as it was.
"""

import argparse
import json
import logging
import math
import os
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from nf_robot.ml.dino_trunk import SharedTrunkMixin, drop_trunk_weights

ORTHO_FEED = 3


def ortho_key():
    """The dataset feature the ortho view lives in.

    A function rather than a module constant so that the lerobot import it needs happens
    in the distill path and nowhere else. Everything downstream of this module that only
    wants the model, or the normalization constants below, then costs nothing to import:
    the observer reaches ortho_target through visual_servoing/dataset.py on the way to
    three numbers, and pulling the whole training stack in behind that made the visual
    servoing grasp unusable on a host-only install.
    """
    from nf_robot.ml.stringman_lerobot import _FEED_NAMES

    return f"observation.images.{_FEED_NAMES[ORTHO_FEED]}"
# Both feeds this dataset carries: the ortho composite for this model, and the
# gripper camera at the visual servoing model's input size. See combined_targets.yaml.
ORTHO_CAMERA_MODE = "gripper_ortho"

# Metres of floor the square ortho map spans, per side, centred on the room origin.
# host/floor_view.py's EXTENT_M and the map_extent_meters observer._ortho_worker
# renders with; a recording made with a different extent would need its own value.
ORTHO_EXTENT_M = 5.0

DEFAULT_SOURCE_REPO_ID = "naavox/combined_targets"
DEFAULT_DATASET_ID = "naavox/ortho-target-dataset"
LOCAL_DATASET_ROOT = "ortho_target_data"

# Labels the UI's "Add targets to dataset" writes, and the hub repo upload_labels puts
# them in - under the uploader's own account, not naavox's, since they are one operator's
# floor labelled by that operator. See write_user_labels.
USER_LABEL_ROOT = "ortho_target_user_labels"
USER_LABEL_DATASET_NAME = "ortho-target-user-labels"

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


def ortho_px_to_room(u, v, width, height, extent_m=ORTHO_EXTENT_M):
    """Ortho pixel coordinates -> room-frame metres; the inverse of room_to_ortho_px."""
    x_m = (u - width / 2.0) * extent_m / width
    y_m = -(v - height / 2.0) * extent_m / height
    return x_m, y_m


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


def episode_frame_offsets(contact_index, frame_offset, count, stride):
    """Which frames of an episode to distil, as offsets from its start.

    Every frame before the grasp shows the same floor with the target in the same place,
    so one episode is worth `count` samples rather than one - the ortho view is a room
    frame projection, so the label does not move as the gantry does. What changes is
    where the robot is, what it occludes and how the light falls, which is exactly the
    variation the model has to be robust to and none of it is free from augmentation.

    Frames are taken from frame_offset forward, stopping before contact: after the grasp
    the object is in the jaws and the floor no longer holds it.
    """
    last = contact_index - 1
    offsets = [frame_offset + i * stride for i in range(count)]
    return [o for o in offsets if o <= last]


def build_samples(dataset, pressure_threshold, frame_offset, min_coverage, limit,
                  annotate_dir, frames_per_episode=1, frame_stride=1):
    """Samples from each episode: ortho frames before the grasp, and the contact pixel.

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

        from nf_robot.ml.lerobot_label_contact_actions import contact_blend_alphas

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
        offsets = episode_frame_offsets(contact_index, frame_offset,
                                        frames_per_episode, frame_stride)
        if not offsets:
            skipped["too_short"] += 1
            continue

        x_m, y_m, z_m = contact["gripper_pos"]
        for n, offset in enumerate(offsets):
            bgr = frame_to_bgr(dataset[meta["start"] + offset][ortho_key()])
            height, width = bgr.shape[:2]

            if coverage_fraction(bgr) < min_coverage:
                skipped["blank_frame"] += 1
                continue

            u, v = room_to_ortho_px(x_m, y_m, width, height)
            if not (0 <= u < width and 0 <= v < height):
                skipped["off_map"] += 1
                # the label is a property of the episode, not of the frame, so if it is
                # off the map for one frame it is off it for all of them
                break

            # One file per frame, still sorted by episode. Frames of an episode are near
            # duplicates, so they have to travel together into one split - which they do,
            # because the split is chosen upstream by episode.
            file_name = f"ep{ep:06d}.jpg" if frames_per_episode == 1 else f"ep{ep:06d}_{n:02d}.jpg"
            images[file_name] = bgr
            samples.append({
                "file_name": file_name,
                # "points" (a list, of one) rather than a bare pair, leaving room for
                # rows that carry more than a single label
                "points": [[round(u, 2), round(v, 2)]],
                "contact_m": [round(c, 4) for c in (x_m, y_m, z_m)],
                "episode_index": ep,
                "frame_offset": offset,
                "contact_frame_index": contact["frame_index"],
                "contact_time_s": round(contact["timestamp"], 3),
                "task": meta["task"],
            })

            if annotate_dir:
                cv2.imwrite(os.path.join(annotate_dir, file_name), annotate(bgr, u, v))

    return samples, images, skipped


SHARD_TARGET_BYTES = 256 * 1024 * 1024
# Columns beside the JPEG bytes. Read without the image column, they are the whole label
# table, which is what lets the baseline and the split statistics be computed without
# decoding a single frame.
LABEL_COLUMNS = ("file_name", "u", "v", "contact_m", "episode_index", "frame_offset",
                 "contact_frame_index", "contact_time_s", "task")


def shard_schema():
    import pyarrow as pa

    return pa.schema([
        pa.field("file_name", pa.string()),
        pa.field("image", pa.binary()),
        pa.field("u", pa.float64()),
        pa.field("v", pa.float64()),
        pa.field("contact_m", pa.list_(pa.float64())),
        pa.field("episode_index", pa.int32()),
        pa.field("frame_offset", pa.int32()),
        pa.field("contact_frame_index", pa.int32()),
        pa.field("contact_time_s", pa.float64()),
        pa.field("task", pa.string()),
    ])


def write_shards(split_dir: Path, samples, images, target_bytes=SHARD_TARGET_BYTES) -> int:
    """Write the split as parquet shards of JPEG bytes plus their labels.

    One file per sample is what this used to be, and it does not survive the hub: at
    eight frames an episode the dataset is 6,500 loose jpegs, and the download asks for a
    read token per file until it is rate limited. Shards make that a handful of files,
    which is the same reason visual_servoing's miner writes them.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = shard_schema()
    written, shard, rows, pending = 0, 0, [], 0

    def flush():
        nonlocal shard, rows, pending
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, split_dir / f"shard-{shard:04d}.parquet")
        shard, rows, pending = shard + 1, [], 0

    for sample in sorted(samples, key=lambda s: s["file_name"]):
        ok, buf = cv2.imencode(".jpg", images[sample["file_name"]])
        if not ok:
            raise ValueError(f"could not encode {sample['file_name']}")
        blob = buf.tobytes()
        (u, v), = sample["points"]
        rows.append({
            "file_name": sample["file_name"],
            "image": blob,
            "u": u,
            "v": v,
            "contact_m": sample["contact_m"],
            "episode_index": int(sample["episode_index"]),
            "frame_offset": int(sample.get("frame_offset", 0)),
            "contact_frame_index": int(sample["contact_frame_index"]),
            "contact_time_s": float(sample["contact_time_s"]),
            "task": sample.get("task") or "",
        })
        pending += len(blob)
        written += 1
        if pending >= target_bytes:
            flush()
    flush()
    return written


def write_split(output_root: Path, split: str, samples, images) -> int:
    """Replace one split of the image folder, leaving the other one alone.

    Which episodes are in a split is decided before this, by lerobot_split_dataset: a
    distill run takes every episode of the dataset it is pointed at, and --split only
    names the directory it writes them to. So the two runs have to be pointed at
    <repo_id>_train and <repo_id>_eval respectively - nothing here can check that they
    were, and the halves only stay disjoint while both come from one split run.
    """
    split_dir = output_root / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    written = write_shards(split_dir, samples, images)

    write_dataset_readme(output_root)
    return written


def write_dataset_readme(output_root: Path):
    """The hub's dataset config block, naming whichever splits are actually present."""
    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for name, dirname in (("train", "train"), ("test", "eval")):
        if any((output_root / dirname).glob("*.parquet")):
            lines += [f"  - split: {name}", f"    path: {dirname}/*.parquet"]
    (output_root / "README.md").write_text("\n".join(lines) + "\n---\n")


def write_user_labels(rgb, targets_m, output_root=USER_LABEL_ROOT, extent_m=ORTHO_EXTENT_M):
    """One row per operator-placed target, all sharing the ortho frame they were placed on.

    The same schema the distilled shards use, but in its own directory and one parquet per
    submission, because these labels are not quite the same thing as the teleop ones:
    nobody reached for them, so nothing confirms the object was really there or that it
    could be grasped. Whether a pile of them is worth training on is a decision to make
    later, and acting on it is a matter of moving these files into a split directory.

    The frame is written at whatever size the observer renders it (host/observer.py's
    _ortho_worker), which need not be the size the distilled shards hold, and the rows of
    one submission are all the same frame - so a split cannot cut through them the way it
    cannot cut through an episode. merge_user_labels handles both; a hand-rolled merge has
    to handle them itself.

    Returns (path, row count), or (None, 0) if every target fell outside the map.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("could not encode the ortho frame")
    blob = buf.tobytes()

    batch = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    rows = []
    for x_m, y_m, z_m in targets_m:
        u, v = room_to_ortho_px(x_m, y_m, width, height, extent_m)
        if not (0 <= u < width and 0 <= v < height):
            continue
        rows.append({
            "file_name": f"user-{batch}-{len(rows):02d}.jpg",
            "image": blob,
            "u": round(float(u), 2),
            "v": round(float(v), 2),
            "contact_m": [round(float(c), 4) for c in (x_m, y_m, z_m)],
            # Nobody reached for these, so every field that indexes into an episode is -1
            # - which is also what tells them apart from distilled rows after a merge.
            "episode_index": -1,
            "frame_offset": 0,
            "contact_frame_index": -1,
            "contact_time_s": 0.0,
            "task": "user target",
        })
    if not rows:
        return None, 0

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"user-{batch}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=shard_schema()), path)
    return path, len(rows)


def upload_dataset(output_root: Path, dataset_id: str):
    """Replace the hub copy with this one.

    Every sample here is derived from the teleop datasets by a deterministic rule, so
    a full replacement loses nothing that cannot be regenerated - and pruning is what
    keeps samples of episodes since dropped by a recipe out of the dataset.
    """
    from huggingface_hub import HfApi, create_repo

    missing = [s for s in ("train", "eval") if not any((output_root / s).glob("*.parquet"))]
    if missing:
        raise ValueError(
            f"{output_root} has no {missing} split yet. The upload prunes hub files that are "
            f"absent locally, so uploading now would delete the other split's shards. Distill "
            f"both splits into this directory first."
        )

    create_repo(dataset_id, repo_type="dataset", exist_ok=True)
    HfApi().upload_folder(
        folder_path=str(output_root),
        repo_id=dataset_id,
        repo_type="dataset",
        # *.jpg clears the loose frames of the pre-shard layout, which are otherwise left
        # behind on the hub for as long as the repo lives.
        delete_patterns=["*.jpg", "*.parquet", "*/metadata.jsonl"],
    )
    logging.info(f"Uploaded to {dataset_id}")


USER_LABEL_README = """---
configs:
- config_name: default
  data_files:
  - split: train
    path: "*.parquet"
---

Targets placed by hand in the stringman UI, on the orthographic floor view the robot was
looking at when they were placed. One row per target, one file per submission, in the same
schema as naavox/ortho-target-dataset - so `ortho_target.py merge_labels --repo_id <this>`
folds them into a distilled dataset. See nf_robot/ml/ortho_target.py.
"""


def user_label_dataset_id(dataset_id=None):
    """Where this account's user labels live on the hub.

    The distilled dataset has one canonical copy under naavox/ because it is a
    deterministic function of teleop recordings anyone can rebuild. These are not that:
    they are pictures of one person's floor, labelled by that person's own judgement of
    what is worth picking up. So the default repo belongs to whoever `hf auth login`
    left logged in, and upload_user_labels creates it private.
    """
    if dataset_id:
        return dataset_id
    from huggingface_hub import HfApi

    try:
        account = HfApi().whoami()["name"]
    except Exception as e:
        raise ValueError(
            "Not logged in to Hugging Face, so there is no account to upload to. Run "
            "`hf auth login`, or name the repo yourself with --dataset_id."
        ) from e
    return f"{account}/{USER_LABEL_DATASET_NAME}"


def upload_user_labels(source_root, dataset_id=None, private=True):
    """Add locally saved user labels to a hub dataset, disturbing nothing already there.

    The opposite policy to upload_dataset, which prunes. A distilled dataset can be
    rebuilt from the recordings at will, so replacing it wholesale loses nothing; a user
    label is one click on a frame of a room that has since been tidied, so the copy on
    the hub is the only copy and pruning would be deleting it. Files are named for the
    submission that produced them, so several robots can fill one repo without colliding
    and re-uploading the same directory is a no-op.

    Returns (dataset_id, files, rows).
    """
    from huggingface_hub import HfApi, create_repo

    source_root = Path(source_root)
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No label files in {source_root}. The UI's 'Add targets to dataset' action writes "
            f"them there, relative to the directory stringman was started from."
        )
    rows = sum(pq.read_metadata(f).num_rows for f in files)

    dataset_id = user_label_dataset_id(dataset_id)
    create_repo(dataset_id, repo_type="dataset", exist_ok=True, private=private)
    api = HfApi()
    api.upload_folder(
        folder_path=str(source_root),
        repo_id=dataset_id,
        repo_type="dataset",
        allow_patterns=["*.parquet"],
    )
    api.upload_file(
        path_or_fileobj=USER_LABEL_README.encode(),
        path_in_repo="README.md",
        repo_id=dataset_id,
        repo_type="dataset",
    )
    logging.info(f"Uploaded {len(files)} submission(s), {rows} label(s), to {dataset_id}")
    return dataset_id, len(files), rows


def first_stored_size(path: Path):
    """(width, height) of the first frame in a shard, without decoding the rest of it."""
    batch = next(pq.ParquetFile(path).iter_batches(batch_size=1, columns=["image"]))
    bgr = cv2.imdecode(np.frombuffer(batch.column("image")[0].as_py(), np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"undecodable first frame in {path}")
    return bgr.shape[1], bgr.shape[0]


def split_stored_size(split_dir: Path):
    """The size a split's distilled frames are stored at, or None if it holds none yet.

    Merged labels are skipped, so the answer is the size the split is trying to be rather
    than whatever the last merge happened to put in it.
    """
    for shard in sorted(split_dir.glob("*.parquet")):
        if not shard.name.startswith("user-"):
            return first_stored_size(shard)
    return None


def resize_rows(rows, size):
    """Rows re-encoded at (width, height), with their labels scaled to match.

    Rows of one submission share a frame - the same bytes object, read from one column -
    so the cache decodes and re-encodes it once however many labels were placed on it.
    """
    width, height = size
    cache, out = {}, []
    for row in rows:
        blob = row["image"]
        if blob not in cache:
            bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"undecodable frame {row['file_name']}")
            h, w = bgr.shape[:2]
            if (w, h) == (width, height):
                cache[blob] = (blob, 1.0, 1.0)
            else:
                interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
                ok, buf = cv2.imencode(".jpg", cv2.resize(bgr, (width, height), interpolation=interp))
                if not ok:
                    raise ValueError(f"could not re-encode {row['file_name']}")
                cache[blob] = (buf.tobytes(), width / w, height / h)
        blob, su, sv = cache[blob]
        out.append({**row, "image": blob,
                    "u": round(row["u"] * su, 2), "v": round(row["v"] * sv, 2)})
    return out


def merge_user_labels(source_root, output_root=LOCAL_DATASET_ROOT, split="train", resize=True):
    """Copy user labels into a distilled dataset's split, so training sees them.

    They stay in their own files, named for the submission that made them, which is what
    makes this reversible: `rm ortho_target_data/train/user-*.parquet` unmerges it, and
    merging twice overwrites rather than duplicates. OrthoTargetDataset globs the split
    directory, so nothing downstream has to know they are there.

    Two things this handles rather than leaving to whoever runs it:
      - Frames are re-encoded at the size the split's distilled shards already use.
        Training scales every label by its own frame, so mixed sizes do train correctly,
        but scaled_labels - and so constant_baseline - probes one sample for the whole
        split and would be quietly wrong about the rest. resize=False keeps the pixels.
      - One submission's rows share one frame, so they must not land on both sides of a
        train/eval cut. A file is a submission and goes to one split, which keeps that
        true for free.

    Mind the order: `distill` rewrites a split directory from scratch (write_split), so
    re-distilling drops merged labels and this has to run again afterwards.

    Returns (files, rows).
    """
    import pyarrow as pa

    source_root = Path(source_root)
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No label files in {source_root}")

    output_root = Path(output_root)
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    target_size = split_stored_size(split_dir) if resize else None
    if resize and target_size is None:
        logging.info(f"{split_dir} holds no distilled shards to match, so frames are merged as they are")

    schema = shard_schema()
    merged = 0
    for path in files:
        # cast rather than trust: a file from the hub was written by some other machine's
        # version of this module, and a column that has drifted should stop the merge
        # rather than reach training as a silently wrong label.
        rows = pq.read_table(path).cast(schema).to_pylist()
        if target_size:
            rows = resize_rows(rows, target_size)
        # the prefix is what unmerges and split_stored_size both key on, so enforce it
        name = path.name if path.name.startswith("user-") else f"user-{path.name}"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), split_dir / name)
        merged += len(rows)

    write_dataset_readme(output_root)
    logging.info(f"Merged {len(files)} submission(s), {merged} label(s), into {split_dir}"
                 + (f" at {target_size[0]}x{target_size[1]}" if target_size else ""))
    return len(files), merged


def merge_labels(args):
    source = args.source
    if args.repo_id:
        from huggingface_hub import snapshot_download

        source = snapshot_download(repo_id=args.repo_id, repo_type="dataset")
        logging.info(f"Merging {args.repo_id} from the hub at {source}")
    merge_user_labels(source, args.output, args.split, resize=not args.no_resize)
    logging.info(f"Train on the result with: python -m nf_robot.ml.ortho_target train "
                 f"--data_root {args.output}")


def distill(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root) if args.root else None
    # A repo id and no --root means the hub's copy, so re-sync before reading: without
    # this, LeRobotDataset serves whatever is in the LeRobot home and a name that has
    # since been rebuilt on the hub silently distills the old dataset. That is not
    # hypothetical - it is how an eval split of one room outlived the recipe that made
    # it, and the numbers it produced looked fine. --root is the way to say "this copy,
    # whatever the hub has".
    dataset = LeRobotDataset(repo_id=args.repo_id, root=root, force_cache_sync=root is None)
    key = ortho_key()
    if key not in dataset.meta.video_keys:
        raise ValueError(
            f"'{args.repo_id}' has no '{key}' feature, so it carries no ortho view. "
            f"Build it with recipes/ortho_target.yaml. Present: {list(dataset.meta.video_keys)}"
        )

    # The resolved root is logged because a stale one is invisible otherwise: the repo id
    # in the command says nothing about which copy of it answered.
    logging.info(
        f"Distilling {dataset.meta.total_episodes} episode(s) of '{args.repo_id}' "
        f"({dataset.meta.total_frames} frames) from {dataset.root}"
    )
    samples, images, skipped = build_samples(
        dataset,
        pressure_threshold=args.pressure_threshold,
        frame_offset=args.frame_offset,
        min_coverage=args.min_coverage,
        limit=args.limit,
        annotate_dir=args.annotate_dir,
        frames_per_episode=args.frames_per_episode,
        frame_stride=args.frame_stride,
    )
    if not samples:
        raise ValueError(f"No usable episodes found. Skipped: {skipped}")

    output_root = Path(args.output)
    written = write_split(output_root, args.split, samples, images)

    logging.info(f"Wrote {written} sample(s) to {output_root / args.split}")
    dropped = ", ".join(f"{reason}={n}" for reason, n in skipped.items() if n)
    logging.info(f"Dropped episodes: {dropped or 'none'}")
    if args.annotate_dir:
        logging.info(f"Annotated previews in {args.annotate_dir}")

    if args.upload:
        upload_dataset(output_root, args.dataset_id)
    else:
        logging.info(f"Not uploading; pass --upload to push to {args.dataset_id}")


# ==========================================
# AUGMENTATION
# ==========================================
# The ortho view is a metric top-down map and gravity points out of the image, so the
# floor has no canonical orientation: any of the 8 ways of turning a square onto
# itself (4 rotations x optionally mirrored) produces an image that could equally have
# been recorded. Those 8 are exact - whole-pixel moves, no resampling and no
# interpolation blur - and the label moves with the image by the same rule, so this is
# an 8x dataset with no added label noise. Nothing else in the augmentation set is
# free like this; it is the main defence against 1400 samples of memorised floor.


def dihedral_image(img, t: int):
    """One of the 8 square symmetries applied to a CHW (or BCHW) tensor."""
    out = torch.rot90(img, t % 4, dims=(-2, -1))
    return torch.flip(out, dims=(-1,)) if t >= 4 else out


def dihedral_point(u, v, t: int, size: int):
    """The same symmetry applied to a point, in pixel-centre coordinates."""
    last = size - 1
    for _ in range(t % 4):
        u, v = v, last - u  # rot90 on dims (-2, -1) is counter-clockwise
    if t >= 4:
        u = last - u
    return u, v


def translate(img, u, v, max_px: int, rng):
    """Shift the map and its labels together by up to max_px in each axis.

    Valid because the ortho view is a metric projection of the floor: a shifted map is
    the same floor with the window moved, which is a view the robot really can have. It
    is also the antidote to the strongest shortcut in this dataset - the labels cluster
    near the middle of the map, so a model can score respectably by ignoring the image
    and predicting the mean. Shifting breaks the tie between "where the target is" and
    "where targets usually are".

    Vacated edges are filled with black, which is what unobserved floor already looks
    like in these maps, so it introduces nothing the model has not seen.
    """
    # Bounded by the outermost label on each axis, so none of them can leave the map:
    # objectness_loss clamps an out-of-range target to an edge cell, which would teach
    # the edge as the answer.
    height, width = img.shape[-2:]
    lo_x, hi_x = int(math.ceil(-u.min())), int(math.floor(width - 1 - u.max()))
    lo_y, hi_y = int(math.ceil(-v.min())), int(math.floor(height - 1 - v.max()))
    dx = int(torch.randint(max(-max_px, lo_x), min(max_px, hi_x) + 1, (1,), generator=rng).item())
    dy = int(torch.randint(max(-max_px, lo_y), min(max_px, hi_y) + 1, (1,), generator=rng).item())
    if dx == 0 and dy == 0:
        return img, u, v
    out = torch.zeros_like(img)
    src_x0, dst_x0 = max(0, -dx), max(0, dx)
    src_y0, dst_y0 = max(0, -dy), max(0, dy)
    w = width - abs(dx)
    h = height - abs(dy)
    out[..., dst_y0:dst_y0 + h, dst_x0:dst_x0 + w] = img[..., src_y0:src_y0 + h, src_x0:src_x0 + w]
    return out, u + dx, v + dy


def inverse_dihedral_map(m, t: int):
    """Undo dihedral_image on a spatial map, so predictions can be averaged."""
    if t >= 4:
        m = torch.flip(m, dims=(-1,))
    return torch.rot90(m, -(t % 4), dims=(-2, -1))


def photometric_jitter(img, rng: torch.Generator):
    """Brightness/contrast/saturation jitter, plus an occasional drop to grayscale.

    Floor colour is the shortcut this model must not take, so the augmentation that
    matters most is the one that makes colour unreliable.
    """
    def uniform(lo, hi):
        return torch.empty((), device=img.device).uniform_(lo, hi, generator=rng)

    img = img * uniform(0.7, 1.3)                                  # brightness
    mean = img.mean(dim=(-2, -1), keepdim=True)
    img = (img - mean) * uniform(0.7, 1.3) + mean                  # contrast
    gray = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device).view(3, 1, 1)).sum(0, keepdim=True)
    if torch.rand((), device=img.device, generator=rng) < 0.15:
        img = gray.expand_as(img).clone()
    else:
        img = gray + (img - gray) * uniform(0.6, 1.4)              # saturation
    return img.clamp(0.0, 1.0)


# ==========================================
# DATASET
# ==========================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Labels kept per frame. Teleop frames carry one; a hand-labelled frame carries as many
# as the operator placed, and the padding this fixes lets the default collate stack them.
MAX_TARGETS = 16


def frame_key(row):
    """What makes two rows the same frame, for collecting a frame's targets together.

    Not the image bytes, tempting as that is: a still scene re-encodes to byte-identical
    JPEGs across consecutive frame offsets of one episode, and those are two frames that
    look alike rather than one frame labelled twice. Keying on them merged an eighth of
    the distilled rows and shrank the training set.

    A distilled row's file name already names its episode and offset, so it is unique per
    frame. write_user_labels marks its rows with episode_index -1 and names them
    user-{batch}-{NN}.jpg, all targets of one submission sharing the batch, so dropping
    that trailing index is what collects them.
    """
    return row["file_name"] if row["episode_index"] >= 0 else row["file_name"].rsplit("-", 1)[0]


class OrthoTargetDataset(torch.utils.data.Dataset):
    """Distilled ortho frames and their targets, as (image, points, mask) triples.

    `points` is MAX_TARGETS rows of pixel coordinates and `mask` says how many of them
    are real, so a frame labelled with four objects and one labelled with one both come
    out of the loader as the same shape.
    """

    def __init__(self, root: Path, split: str, image_size: int, augment: bool, seed: int = 0,
                 translate_px: int = 0):
        self.dir = Path(root) / split
        shards = sorted(self.dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No parquet shards at {self.dir}. A distilled dataset from before this was "
                f"sharded holds loose jpegs and a metadata.jsonl; re-run distill to convert it.")

        # Frames are held as their JPEG bytes rather than decoded: a split is a few
        # hundred MB compressed and several GB decoded, and every one of them is about to
        # be resized and augmented anyway.
        self.images: list[bytes] = []
        self.samples: list[dict] = []
        # Rows are one per target, so the rows of one hand-labelled frame have to be
        # collected into a single multi-target sample rather than N contradictory
        # single-target ones. See frame_key for why that is not keyed on the image.
        by_frame: dict[str, dict] = {}
        for shard in shards:
            table = pq.read_table(shard)
            blobs = table.column("image").to_pylist()
            labels = table.select(list(LABEL_COLUMNS)).to_pylist()
            for blob, row in zip(blobs, labels):
                key = frame_key(row)
                sample = by_frame.get(key)
                if sample is None:
                    sample = by_frame[key] = {**row, "points": []}
                    self.images.append(blob)
                    self.samples.append(sample)
                sample["points"].append([row["u"], row["v"]])

        self.image_size = image_size
        self.augment = augment
        self.seed = seed
        self.translate_px = translate_px
        total = sum(len(b) for b in self.images)
        targets = sum(len(s["points"]) for s in self.samples)
        logging.info(f"{self.dir}: {len(self.samples)} samples ({targets} targets) in "
                     f"{len(shards)} shard(s), {total / 1e6:.0f} MB of frames in memory")

    def __len__(self):
        return len(self.samples)

    def decode(self, idx):
        """One stored frame as BGR, at whatever size it was written."""
        bgr = cv2.imdecode(np.frombuffer(self.images[idx], np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"undecodable frame {self.samples[idx]['file_name']} in {self.dir}")
        return bgr

    def stored_size(self):
        probe = self.decode(0)
        return probe.shape[1], probe.shape[0]

    def scaled_labels(self):
        """Each sample's first label in the model's pixel space, without decoding images.

        First rather than all, because the baseline it feeds predicts one point and is
        only meaningful against one.
        """
        width, height = self.stored_size()
        scale = np.array([self.image_size / width, self.image_size / height])
        return np.array([s["points"][0] for s in self.samples], dtype=np.float64) * scale

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bgr = self.decode(idx)

        points = np.asarray(sample["points"], dtype=np.float64)
        h, w = bgr.shape[:2]
        if (w, h) != (self.image_size, self.image_size):
            points = points * [self.image_size / w, self.image_size / h]
            bgr = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        if self.augment:
            # Seeded per (epoch-agnostic) index draw so workers don't share a stream.
            rng = torch.Generator().manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
            t = int(torch.randint(0, 8, (1,), generator=rng).item())
            img = dihedral_image(img, t)
            u, v = dihedral_point(points[:, 0], points[:, 1], t, self.image_size)
            if self.translate_px:
                img, u, v = translate(img, u, v, self.translate_px, rng)
            points = np.stack([u, v], axis=1)
            img = photometric_jitter(img, rng)

        img = (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)

        kept = min(len(points), MAX_TARGETS)
        padded = np.zeros((MAX_TARGETS, 2), dtype=np.float32)
        padded[:kept] = points[:kept]
        mask = np.zeros(MAX_TARGETS, dtype=np.float32)
        mask[:kept] = 1.0
        return img, torch.from_numpy(padded), torch.from_numpy(mask)


# ==========================================
# MODEL
# ==========================================

DEFAULT_BACKBONE = "facebook/dinov2-with-registers-base"
# 448 = 14 x 32, so the /14 trunk gives a 32x32 token grid and DEFAULT_GRID stays a
# power-of-two multiple of it. Moves with the backbone: a /16 trunk wants 512.
DEFAULT_IMAGE_SIZE = 448
DEFAULT_GRID = 128
DEFAULT_MODEL_PATH = "models/ortho_target.pth"
# (cells) width of the Gaussian the cell head is trained against. 1.5 cells is about 6cm
# of floor at the default grid, which is well inside "the operator would have grabbed
# that". Matches visual_servoing/train.py, which has the same head and the same problem.
CELL_SIGMA = 1.5
# Objectness above which a cell is reported as a target. A per-cell probability means
# the same thing in every frame, so this is a real bar rather than a tuned ratio.
TARGET_THRESHOLD = 0.5
# Which metric picks the saved checkpoint. top5 rather than the single-answer recall,
# because the job is to land on something a person would have picked and there is
# usually more than one such thing in frame; recall@20cm scores those as failures.
SELECTION_METRIC = "top5@20cm"


class OrthoTargetNet(SharedTrunkMixin, nn.Module):
    """Frozen DINOv2 patch features -> an independent objectness per floor location.

    Every cell of the grid x grid map gets its own sigmoid: "is there something here
    worth reaching for". Several objects on the floor are all plausible next targets and
    a frame may hold any number of them, so they must be able to score high at the same
    time - which one softmax over locations cannot do, because its cells compete for a
    fixed unit of mass and a peak's height then says as much about how crowded the frame
    is as about the object under it. Independent cells also make the score comparable
    between frames, so callers threshold it instead of ranking peaks against each other.

    The offset head then places the point inside each winning cell, which is what keeps
    precision finer than the cell size.
    """

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 grid=DEFAULT_GRID, fuse_layers=4, width=256, freeze=True):
        super().__init__()
        trunk = self._init_trunk(backbone_id, freeze)
        self.backbone_id = backbone_id
        self.image_size = image_size
        self.grid = grid
        self.fuse_layers = fuse_layers
        self.freeze = freeze

        config = trunk.config
        self.patch_size = config.patch_size
        self.token_grid = image_size // self.patch_size
        if image_size % self.patch_size:
            raise ValueError(f"image_size {image_size} is not a multiple of patch {self.patch_size}")

        in_ch = config.hidden_size * fuse_layers
        # Two bilinear x2 steps take the 16px token grid to 4px cells at 512 input.
        ups = int(math.log2(grid / self.token_grid))
        if 2 ** ups * self.token_grid != grid:
            raise ValueError(f"grid {grid} is not a power-of-two multiple of {self.token_grid}")

        layers = [nn.Conv2d(in_ch, width, 1), nn.GroupNorm(32, width), nn.GELU()]
        channels = width
        for _ in range(ups):
            nxt = max(64, channels // 2)
            layers += [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels, nxt, 3, padding=1), nn.GroupNorm(32, nxt), nn.GELU(),
            ]
            channels = nxt
        self.decoder = nn.Sequential(*layers)
        self.logit_head = nn.Conv2d(channels, 1, 1)
        self.offset_head = nn.Conv2d(channels, 2, 1)

    def features(self, pixel_values):
        with torch.set_grad_enabled(self.training and not self.freeze):
            out = self.trunk(pixel_values, output_hidden_states=True)
        n_patches = self.token_grid ** 2
        # [CLS] and the register tokens lead the sequence; patches are always the tail.
        feats = [h[:, -n_patches:, :] for h in out.hidden_states[-self.fuse_layers:]]
        x = torch.cat(feats, dim=-1).transpose(1, 2)
        return x.reshape(x.shape[0], x.shape[1], self.token_grid, self.token_grid)

    def forward(self, pixel_values):
        x = self.decoder(self.features(pixel_values))
        return self.logit_head(x).squeeze(1), self.offset_head(x)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.trunk.eval()  # a frozen backbone must not update its norm statistics
        return self


def target_map(cells, mask, grid, sigma):
    """Per-cell objectness target for a batch of labels in continuous cell coordinates.

    A Gaussian around each label rather than a single hot cell, because the cells are
    3.9cm of floor and a prediction one cell out is very nearly right; one-hot scores that
    exactly as wrong as the other side of the room.

    The bumps combine with a max, not a sum: two labels a few cells apart would otherwise
    build a taller peak in the gap between them than either object has on its own, and the
    decoder would then report the empty floor between two socks.
    """
    axis = torch.arange(grid, device=cells.device, dtype=cells.dtype)
    # cell centres are at +0.5, and a label is a continuous cell coordinate
    dx = axis[None, None, :] - (cells[..., 0:1] - 0.5)
    dy = axis[None, None, :] - (cells[..., 1:2] - 0.5)
    gauss = torch.exp(-(dy[..., :, None] ** 2 + dx[..., None, :] ** 2) / (2 * sigma ** 2))
    return (gauss * mask[..., None, None]).amax(dim=1)


def balanced_pos_weight(grid, cell_sigma):
    """Cells in the map per cell inside one label's bump.

    The weight at which a frame's one labelled object counts for as much as all of its
    background, which is the usual starting point for a dense head with a handful of
    positives among thousands of cells.
    """
    return grid * grid / max(2.0 * math.pi * cell_sigma ** 2, 1.0)


def objectness_loss(logits, offsets, points, mask, image_size, grid, offset_weight=1.0,
                    cell_sigma=CELL_SIGMA, pos_weight=None):
    """Per-cell binary objectness, plus L1 on the sub-cell offset of each label's cell.

    The difficulty this head has to survive: a teleop frame labels the one object the
    operator reached for and says nothing about the others, so every other graspable thing
    in the frame is unlabelled rather than absent. A plain BCE reads them as negatives and
    trains the model to deny exactly the objects it is supposed to find.

    `pos_weight` is the correction, and it is the knob worth understanding. Weighting the
    positive term by w moves the equilibrium for a cell whose appearance is the operator's
    pick in a fraction f of the frames it appears in from f to f*w / (1 + f*w) - so
    anything that gets grabbed sometimes rises well clear of floor that never does, and
    the threshold separates "an object" from "not an object" rather than "the object the
    operator happened to pick" from everything else. Raising w finds more and admits more
    false positives; the default balances one bump against a whole map of background.

    Frames that really do carry several labels - the hand-labelled ones - need none of
    this: their positives are all marked, and the same loss handles them unchanged.
    """
    scale = image_size / grid
    cells = points / scale
    # Only as many target slots as this batch actually uses: the padding is masked out
    # anyway, and the Gaussians are the largest tensor here.
    used = max(1, int(mask.sum(1).max().item()))
    cells, mask = cells[:, :used], mask[:, :used]

    if pos_weight is None:
        pos_weight = balanced_pos_weight(grid, cell_sigma)
    target = target_map(cells, mask, grid, cell_sigma)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype))

    cx = cells[..., 0].floor().clamp(0, grid - 1).long()
    cy = cells[..., 1].floor().clamp(0, grid - 1).long()
    index = cy * grid + cx
    frac = cells - torch.stack([cx, cy], dim=-1).to(cells.dtype)
    # (B, 2, T) out of the gather, one offset pair per label, back to (B, T, 2).
    picked = offsets.flatten(2).gather(2, index[:, None, :].expand(-1, 2, -1)).permute(0, 2, 1)
    per_label = F.l1_loss(picked.sigmoid(), frac.clamp(0.0, 1.0), reduction="none").mean(-1)
    l1 = (per_label * mask).sum() / mask.sum().clamp_min(1.0)
    return bce + offset_weight * l1, bce.detach(), l1.detach()


def decode(probs, offsets, image_size, grid, top_k=1, nms_radius=2):
    """Peak cells plus their offsets, as (B, k, 2) pixel coordinates and (B, k) probabilities.

    Peaks, not the expectation over the map: averaging two candidate objects would
    land the prediction on the empty floor between them.
    """
    scale = image_size / grid
    prob = probs.view(-1, 1, grid, grid)

    if top_k > 1:
        # Suppress everything that is not a local maximum so the k results are k
        # distinct candidates rather than k cells of one blob.
        pooled = F.max_pool2d(prob, nms_radius * 2 + 1, stride=1, padding=nms_radius)
        prob = torch.where(prob >= pooled, prob, torch.zeros_like(prob))

    scores, index = prob.flatten(1).topk(top_k, dim=1)
    cx = (index % grid).float()
    cy = torch.div(index, grid, rounding_mode="floor").float()

    off = offsets.flatten(2).sigmoid()
    picked = torch.stack([
        off[:, 0].gather(1, index),
        off[:, 1].gather(1, index),
    ], dim=-1)
    uv = (torch.stack([cx, cy], dim=-1) + picked) * scale
    return uv, scores


def predict(model, images, tta=False, top_k=1):
    """Model outputs for a batch, optionally averaged over the 8 square symmetries.

    Averaged as probabilities, which is also what comes back, so a caller can threshold
    the result whether or not TTA ran. Offsets stay from the untransformed pass, being
    sub-cell either way.
    """
    logits, offsets = model(images)
    probs = logits.sigmoid()
    if tta:
        acc = probs
        for t in range(1, 8):
            transformed, _ = model(dihedral_image(images, t))
            acc = acc + inverse_dihedral_map(transformed.sigmoid(), t)
        probs = acc / 8
    return decode(probs, offsets, model.image_size, model.grid, top_k=top_k)


# ==========================================
# LIVE INFERENCE
# ==========================================
# The observer's ortho worker renders a larger map than the recordings stored, so the
# only step between the live frame and the model is the same resize the training
# loader applies. Room metres come back out analytically, no camera pose involved.

TARGETING_MODEL_REPOID = "naavox/targeting"
TARGETING_MODEL_FILENAME = "ortho_target.pth"


def prepare_ortho_image(rgb, image_size, device):
    """A live ortho frame as a one-image normalized batch, as the training loader made them.

    The frame is RGB, not BGR: anchor clients decode to rgb24 and the floor projection
    keeps that order, which is also the order that reached training - the recorded
    videos are converted to BGR only on their way into the video streamer.
    """
    if rgb.shape[:2] != (image_size, image_size):
        rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    img = (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return img[None].to(device)


@torch.no_grad()
def predict_room_targets(model, rgb, device, top_k=1, tta=False,
                         min_probability=TARGET_THRESHOLD):
    """Every target in one ortho frame, as [(x_m, y_m, probability)] in the room frame.

    A probability is one cell's own objectness, decided independently of every other cell,
    so one absolute bar works on a bare floor and on a crowded one alike - no comparison
    against the best peak, and no multiple of chance, because the number does not shrink
    when a second object enters the frame.

    `top_k` bounds how many peaks are considered at all, so it has to be at least as large
    as the number of objects worth reporting; the threshold does the rest of the work.
    Results come back in descending probability.
    """
    batch = prepare_ortho_image(rgb, model.image_size, device)
    uv, scores = predict(model, batch, tta=tta, top_k=top_k)
    uv, scores = uv[0].cpu().numpy(), scores[0].cpu().numpy()

    out = []
    for (u, v), score in zip(uv, scores):
        if score < min_probability:
            continue
        x_m, y_m = ortho_px_to_room(float(u), float(v), model.image_size, model.image_size)
        out.append((x_m, y_m, float(score)))
    return out


# ==========================================
# TRAINING
# ==========================================


def pixels_to_cm(pixels, image_size):
    """Ortho pixels to centimetres of floor, using the map's fixed metres per side."""
    return pixels * (ORTHO_EXTENT_M * 100.0) / image_size


@torch.no_grad()
def evaluate_model(model, loader, device, top_k=5, tta=False, radii_cm=(10, 20, 50),
                   min_probability=TARGET_THRESHOLD):
    """Score a checkpoint against every label, not just one per frame.

    Distances are measured from each label to its nearest prediction, so a frame with
    four labels contributes four measurements and a model that finds three of them scores
    as having found three. `targets_per_frame` is the count that clears the threshold,
    which is the number that says whether the head has learned to find several objects or
    has quietly collapsed onto one.
    """
    model.eval()
    nearest, covered, predicted, total_bce, count = [], [], [], 0.0, 0
    for images, points, mask in loader:
        images = images.to(device)
        points, mask = points.to(device), mask.to(device)
        logits, offsets = model(images)
        _, bce, _ = objectness_loss(logits, offsets, points, mask, model.image_size, model.grid)
        total_bce += bce.item() * images.shape[0]
        count += images.shape[0]

        uv, scores = predict(model, images, tta=tta, top_k=top_k)
        # (B, labels, k): every label against every decoded candidate.
        distance = (points[:, :, None, :] - uv[:, None, :, :]).norm(dim=-1)
        real = mask > 0
        nearest.append(distance[:, :, 0][real].cpu())
        covered.append(distance.min(dim=2).values[real].cpu())
        predicted.append((scores >= min_probability).sum(dim=1).float().cpu())

    nearest = pixels_to_cm(torch.cat(nearest), model.image_size)
    covered = pixels_to_cm(torch.cat(covered), model.image_size)
    metrics = {
        "bce": total_bce / max(count, 1),
        "median_cm": nearest.median().item(),
        "mean_cm": nearest.mean().item(),
        "targets_per_frame": torch.cat(predicted).mean().item(),
    }
    for radius in radii_cm:
        metrics[f"recall@{radius}cm"] = (nearest <= radius).float().mean().item()
    metrics[f"top{top_k}@20cm"] = (covered <= 20).float().mean().item()
    return metrics


def constant_baseline(train_set, eval_set, image_size, radii_cm=(10, 20, 50)):
    """Score of always predicting the mean training label.

    A floor-target prior is strong enough that this is not a trivial number, and any
    model that fails to beat it has learned nothing about the image.
    """
    train_uv = train_set.scaled_labels()
    eval_uv = eval_set.scaled_labels()
    errors = pixels_to_cm(np.linalg.norm(eval_uv - train_uv.mean(0), axis=1), image_size)
    out = {"median_cm": float(np.median(errors)), "mean_cm": float(errors.mean())}
    for radius in radii_cm:
        out[f"recall@{radius}cm"] = float((errors <= radius).mean())
    return out


def resolve_data_root(args) -> Path:
    """Local distilled dataset, or the hub's copy when no local one was named.

    A repo id on its own - typed or defaulted - means the hub's current version, so this
    syncs rather than serving whatever was downloaded last time. snapshot_download does
    that already; the path is logged because otherwise nothing in the run says which copy
    of a name answered.
    """
    if args.data_root:
        return Path(args.data_root)
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo_id=args.dataset_id, repo_type="dataset"))
    logging.info(f"Using {args.dataset_id} from the hub at {root}")
    return root


def resolve_model_path(model_path) -> str:
    """The checkpoint to evaluate, downloading the published one if there is none local.

    The same shape as resolve_data_root. The default path is only where `train` writes,
    so on a machine that has not trained one there is nothing there and the published
    model is what the user meant; a path they named themselves is an error if it is
    missing, because silently scoring a different model would be worse than stopping.
    """
    if Path(model_path).exists():
        return model_path
    if model_path != DEFAULT_MODEL_PATH:
        raise FileNotFoundError(f"No checkpoint at {model_path}")
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=TARGETING_MODEL_REPOID, filename=TARGETING_MODEL_FILENAME)
    logging.info(f"No {model_path}; using {TARGETING_MODEL_REPOID}/{TARGETING_MODEL_FILENAME} at {path}")
    return path


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    data_root = resolve_data_root(args)
    train_set = OrthoTargetDataset(data_root, "train", args.image_size, augment=True,
                                   seed=args.seed, translate_px=args.translate_px)
    eval_set = OrthoTargetDataset(data_root, "eval", args.image_size, augment=False)
    logging.info(f"train {len(train_set)} sample(s) | eval {len(eval_set)} sample(s) from {data_root}")
    logging.info(f"objectness pos_weight {args.pos_weight or balanced_pos_weight(args.grid, args.cell_sigma):.0f}")

    baseline = constant_baseline(train_set, eval_set, args.image_size)
    logging.info(f"constant-prediction baseline: {_format_metrics(baseline)}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        drop_last=len(train_set) > args.batch_size, pin_memory=device.type == "cuda",
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = OrthoTargetNet(
        backbone_id=args.backbone, image_size=args.image_size, grid=args.grid,
        fuse_layers=args.fuse_layers, freeze=not args.unfreeze_backbone,
    ).to(device)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if args.unfreeze_backbone:
        groups.append({"params": list(model.trunk.parameters()), "lr": args.lr * args.backbone_lr_scale})
        logging.info(f"backbone unfrozen at {args.backbone_lr_scale}x the head learning rate")
    else:
        logging.info(f"backbone frozen; training {sum(p.numel() for p in head_params) / 1e6:.1f}M head parameters")

    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    steps = max(1, len(train_loader)) * args.epochs
    warmup = max(1, int(0.05 * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (
        (s + 1) / warmup if s < warmup
        else 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup)))
    ))
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(3)
        for images, points, mask in train_loader:
            images = images.to(device, non_blocking=True)
            points, mask = points.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                logits, offsets = model(images)
                loss, bce, l1 = objectness_loss(
                    logits.float(), offsets.float(), points, mask,
                    args.image_size, args.grid, args.offset_weight,
                    cell_sigma=args.cell_sigma, pos_weight=args.pos_weight or None,
                )
            loss.backward()
            optimizer.step()
            schedule.step()
            totals += [loss.item(), bce.item(), l1.item()]

        totals /= max(1, len(train_loader))
        line = f"epoch {epoch + 1}/{args.epochs} loss {totals[0]:.4f} (bce {totals[1]:.4f} off {totals[2]:.4f})"

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            metrics = evaluate_model(model, eval_loader, device, top_k=args.top_k)
            logging.info(f"{line} | {_format_metrics(metrics)}")
            score = metrics[args.select_metric]
            if score > best:
                best = score
                torch.save({
                    "state_dict": model.state_dict(),
                    "backbone_id": args.backbone,
                    "image_size": args.image_size,
                    "grid": args.grid,
                    "fuse_layers": args.fuse_layers,
                    # Whether the state dict above holds a backbone at all: a frozen one
                    # is left to dino_trunk's shared instance and never written.
                    "freeze": not args.unfreeze_backbone,
                    "metrics": metrics,
                    "epoch": epoch + 1,
                }, args.model_path)
                logging.info(f"saved {args.model_path} ({args.select_metric} {score:.3f})")
        else:
            logging.info(line)

    logging.info(f"done; best eval {args.select_metric} {best:.3f}, checkpoint at {args.model_path}")


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    freeze = checkpoint.get("freeze", True)
    model = OrthoTargetNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        grid=checkpoint["grid"], fuse_layers=checkpoint["fuse_layers"], freeze=freeze,
    ).to(device)
    state = checkpoint["state_dict"]
    if freeze:
        # Checkpoints written before the trunk was shared still carry it; verify they
        # really are the pretrained weights when the checkpoint does not say so itself.
        state = drop_trunk_weights(state, model.trunk, verify="freeze" not in checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def evaluate(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = resolve_data_root(args)
    model, checkpoint = load_checkpoint(resolve_model_path(args.model_path), device)

    eval_set = OrthoTargetDataset(data_root, args.split, model.image_size, augment=False)
    loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, num_workers=args.workers)
    metrics = evaluate_model(model, loader, device, top_k=args.top_k, tta=args.tta)
    logging.info(f"checkpoint from epoch {checkpoint.get('epoch')} | {_format_metrics(metrics)}")

    if args.preview_dir:
        _write_previews(model, eval_set, device, Path(args.preview_dir), args.top_k, args.tta)
        logging.info(f"previews in {args.preview_dir}")


@torch.no_grad()
def _write_previews(model, dataset, device, out_dir: Path, top_k: int, tta: bool):
    """Ground truth in green, ranked predictions in red, for looking at with your eyes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        image, points, mask = dataset[i]
        uv, scores = predict(model, image[None].to(device), tta=tta, top_k=top_k)
        canvas = cv2.resize(dataset.decode(i), (model.image_size, model.image_size))
        for (gx, gy) in points[mask > 0].numpy():
            cv2.drawMarker(canvas, (int(gx), int(gy)), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        for rank, ((u, v), score) in enumerate(zip(uv[0].cpu().numpy(), scores[0].cpu().numpy())):
            cv2.circle(canvas, (int(u), int(v)), 10, (0, 0, 255), 2 if rank == 0 else 1)
            cv2.putText(canvas, f"{score:.2f}", (int(u) + 12, int(v)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(str(out_dir / dataset.samples[i]["file_name"]), canvas)


def _format_metrics(metrics):
    return " ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
    )


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
                                help="directory to write the distilled dataset into")
    distill_parser.add_argument("--split", default="train", choices=["train", "eval"],
                                help="which split this source becomes; only that directory is "
                                     "rewritten, so the two recipes can be distilled independently")
    distill_parser.add_argument("--dataset_id", default=DEFAULT_DATASET_ID,
                                help="hub repo to upload the distilled dataset to")
    distill_parser.add_argument("--upload", action="store_true",
                                help="upload the result, replacing the hub copy")
    distill_parser.add_argument("--pressure_threshold", type=float, default=0.1,
                                help="finger_pressure above which an episode counts as having made contact")
    distill_parser.add_argument("--frame_offset", type=int, default=0,
                                help="which frame of each episode to keep, counted from its start")
    distill_parser.add_argument("--frames_per_episode", type=int, default=8,
                                help="ortho frames to take from each episode. They share the "
                                     "episode's one label, which is valid for every frame "
                                     "before the grasp because the target does not move")
    distill_parser.add_argument("--frame_stride", type=int, default=4,
                                help="frames between them; too small and they are the same picture")
    distill_parser.add_argument("--min_coverage", type=float, default=0.02,
                                help="skip frames where less than this fraction of the ortho map was "
                                     "painted by any camera")
    distill_parser.add_argument("--limit", type=int, default=0,
                                help="stop after this many samples, for a quick trial run")
    distill_parser.add_argument("--annotate_dir", default=None,
                                help="also write copies with the label drawn on them, to check the projection")

    upload_labels_parser = subparsers.add_parser(
        "upload_labels", help="push targets saved from the UI to a hub dataset of your own")
    upload_labels_parser.add_argument("--source", default=USER_LABEL_ROOT,
                                      help="directory the UI action wrote them to")
    upload_labels_parser.add_argument("--dataset_id", default=None,
                                      help=f"hub dataset to add them to "
                                           f"(default: <your account>/{USER_LABEL_DATASET_NAME})")
    upload_labels_parser.add_argument("--public", action="store_true",
                                      help="create the repo public. It is private by default "
                                           "because these are pictures of your floor")

    merge_labels_parser = subparsers.add_parser(
        "merge_labels", help="fold targets saved from the UI into the distilled dataset")
    source = merge_labels_parser.add_mutually_exclusive_group()
    source.add_argument("--source", default=USER_LABEL_ROOT,
                        help="local directory of label files to merge")
    source.add_argument("--repo_id", default=None,
                        help="merge a hub dataset of label files instead, e.g. one upload_labels wrote")
    merge_labels_parser.add_argument("--output", default=LOCAL_DATASET_ROOT,
                                     help="distilled dataset directory to merge them into")
    merge_labels_parser.add_argument("--split", default="train", choices=["train", "eval"],
                                     help="which split they join. In eval they stop being training "
                                          "data and become a measurement of a different thing: "
                                          "agreement with an operator's click rather than with a "
                                          "grasp that actually happened")
    merge_labels_parser.add_argument("--no_resize", action="store_true",
                                     help="keep the frames at the size they were saved at instead "
                                          "of matching the split's distilled shards")

    def add_data_args(sub):
        sub.add_argument("--dataset_id", default=DEFAULT_DATASET_ID, help="distilled dataset on the hub")
        sub.add_argument("--data_root", default=None,
                         help="local distilled dataset directory (default: download --dataset_id)")
        sub.add_argument("--batch_size", type=int, default=32)
        sub.add_argument("--workers", type=int, default=4)
        sub.add_argument("--top_k", type=int, default=5,
                         help="candidates to decode; the top-k hit rate is the 'picked something "
                              "plausible' measure, which single-answer error cannot capture")
        sub.add_argument("--device", default=None)
        sub.add_argument("--model_path", default=DEFAULT_MODEL_PATH)

    train_parser = subparsers.add_parser("train", help="fit the model on the distilled dataset")
    add_data_args(train_parser)
    train_parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    train_parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE)
    train_parser.add_argument("--grid", type=int, default=DEFAULT_GRID,
                              help="output cells per side; 128 over a 5m map is 3.9cm per cell")
    train_parser.add_argument("--fuse_layers", type=int, default=4,
                              help="how many of the backbone's last blocks to concatenate")
    train_parser.add_argument("--epochs", type=int, default=60)
    # Paired with --batch_size: a batch four times larger takes a quarter as many steps,
    # so the rate rises with it (sqrt of the ratio, the usual compromise for AdamW).
    train_parser.add_argument("--lr", type=float, default=6e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.05)
    train_parser.add_argument("--offset_weight", type=float, default=1.0)
    train_parser.add_argument("--cell_sigma", type=float, default=CELL_SIGMA,
                              help="width in cells of the Gaussian the cell head is trained "
                                   "against; 0 restores a one-hot target")
    train_parser.add_argument("--pos_weight", type=float, default=0,
                              help="weight on the objectness positives; 0 uses the balanced "
                                   "value for --grid and --cell_sigma. Raising it finds more "
                                   "of the objects nobody labelled, at more false positives. "
                                   "See objectness_loss")
    train_parser.add_argument("--select_metric", default=SELECTION_METRIC,
                              help="eval metric that picks the saved checkpoint")
    train_parser.add_argument("--translate_px", type=int, default=48,
                              help="max random shift of the map and its label, in model "
                                   "pixels; 0 disables it")
    train_parser.add_argument("--eval_every", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--unfreeze_backbone", action="store_true",
                              help="also train the backbone, at --backbone_lr_scale of the head's rate")
    train_parser.add_argument("--backbone_lr_scale", type=float, default=0.05)

    eval_parser = subparsers.add_parser("evaluate", help="score a checkpoint on a held-out split")
    add_data_args(eval_parser)
    eval_parser.add_argument("--split", default="eval", choices=["train", "eval"])
    eval_parser.add_argument("--tta", action="store_true",
                             help="average predictions over the 8 square symmetries")
    eval_parser.add_argument("--preview_dir", default=None,
                             help="write images with the label and the predictions drawn on")

    args = parser.parse_args()
    if args.command == "distill":
        distill(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "upload_labels":
        upload_user_labels(args.source, args.dataset_id, private=not args.public)
    elif args.command == "merge_labels":
        merge_labels(args)


if __name__ == "__main__":
    main()
