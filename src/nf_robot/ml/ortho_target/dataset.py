#!/usr/bin/env python

"""The ortho target dataset: distilling, hand labels, splits and the training loader."""

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

from nf_robot.ml.image_input import normalize, photometric_jitter, to_tensor
from nf_robot.ml.ortho_target.model import (
    ORTHO_EXTENT_M,
    dihedral_image,
    dihedral_point,
    room_to_ortho_px,
)

ORTHO_FEED = 3


def ortho_key():
    """The dataset feature holding the ortho view, a function so lerobot is only imported
    when distilling."""
    from nf_robot.ml.lerobot.stringman import _FEED_NAMES

    return f"observation.images.{_FEED_NAMES[ORTHO_FEED]}"
# Feeds this dataset carries: the ortho composite and the gripper camera at the visual
# servoing size.
ORTHO_CAMERA_MODE = "gripper_ortho"


DEFAULT_SOURCE_REPO_ID = "naavox/grip_o"
DEFAULT_DATASET_ID = "naavox/ortho-target-dataset"
LOCAL_DATASET_ROOT = "ortho_target_data"

# Where distill and merge_labels put rows before `split` deals them into train and eval.
POOL_SPLIT = "all"

# Local directory for UI-saved labels, and the hub repo name upload_labels uses under the
# uploader's account.
USER_LABEL_ROOT = "ortho_target_user_labels"
USER_LABEL_DATASET_NAME = "ortho-target-user-labels"

STATE_COMPONENTS = ("gripper_pos_x", "gripper_pos_y", "gripper_pos_z", "finger_pressure")


def is_complete(sample) -> bool:
    return int(sample["episode_index"]) < 0


def frame_to_bgr(frame):
    """A LeRobot video frame (CHW float RGB, or HWC uint8) as an HWC uint8 BGR array."""
    arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] != arr.shape[2]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def coverage_fraction(bgr):
    return float(np.count_nonzero(bgr.any(axis=2))) / float(bgr.shape[0] * bgr.shape[1])


def scan_episode_states(root: Path) -> dict[int, list[dict]]:
    """Per-episode gripper position, pressure and timestamp, read straight from the parquets."""
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
    """The frame with the label drawn on it."""
    out = bgr.copy()
    x, y = int(round(u)), int(round(v))
    cv2.drawMarker(out, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.circle(out, (x, y), 12, (0, 255, 0), 2)
    return out


def episode_frame_offsets(contact_index, frame_offset, count, stride):
    last = contact_index - 1
    offsets = [frame_offset + i * stride for i in range(count)]
    return [o for o in offsets if o <= last]


def build_samples(dataset, pressure_threshold, frame_offset, min_coverage, limit,
                  annotate_dir, frames_per_episode=1, frame_stride=1):
    """Ortho frames before each episode's grasp with the contact pixel, as (samples, images,
    skipped)."""
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

        from nf_robot.ml.lerobot.label_contact_actions import contact_blend_alphas

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
                # The label belongs to the episode, so off the map once is off for every
                # frame.
                break

            # One file per frame, still sorted by episode.
            file_name = f"ep{ep:06d}.jpg" if frames_per_episode == 1 else f"ep{ep:06d}_{n:02d}.jpg"
            images[file_name] = bgr
            samples.append({
                "file_name": file_name,
                # A teleop episode confirms only the one place someone reached.
                "points": [[round(u, 2), round(v, 2)]],
                "contacts_m": [[round(c, 4) for c in (x_m, y_m, z_m)]],
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
# Label columns beside the JPEG bytes, readable without decoding any frame.
LABEL_COLUMNS = ("file_name", "points", "contacts_m", "episode_index", "frame_offset",
                 "contact_frame_index", "contact_time_s", "task")


def shard_schema():
    import pyarrow as pa

    return pa.schema([
        pa.field("file_name", pa.string()),
        pa.field("image", pa.binary()),
        # One row per frame with every target in it; contacts_m runs parallel to points.
        pa.field("points", pa.list_(pa.list_(pa.float64()))),
        pa.field("contacts_m", pa.list_(pa.list_(pa.float64()))),
        pa.field("episode_index", pa.int32()),
        pa.field("frame_offset", pa.int32()),
        pa.field("contact_frame_index", pa.int32()),
        pa.field("contact_time_s", pa.float64()),
        pa.field("task", pa.string()),
    ])


def write_row_shards(split_dir: Path, rows, target_bytes=SHARD_TARGET_BYTES) -> int:
    """Write rows as parquet shards of roughly target_bytes each, since loose files get rate
    limited on the hub."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = shard_schema()
    written, shard, batch, pending = 0, 0, [], 0

    def flush():
        nonlocal shard, batch, pending
        if not batch:
            return
        pq.write_table(pa.Table.from_pylist(batch, schema=schema),
                       split_dir / f"shard-{shard:04d}.parquet")
        shard, batch, pending = shard + 1, [], 0

    for row in sorted(rows, key=lambda r: r["file_name"]):
        batch.append(row)
        pending += len(row["image"])
        written += 1
        if pending >= target_bytes:
            flush()
    flush()
    return written


def write_shards(split_dir: Path, samples, images, target_bytes=SHARD_TARGET_BYTES) -> int:
    """Encode each sample's frame and write the lot as shards."""
    rows = []
    for sample in samples:
        ok, buf = cv2.imencode(".jpg", images[sample["file_name"]])
        if not ok:
            raise ValueError(f"could not encode {sample['file_name']}")
        rows.append({
            "file_name": sample["file_name"],
            "image": buf.tobytes(),
            "points": sample["points"],
            "contacts_m": sample["contacts_m"],
            "episode_index": int(sample["episode_index"]),
            "frame_offset": int(sample.get("frame_offset", 0)),
            "contact_frame_index": int(sample["contact_frame_index"]),
            "contact_time_s": float(sample["contact_time_s"]),
            "task": sample.get("task") or "",
        })
    return write_row_shards(split_dir, rows, target_bytes)


def write_split(output_root: Path, split: str, samples, images) -> int:
    """Rebuild one directory of the dataset from scratch, leaving the others alone."""
    split_dir = output_root / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    written = write_shards(split_dir, samples, images)

    write_dataset_readme(output_root)
    return written


def read_rows(split_dir: Path):
    """Every row of a directory of shards, images and all."""
    rows = []
    for shard in sorted(split_dir.glob("*.parquet")):
        rows.extend(pq.read_table(shard).cast(shard_schema()).to_pylist())
    return rows


def sample_group(sample):
    """What a row's near-duplicates share: its episode, or its labelling run."""
    if not is_complete(sample):
        return ("episode", int(sample["episode_index"]))
    # user-<contributor>-<run>-<frame>.jpg, so everything up to the frame is the run
    return ("run", sample["file_name"].rsplit(".", 1)[0].rsplit("-", 1)[0])


def split_pool(output_root, eval_fraction=0.1, seed=0):
    """Deal the pool into train and eval at random, one row at a time."""
    output_root = Path(output_root)
    rows = read_rows(output_root / POOL_SPLIT)
    if not rows:
        raise FileNotFoundError(
            f"No shards in {output_root / POOL_SPLIT}. Distill into the pool first, then "
            f"merge any hand labels into it, then split.")

    order = np.random.default_rng(seed).permutation(len(rows))
    cut = int(round(len(rows) * eval_fraction))
    picked = {"eval": [rows[i] for i in order[:cut]], "train": [rows[i] for i in order[cut:]]}

    for split, chosen in picked.items():
        split_dir = output_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        write_row_shards(split_dir, chosen)
        complete = [r for r in chosen if is_complete(r)]
        logging.info(
            f"{split}: {len(chosen)} frame(s), {sum(len(r['points']) for r in chosen)} target(s), "
            f"{len(complete)} of them complete ({sum(len(r['points']) for r in complete)} target(s))")
    write_dataset_readme(output_root)

    trained = {sample_group(r) for r in picked["train"]}
    shared = sum(1 for r in picked["eval"] if sample_group(r) in trained)
    logging.info(
        f"{shared} of {len(picked['eval'])} eval frame(s) come from an episode or labelling run "
        f"that also appears in train, so they are near-duplicates of something trained on; see "
        f"split_pool for what that makes the eval numbers mean")
    return len(picked["train"]), len(picked["eval"])

def write_dataset_readme(output_root: Path):
    """The hub's dataset config block, naming whichever splits are actually present."""
    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for name, dirname in (("train", "train"), ("test", "eval")):
        if any((output_root / dirname).glob("*.parquet")):
            lines += [f"  - split: {name}", f"    path: {dirname}/*.parquet"]
    (output_root / "README.md").write_text("\n".join(lines) + "\n---\n")


def write_user_labels(rgb, targets_m, output_root=USER_LABEL_ROOT, extent_m=ORTHO_EXTENT_M,
                      name=None, allow_empty=False):
    """Write one parquet row of an ortho frame and every target the operator placed on it;
    returns (path, target count)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("could not encode the ortho frame")
    blob = buf.tobytes()

    batch = name or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    targets_m = list(targets_m)
    points, contacts = [], []
    for x_m, y_m, z_m in targets_m:
        u, v = room_to_ortho_px(x_m, y_m, width, height, extent_m)
        if not (0 <= u < width and 0 <= v < height):
            continue
        points.append([round(float(u), 2), round(float(v), 2)])
        contacts.append([round(float(c), 4) for c in (x_m, y_m, z_m)])
    # Deliberately empty is a label; empty because the targets missed the map is a mistake.
    if not points and (targets_m or not allow_empty):
        return None, 0

    rows = [{
        "file_name": f"user-{batch}.jpg",
        "image": blob,
        "points": points,
        "contacts_m": contacts,
        # No episode stands behind these, so the episode fields are -1.
        "episode_index": -1,
        "frame_offset": 0,
        "contact_frame_index": -1,
        "contact_time_s": 0.0,
        "task": "user target",
    }]

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"user-{batch}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=shard_schema()), path)
    return path, len(points)


def upload_dataset(output_root: Path, dataset_id: str):
    """Replace the hub copy with this one, pruning anything absent locally."""
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
        # The pool stays local; only the splits go up.
        ignore_patterns=[f"{POOL_SPLIT}/*"],
        # *.jpg clears loose frames left by the pre-shard layout.
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
schema as naavox/ortho-target-dataset - so `python -m nf_robot.ml.ortho_target merge_labels
--repo_id <this>` folds them into a distilled dataset. See nf_robot/ml/ortho_target/readme.md.
"""


def user_label_dataset_id(dataset_id=None):
    """The hub repo for this account's user labels, defaulting to the logged-in account."""
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
    """Add local user labels to a hub dataset without pruning, since the hub copy may be the
    only one."""
    from huggingface_hub import HfApi, create_repo

    source_root = Path(source_root)
    files = sorted(source_root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No label files in {source_root}. The UI's 'Add targets to dataset' action writes "
            f"them there, relative to the directory stringman was started from."
        )
    rows = sum(len(points) for f in files
               for points in pq.read_table(f, columns=["points"]).column("points").to_pylist())

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
    """The size a split's distilled frames are stored at, ignoring merged labels, or None."""
    for shard in sorted(split_dir.glob("*.parquet")):
        if not shard.name.startswith("user-"):
            return first_stored_size(shard)
    return None


def resize_rows(rows, size):
    """Rows re-encoded at (width, height), with their labels scaled to match."""
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
                    "points": [[round(u * su, 2), round(v * sv, 2)] for u, v in row["points"]]})
    return out


def merged_label_name(path, tag=None):
    """Destination filename for one submission, namespaced by contributor tag so
    contributors don't overwrite each other."""
    stem = path.stem[len("user-"):] if path.stem.startswith("user-") else path.stem
    return f"user-{tag}-{stem}.parquet" if tag else f"user-{stem}.parquet"


def merge_user_labels(source_root, output_root=LOCAL_DATASET_ROOT, split=POOL_SPLIT,
                      resize=True, tag=None):
    """Copy user labels into the distilled dataset (the pool by default), resized to match
    and kept in their own files."""
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
        # Cast rather than trust, so a drifted column from another machine fails loudly.
        rows = pq.read_table(path).cast(schema).to_pylist()
        if target_size:
            rows = resize_rows(rows, target_size)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                       split_dir / merged_label_name(path, tag))
        merged += sum(len(row["points"]) for row in rows)

    write_dataset_readme(output_root)
    logging.info(f"Merged {len(files)} submission(s), {merged} label(s), into {split_dir}"
                 + (f" at {target_size[0]}x{target_size[1]}" if target_size else ""))
    return len(files), merged


def merge_labels(args):
    source = args.source
    # Tag merged labels with the repo owner by default.
    tag = args.tag
    if args.repo_id:
        from huggingface_hub import snapshot_download

        source = snapshot_download(repo_id=args.repo_id, repo_type="dataset")
        tag = tag or args.repo_id.split("/")[0]
        logging.info(f"Merging {args.repo_id} from the hub at {source}")
    merge_user_labels(source, args.output, args.split, resize=not args.no_resize, tag=tag)
    if args.split == POOL_SPLIT:
        logging.info(f"Deal the splits from the pool next: python -m nf_robot.ml.ortho_target "
                     f"split --data_root {args.output}")
    else:
        logging.info(f"Train on the result with: python -m nf_robot.ml.ortho_target train "
                     f"--data_root {args.output}")


def split_dataset(args):
    train, evaluation = split_pool(args.data_root, args.eval_fraction, args.seed)
    logging.info(f"Dealt {train + evaluation} pooled frame(s) into train and eval with "
                 f"seed {args.seed}; re-run with another --seed to re-deal")
    if args.upload:
        upload_dataset(Path(args.data_root), args.dataset_id)
    else:
        logging.info(f"Not uploading; pass --upload to push to {args.dataset_id}")
    logging.info(f"Train on the result with: python -m nf_robot.ml.ortho_target train "
                 f"--data_root {args.data_root}")


def distill(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(args.root) if args.root else None
    # A repo id without --root means the hub's copy, so re-sync rather than read a stale
    # cache.
    dataset = LeRobotDataset(repo_id=args.repo_id, root=root, force_cache_sync=root is None)
    key = ortho_key()
    if key not in dataset.meta.video_keys:
        raise ValueError(
            f"'{args.repo_id}' has no '{key}' feature, so it carries no ortho view. "
            f"Build it with recipes/combined_targets_reblend.yaml. Present: {list(dataset.meta.video_keys)}"
        )

    # Log the resolved root, since the repo id doesn't say which copy answered.
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
    written = write_split(output_root, POOL_SPLIT, samples, images)

    logging.info(f"Wrote {written} sample(s) to {output_root / POOL_SPLIT}")
    dropped = ", ".join(f"{reason}={n}" for reason, n in skipped.items() if n)
    logging.info(f"Dropped episodes: {dropped or 'none'}")
    if args.annotate_dir:
        logging.info(f"Annotated previews in {args.annotate_dir}")
    logging.info("Merge any hand labels into the pool next (merge_labels), then deal the "
                 "splits from it (split)")


# ==========================================
# AUGMENTATION
# ==========================================
# The 8 square symmetries are in model.py.

def translate(img, u, v, max_px: int, rng):
    """Shift the map and its labels together by up to max_px, filling vacated edges with black."""
    # Bounded by the outermost label so no label leaves the map.
    height, width = img.shape[-2:]
    if len(u):
        lo_x, hi_x = int(math.ceil(-u.min())), int(math.floor(width - 1 - u.max()))
        lo_y, hi_y = int(math.ceil(-v.min())), int(math.floor(height - 1 - v.max()))
    else:
        # A frame with no labels is limited only by max_px.
        lo_x, hi_x = -max_px, max_px
        lo_y, hi_y = -max_px, max_px
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


# ==========================================
# DATASET
# ==========================================

# Labels kept per frame, padded so the default collate can stack them.
MAX_TARGETS = 16


class OrthoTargetDataset(torch.utils.data.Dataset):
    """Distilled ortho frames as (image, points, mask, complete), with points padded to
    MAX_TARGETS."""

    def __init__(self, root: Path, split: str, image_size: int, augment: bool, seed: int = 0,
                 translate_px: int = 0):
        self.dir = Path(root) / split
        shards = sorted(self.dir.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No parquet shards at {self.dir}. A distilled dataset from before this was "
                f"sharded holds loose jpegs and a metadata.jsonl; re-run distill to convert it.")

        # Frames are held as JPEG bytes; decoded they would be several GB.
        self.images: list[bytes] = []
        self.samples: list[dict] = []
        for shard in shards:
            table = pq.read_table(shard)
            if "points" not in table.column_names:
                raise ValueError(
                    f"{shard} predates the one-row-per-frame format: it holds flat u/v "
                    f"columns and one row per target. Re-run distill to rebuild the split.")
            blobs = table.column("image").to_pylist()
            self.images.extend(blobs)
            self.samples.extend(table.select(list(LABEL_COLUMNS)).to_pylist())

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
        """Each sample's first label in the model's pixel space, for the one-point baseline."""
        width, height = self.stored_size()
        scale = np.array([self.image_size / width, self.image_size / height])
        first = [s["points"][0] for s in self.samples if s["points"]]
        return np.array(first, dtype=np.float64).reshape(-1, 2) * scale

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bgr = self.decode(idx)

        # reshape so an empty label list still gives an (n, 2) array.
        points = np.asarray(sample["points"], dtype=np.float64).reshape(-1, 2)
        h, w = bgr.shape[:2]
        if (w, h) != (self.image_size, self.image_size):
            points = points * [self.image_size / w, self.image_size / h]
            bgr = cv2.resize(bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = to_tensor(rgb)

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

        img = normalize(img)

        kept = min(len(points), MAX_TARGETS)
        padded = np.zeros((MAX_TARGETS, 2), dtype=np.float32)
        padded[:kept] = points[:kept]
        mask = np.zeros(MAX_TARGETS, dtype=np.float32)
        mask[:kept] = 1.0
        complete = torch.tensor(float(is_complete(sample)), dtype=torch.float32)
        return img, torch.from_numpy(padded), torch.from_numpy(mask), complete
