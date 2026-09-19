#!/usr/bin/env python

"""Mine visual servoing labels by projecting each teleop grasp point back into the frames before it.

Usage:
    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/empty-floor-sweep --negatives \
        --output_root datasets/visual_servoing

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/false-grabs --false_grabs \
        --output_root datasets/visual_servoing

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/bedroom-laundry-aug7-2 \
        --root datasets/bedroom-laundry-aug7-2 \
        --output_root data/visual_servoing \
        --preview_dir data/visual_servoing/preview

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/bedroom-laundry-aug7-2 --preview_only \
        --preview_dir /tmp/label_check --limit 20
"""

import argparse
import json
import logging
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.visual_servoing.uv_methods import (
    DEFAULT_UV_METHOD, MIN_DEPTH_M, PIXEL_METHODS, add_uv_arguments,
    anchor_is_usable, gripper_camera_calibration, grasp_point_room, project,
    target_track,
)
from nf_robot.ml.lerobot.trim_to_grasp import (
    MIN_GRASP_SECONDS,
    PRESSURE_THRESHOLD,
    RISE_M,
    find_grasp,
)

# (seconds) how much of the approach before the grasp to label.
APPROACH_SECONDS = 10.0
# (seconds) how much of the carry after the grasp to keep, for the holding head.
CARRY_SECONDS = 3.0
# The canvas the target head predicts over, as a fraction of the frame; must match
# model.CANVAS_SCALE.
CANVAS_SCALE = 1.25
# How far outside the frame, as a fraction of it, a target keeps its position labels.
OFF_SCREEN_MARGIN = 0.10
# Shard prefixes of the whole-recording modes, so each rerun replaces only its own output.
NEGATIVE_PREFIX = "negative"
FALSE_GRAB_PREFIX = "false_grab"

# Every producer writes into this pool, which split_pool deals into train/ and eval/.
POOL_SPLIT = "all"
# Keep one frame in this many when a whole recording carries one label.
SWEEP_STRIDE = 5

# What a source recording is, and so what its frames can be labelled with.
MODE_GRASPS = "grasps"
MODE_NEGATIVES = "negatives"
MODE_FALSE_GRABS = "false_grabs"
# (frames) the longest pause in the closing command that still counts as the same close.
CLOSE_GAP_FRAMES = 10
# (metres, seconds) what counts as the lift having started: this much climb, still
# climbing this long later.
LIFT_ONSET_M = 0.03
LIFT_CONFIRM_SECONDS = 0.5
# (normalized 0-1) finger_pressure held this low for MIN_GRASP_SECONDS means the object is
# gone.
RELEASE_PRESSURE = 0.05
# Full-scale commanded finger speed, used to normalize the finger label into -1..1.
FINGER_SPEED_FULL_SCALE = 90.0
# Stored frame size, the model's input; 252 = 14 x 18 for the /14 backbone.
IMAGE_SIZE = (448, 252)
JPEG_QUALITY = 90
# Target bytes of image data per parquet shard, to keep the hub file count down.
SHARD_TARGET_BYTES = 512 * 1024 * 1024
# Rows per parquet row group, small so the preview can read scattered rows cheaply.
ROW_GROUP_SIZE = 256

STATE_NEEDED = (
    "gripper_pos_x", "gripper_pos_y", "gripper_pos_z",
    "spin", "finger_pressure", "wrist_angle", "finger_angle",
    "laser_rangefinder", "target_force",
)
# Only the dead-reckoning uv methods need these, so they read as None when absent.
VELOCITY_OPTIONAL = {"vel_cmd": ("action", ("vel_x", "vel_y", "vel_z")),
                     "vel_obs": ("state", ("vel_x", "vel_y", "vel_z"))}


def row_schema():
    """Parquet schema for one labelled frame, where a null label masks that head's loss."""
    import pyarrow as pa

    return pa.schema([
        ("image", pa.binary()),
        ("split_source", pa.string()),
        ("source_repo_id", pa.string()),
        ("episode_index", pa.int32()),
        ("frame_index", pa.int32()),
        ("seconds_to_grasp", pa.float32()),
        # A plain list, since parquet cannot store a null in a fixed-size list.
        ("target_uv", pa.list_(pa.float32())),
        ("target_range_m", pa.float32()),
        ("grasp_axis_rad", pa.float32()),
        ("finger", pa.float32()),
        # 1 from the frame the operator began closing on, 0 before it
        ("close_now", pa.int8()),
        # Grip force at the lift, the same on every frame of the episode.
        ("grasp_pressure", pa.float32()),
        ("target_present", pa.int8()),
        ("holding", pa.int8()),
        ("state", pa.struct([
            ("laser_rangefinder", pa.float32()),
            ("finger_angle", pa.float32()),
            ("target_force", pa.float32()),
        ])),
    ])


class ShardWriter:
    """Buffers rows and flushes them as parquet shards of roughly SHARD_TARGET_BYTES."""

    # The miner's shard prefix; each producer only deletes its own files.
    DEFAULT_PREFIX = "shard"

    def __init__(self, split_dir: Path, target_bytes: int = SHARD_TARGET_BYTES,
                 prefix: str = DEFAULT_PREFIX):
        self.split_dir = split_dir
        # Shards are named by producer so producers can share a split.
        self.prefix = prefix
        self.target_bytes = target_bytes
        self.schema = row_schema()
        self.rows: list[dict] = []
        self.pending = 0
        self.shards = 0
        self.total = 0

    def add(self, row: dict):
        self.rows.append(row)
        self.pending += len(row["image"])
        self.total += 1
        if self.pending >= self.target_bytes:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.split_dir / f"{self.prefix}-{self.shards:04d}.parquet"
        pq.write_table(
            pa.Table.from_pylist(self.rows, schema=self.schema),
            path, compression="snappy", row_group_size=ROW_GROUP_SIZE,
        )
        logging.info(f"wrote {path.name}: {len(self.rows)} rows, {self.pending / 1e6:.0f} MB")
        self.shards += 1
        self.rows = []
        self.pending = 0


class ReservoirSampler:
    """A ShardWriter stand-in that keeps a deterministic uniform random sample of `count` rows."""

    def __init__(self, count: int, seed: int):
        self.count = count
        self.rng = random.Random(seed)
        self.rows: list[dict] = []
        self.total = 0

    def add(self, row: dict):
        self.total += 1
        if len(self.rows) < self.count:
            self.rows.append(row)
            return
        i = self.rng.randrange(self.total)
        if i < self.count:
            self.rows[i] = row

    def flush(self):
        pass


def shard_prefix(mode):
    """The shard prefix a mode writes under, so a rerun replaces only its own output."""
    return {
        MODE_NEGATIVES: NEGATIVE_PREFIX,
        MODE_FALSE_GRABS: FALSE_GRAB_PREFIX,
    }.get(mode, ShardWriter.DEFAULT_PREFIX)


def encode_frame(bgr, image_size=IMAGE_SIZE):
    """One frame as JPEG bytes at the model's input resolution."""
    resized = cv2.resize(bgr, tuple(image_size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def read_columns(root: Path):
    """Per-episode state and action rows, read as columns from the parquets."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    state_names = info["features"]["observation.state"]["names"]
    action_names = info["features"]["action"]["names"]
    missing = [n for n in STATE_NEEDED if n not in state_names]
    if missing:
        raise ValueError(f"{root} observation.state is missing {missing}; present: {state_names}")
    if "finger_speed" not in action_names:
        raise ValueError(f"{root} action has no finger_speed; present: {action_names}")

    si = {n: i for i, n in enumerate(state_names)}
    finger_idx = action_names.index("finger_speed")
    names = {"state": state_names, "action": action_names}
    velocity = {
        field: [names[where].index(n) for n in components]
        for field, (where, components) in VELOCITY_OPTIONAL.items()
        if all(n in names[where] for n in components)
    }

    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data parquets under {root}/data")

    episodes: dict[int, list[dict]] = {}
    for path in files:
        table = pq.read_table(path, columns=[
            "episode_index", "frame_index", "timestamp", "observation.state", "action"])
        for ep, fi, ts, state, action in zip(
            table.column("episode_index").to_pylist(),
            table.column("frame_index").to_pylist(),
            table.column("timestamp").to_pylist(),
            table.column("observation.state").to_pylist(),
            table.column("action").to_pylist(),
        ):
            episodes.setdefault(ep, []).append({
                "frame_index": fi,
                "timestamp": ts,
                "gripper_pos": np.array([state[si[f"gripper_pos_{a}"]] for a in "xyz"]),
                "spin": state[si["spin"]],
                "pressure": state[si["finger_pressure"]],
                "wrist_angle": state[si["wrist_angle"]],
                "finger_angle": state[si["finger_angle"]],
                "laser_rangefinder": state[si["laser_rangefinder"]],
                "target_force": state[si["target_force"]],
                "finger_speed": action[finger_idx],
                "vel_cmd": (np.array([action[i] for i in velocity["vel_cmd"]])
                            if "vel_cmd" in velocity else None),
                "vel_obs": (np.array([state[i] for i in velocity["vel_obs"]])
                            if "vel_obs" in velocity else None),
            })
    for rows in episodes.values():
        rows.sort(key=lambda r: r["frame_index"])
    return episodes, float(info["fps"])


def close_onset(rows, grasp, gap_frames=CLOSE_GAP_FRAMES):
    """The frame the operator began the close that ended in this grasp, or None if it began
    before the window."""
    i, gap = grasp, 0
    onset = grasp
    while i >= 0:
        if rows[i]["finger_speed"] > 0:
            onset, gap = i, 0
        else:
            gap += 1
            if gap > gap_frames:
                break
        i -= 1
    return None if onset == 0 else onset


def find_lift(rows, grasp, fps):
    """The frame the gripper started carrying the object up, or None if it never rose."""
    start_z = rows[grasp]["gripper_pos"][2]
    settle = max(1, int(round(LIFT_CONFIRM_SECONDS * fps)))
    for i in range(grasp, len(rows)):
        if rows[i]["gripper_pos"][2] - start_z < LIFT_ONSET_M:
            continue
        later = rows[min(i + settle, len(rows) - 1)]["gripper_pos"][2]
        if later >= rows[i]["gripper_pos"][2]:
            return i
    return None


def find_drop(rows, lift, fps, release_pressure=RELEASE_PRESSURE,
              min_release_seconds=MIN_GRASP_SECONDS):
    """The frame the object stopped being held after `lift` (an open command or a sustained
    pressure drop), or None."""
    drop = None
    for i in range(lift, len(rows)):
        if rows[i]["finger_speed"] < 0:
            drop = i
            break

    hold = max(1, int(round(min_release_seconds * fps)))
    tail = np.array([r["pressure"] for r in rows[lift:]], dtype=np.float32)
    if len(tail) >= hold:
        under = (tail < release_pressure).astype(np.int32)
        gone = np.convolve(under, np.ones(hold, dtype=np.int32), mode="valid") == hold
        collapsed = np.flatnonzero(gone)
        if collapsed.size:
            slipped = lift + int(collapsed[0])
            drop = slipped if drop is None else min(drop, slipped)
    return drop


def holding_label(i, grasp, lift, drop):
    """Whether frame `i` shows an object securely held: 1 from lift to drop, 0 before the
    grasp, None between."""
    if i < grasp:
        return 0
    if lift is None or i < lift:
        return None
    if drop is not None and i >= drop:
        return 0
    return 1


def wrap_pi(radians):
    """Fold an angle into [-pi/2, pi/2), the range a pi-periodic grasp axis lives in."""
    return (radians + math.pi / 2) % math.pi - math.pi / 2


def in_view(u, v, margin=OFF_SCREEN_MARGIN):
    return -margin <= u <= 1 + margin and -margin <= v <= 1 + margin


def mine_episode(rows, fps, calibration, approach_seconds, carry_seconds, rise_m,
                 margin=OFF_SCREEN_MARGIN, uv_method=DEFAULT_UV_METHOD, jaw_uv=None,
                 frames=None):
    """Labelled rows for one grasp episode as (rows, dropped, blind), or (None, reason, 0)
    if unusable."""
    pressure = np.array([r["pressure"] for r in rows], dtype=np.float32)
    grasp = find_grasp(pressure, fps, PRESSURE_THRESHOLD, MIN_GRASP_SECONDS)
    if grasp is None:
        return None, "no_grasp", 0

    heights = np.array([r["gripper_pos"][2] for r in rows])
    if not np.any(heights[grasp:] >= heights[grasp] + rise_m):
        # closed on nothing, or on something it could not pick up
        return None, "no_rise", 0

    # A grasp frame with no usable rangefinder reading has no target to label.
    if not anchor_is_usable(rows[grasp], calibration):
        return None, "no_range", 0

    track = target_track(rows, grasp, calibration, uv_method, jaw_uv, frames)
    wrist_at_grasp = rows[grasp]["wrist_angle"]
    onset = close_onset(rows, grasp)
    lift = find_lift(rows, grasp, fps)
    drop = None if lift is None else find_drop(rows, lift, fps)
    # The force that turned out to be enough, read at the frame it proved itself on.
    pressure_at_lift = None if lift is None else float(rows[lift]["pressure"])

    first = max(0, grasp - int(round(approach_seconds * fps)))
    last = min(len(rows) - 1, grasp + int(round(carry_seconds * fps)))

    out, dropped, blind = [], 0, 0
    for i in range(first, last + 1):
        r = rows[i]
        sample = {
            "split_source": "teleop",
            "frame_index": r["frame_index"],
            "seconds_to_grasp": round(rows[grasp]["timestamp"] - r["timestamp"], 3),
            "target_uv": None,
            "target_range_m": None,
            "grasp_axis_rad": None,
            "finger": round(float(r["finger_speed"]) / FINGER_SPEED_FULL_SCALE, 4),
            # Whether the close should have begun by this frame, masked when the onset is
            # outside the window.
            "close_now": None if onset is None else (1 if i >= onset else 0),
            "grasp_pressure": (None if pressure_at_lift is None
                               else round(pressure_at_lift, 4)),
            "target_present": 1 if i <= grasp else None,
            "holding": holding_label(i, grasp, lift, drop),
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "target_force": round(float(r["target_force"]), 4),
            },
        }

        # Only up to the grasp, after which the object rides in the jaws.
        if i <= grasp:
            projected = track[i]
            if projected is None:
                dropped += 1
                continue
            u, v, distance = projected
            half = (CANVAS_SCALE - 1.0) / 2.0
            if not (-half <= u <= 1 + half and -half <= v <= 1 + half):
                dropped += 1
                continue
            if in_view(u, v, margin):
                sample["target_uv"] = [round(u, 5), round(v, 5)]
                sample["target_range_m"] = round(distance, 4)
                sample["grasp_axis_rad"] = round(
                    wrap_pi(math.radians(wrist_at_grasp - r["wrist_angle"])), 5)
            else:
                # Too far off-frame: keep the frame but mask the position labels and
                # target_present.
                blind += 1
                sample["target_present"] = None

        out.append(sample)
    return out, dropped, blind


def mine_negative_episode(rows, stride=SWEEP_STRIDE):
    """Rows for one episode of an empty-floor recording: target_present=0 and no position labels."""
    out = []
    for i in range(0, len(rows), max(1, stride)):
        r = rows[i]
        empty = float(r["pressure"]) < PRESSURE_THRESHOLD
        out.append({
            "split_source": "teleop",
            "frame_index": r["frame_index"],
            "seconds_to_grasp": None,
            "target_uv": None,
            "target_range_m": None,
            "grasp_axis_rad": None,
            "finger": round(float(r["finger_speed"]) / FINGER_SPEED_FULL_SCALE, 4),
            # 0 as a fact: bare floor is exactly where a close should not begin.
            "close_now": 0,
            # But how hard to squeeze has no answer with nothing to squeeze.
            "grasp_pressure": None,
            "target_present": 0,
            "holding": 0 if empty else None,
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "target_force": round(float(r["target_force"]), 4),
            },
        })
    return out, 0, 0


def mine_false_grab_episode(rows, stride=SWEEP_STRIDE):
    """Rows for one episode of a false-grab recording: close_now=0 and holding=0, everything
    else masked."""
    out = []
    for i in range(0, len(rows), max(1, stride)):
        r = rows[i]
        out.append({
            "split_source": "teleop",
            "frame_index": r["frame_index"],
            "seconds_to_grasp": None,
            "target_uv": None,
            "target_range_m": None,
            "grasp_axis_rad": None,
            "finger": None,
            "close_now": 0,
            "grasp_pressure": None,
            "target_present": None,
            "holding": 0,
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "target_force": round(float(r["target_force"]), 4),
            },
        })
    return out, 0, 0


def split_image_size(split_dir: Path):
    """The frame size the shards in a split are already written at, or None if empty."""
    import pyarrow.parquet as pq

    for path in sorted(split_dir.glob("*.parquet")):
        table = pq.read_table(path, columns=["image"]).slice(0, 1)
        blobs = table.column("image").to_pylist()
        if not blobs:
            continue
        img = cv2.imdecode(np.frombuffer(blobs[0], np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return (img.shape[1], img.shape[0])
    return None


def source_episode_count(root: Path) -> int:
    return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])


IMAGE_KEY = "observation.images.gripper_camera"


def frame_bgr(dataset, index, image_key=IMAGE_KEY):
    frame = dataset[index][image_key]
    return cv2.cvtColor(
        (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def check_source(source):
    """Whether a teleop dataset (directory or hub repo id) can be mined, from its metadata alone."""
    root = Path(source)
    if root.is_dir():
        info = json.loads((root / "meta" / "info.json").read_text())
    else:
        from huggingface_hub import hf_hub_download

        info = json.loads(Path(hf_hub_download(
            repo_id=str(source), repo_type="dataset",
            filename="meta/info.json")).read_text())

    features = info.get("features", {})
    state_names = features.get("observation.state", {}).get("names") or []
    action_names = features.get("action", {}).get("names") or []
    # names can arrive as {"motors": [...]} in some writers
    if isinstance(state_names, dict):
        state_names = sum(state_names.values(), [])
    if isinstance(action_names, dict):
        action_names = sum(action_names.values(), [])
    cameras = [k for k in features if k.startswith("observation.images.")]

    missing = [n for n in STATE_NEEDED if n not in state_names]
    if "finger_speed" not in action_names:
        missing.append("action.finger_speed")
    if IMAGE_KEY not in cameras:
        missing.append(IMAGE_KEY)

    return {
        "source": str(source),
        "ok": not missing,
        "missing": missing,
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "codebase_version": info.get("codebase_version"),
        "cameras": cameras,
        "state_names": state_names,
    }


def hub_root(repo_id):
    """A hub dataset's local root, downloading it if needed, with a clear error when the
    version tag is missing."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        return LeRobotDataset(repo_id).root
    except Exception as error:
        from huggingface_hub import HfApi

        try:
            tags = [t.name for t in
                    HfApi().list_repo_refs(repo_id, repo_type="dataset").tags]
        except Exception:
            raise error
        if not tags:
            raise RuntimeError(
                f"{repo_id} has no version tag, so lerobot cannot resolve it. A dataset "
                f"uploaded with upload_folder rather than push_to_hub needs one: create a "
                f"tag named after the codebase_version in its meta/info.json, e.g.\n"
                f"    HfApi().create_tag('{repo_id}', tag='v3.0', repo_type='dataset')\n"
                f"Or pass --root to read it from disk and skip the hub entirely."
            ) from error
        raise


def report_sources(sources):
    """Print what check_source found for each, and return the ones that can be mined."""
    usable = []
    for source in sources:
        try:
            result = check_source(source)
        except Exception as error:
            logging.warning(f"{source}: cannot read metadata ({error})")
            continue
        head = "MINEABLE  " if result["ok"] else "unusable  "
        logging.info(f"{head}{result['source']}: {result['episodes']} episodes, "
                     f"{result['frames']} frames at {result['fps']}fps, "
                     f"lerobot v{result['codebase_version']}")
        logging.info(f"    cameras: {result['cameras'] or 'none'}")
        if result["missing"]:
            logging.info(f"    missing: {result['missing']}")
            logging.info(f"    state:   {result['state_names']}")
        else:
            usable.append(result["source"])
    return usable


def mine_source(writer, root: Path, repo_id: str, approach_seconds: float,
                carry_seconds: float, rise_m: float, limit: int | None, progress=None,
                mode: str = MODE_GRASPS, stride: int = SWEEP_STRIDE,
                image_size=IMAGE_SIZE, fetch_images: bool = True,
                uv_method: str = DEFAULT_UV_METHOD, jaw_uv=None):
    # The pixel methods decode the approach window a second time, which makes them the
    # expensive ones.
    """Mine one teleop dataset into an open shard writer, decoding frames only if `fetch_images`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    calibration = gripper_camera_calibration()
    episodes, fps = read_columns(root)

    dataset = LeRobotDataset(repo_id, root=root)
    starts = {
        int(r["episode_index"]): int(r["dataset_from_index"])
        for r in dataset.meta.episodes.select_columns(
            ["episode_index", "dataset_from_index"]).to_list()
    }
    if IMAGE_KEY not in dataset.meta.video_keys:
        raise ValueError(f"{repo_id} has no {IMAGE_KEY}; present: {dataset.meta.video_keys}")
    src_h, src_w = dataset.meta.features[IMAGE_KEY]["shape"][:2]
    if progress is not None:
        progress.set_description(f"{repo_id.split('/')[-1]} {src_w}x{src_h}")

    mined, skipped = 0, {"no_grasp": 0, "no_rise": 0, "no_range": 0}
    dropped_total, blind_total = 0, 0
    considered = 0
    # Report the holding label balance, since it is a judgement call.
    holding_counts = {1: 0, 0: 0, None: 0}
    for n, ep in enumerate(sorted(episodes)):
        if limit and n >= limit:
            break
        considered += 1
        if progress is not None:
            progress.update(1)
        if mode == MODE_NEGATIVES:
            result, info, blind = mine_negative_episode(episodes[ep], stride)
        elif mode == MODE_FALSE_GRABS:
            result, info, blind = mine_false_grab_episode(episodes[ep], stride)
        else:
            result, info, blind = mine_episode(
                episodes[ep], fps, calibration, approach_seconds, carry_seconds, rise_m,
                uv_method=uv_method, jaw_uv=jaw_uv,
                frames=(lambda i, base=starts[ep]: frame_bgr(dataset, base + i))
                if uv_method in PIXEL_METHODS else None)
        if result is None:
            skipped[info] += 1
            continue
        dropped_total += info
        blind_total += blind
        base = starts[ep]
        for sample in result:
            index = base + sample["frame_index"]
            if fetch_images:
                sample["image"] = encode_frame(frame_bgr(dataset, index), image_size)
            else:
                sample["_fetch"] = (repo_id, str(root), index)
            sample["episode_index"] = ep
            sample["source_repo_id"] = repo_id
            writer.add(sample)
            holding_counts[sample["holding"]] += 1
            mined += 1
        if progress is not None:
            progress.set_postfix(frames=writer.total, refresh=False)

    summary = (f"{repo_id}: mined {mined} frames from "
               f"{considered - sum(skipped.values())}/{considered} episodes "
               f"(skipped {skipped}), {dropped_total} frames dropped as off-canvas "
               f"or behind the camera, {blind_total} kept with no target in view "
               f"({blind_total / max(mined, 1) * 100:.0f}% of what was mined), "
               f"holding {holding_counts[1]} held / {holding_counts[0]} empty / "
               f"{holding_counts[None]} masked")
    if progress is not None:
        progress.write(summary)
    else:
        logging.info(summary)
    return mined


def mine(sources, output_root: Path, split: str, approach_seconds: float,
         carry_seconds: float, rise_m: float, limit: int | None,
         mode: str = MODE_GRASPS, stride: int = SWEEP_STRIDE,
         image_size=IMAGE_SIZE, uv_method: str = DEFAULT_UV_METHOD, jaw_uv=None):
    """Replace this producer's share of the pool with rows mined from the given (repo_id,
    root) sources."""
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Frames of two heights in one split fail only at the first batch, so check now.
    existing = split_image_size(split_dir)
    if existing is not None and tuple(existing) != tuple(image_size):
        raise SystemExit(
            f"{split_dir} already holds {existing[0]}x{existing[1]} frames and this run "
            f"would write {image_size[0]}x{image_size[1]}. Pass --image_size "
            f"{existing[0]} {existing[1]} to match it, or mine into a different "
            f"--output_root.")
    # Each mode is its own producer and replaces only what it wrote.
    prefix = shard_prefix(mode)
    for stale in split_dir.glob(f"{prefix}-*.parquet"):
        stale.unlink()

    from tqdm import tqdm

    total = sum(min(source_episode_count(root), limit or 1 << 30) for _, root in sources)
    writer = ShardWriter(split_dir, prefix=prefix)
    with tqdm(total=total, unit="ep", dynamic_ncols=True) as progress:
        for repo_id, root in sources:
            mine_source(writer, root, repo_id, approach_seconds, carry_seconds, rise_m,
                        limit, progress, mode=mode, stride=stride,
                        image_size=image_size, uv_method=uv_method, jaw_uv=jaw_uv)
    writer.flush()

    write_dataset_card(output_root)
    logging.info(f"{writer.total} rows in {writer.shards} shard(s) under {split_dir}")
    return writer.total, split_dir


def mine_preview(sources, approach_seconds: float, carry_seconds: float, rise_m: float,
                 limit: int | None, count: int, seed: int, mode: str = MODE_GRASPS,
                 stride: int = SWEEP_STRIDE, image_size=IMAGE_SIZE,
                 uv_method: str = DEFAULT_UV_METHOD, jaw_uv=None):
    """Label every frame a real run would, keep a random `count` of the rows, write nothing."""
    from tqdm import tqdm

    if uv_method in PIXEL_METHODS:
        logging.info(
            f"--uv_method {uv_method} reads the video to label it, so this run decodes the "
            f"approach window of every episode and --preview_only saves little. --limit is "
            f"the knob that still works.")
    total = sum(min(source_episode_count(root), limit or 1 << 30) for _, root in sources)
    sampler = ReservoirSampler(count, seed)
    with tqdm(total=total, unit="ep", dynamic_ncols=True) as progress:
        for repo_id, root in sources:
            mine_source(sampler, root, repo_id, approach_seconds, carry_seconds, rise_m,
                        limit, progress, mode=mode, stride=stride,
                        image_size=image_size, fetch_images=False,
                        uv_method=uv_method, jaw_uv=jaw_uv)

    logging.info(f"{sampler.total} rows would be written; previewing {len(sampler.rows)}")
    fetch_preview_images(sampler.rows, image_size)
    return sampler.rows


def fetch_preview_images(rows, image_size=IMAGE_SIZE):
    """Fill in `image` on sampled rows, decoding only the frames they name, in dataset order."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    by_source: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        repo_id, root, _ = row["_fetch"]
        by_source.setdefault((repo_id, root), []).append(row)

    for (repo_id, root), group in by_source.items():
        dataset = LeRobotDataset(repo_id, root=Path(root))
        for row in sorted(group, key=lambda r: r["_fetch"][2]):
            row["image"] = encode_frame(frame_bgr(dataset, row.pop("_fetch")[2]), image_size)


def write_dataset_card(output_root: Path):
    """The YAML header that makes the train and eval shards load as a hub dataset."""
    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for split, name in (("train", "train"), ("eval", "test")):
        if any((output_root / split).glob("*.parquet")):
            lines += [f"  - split: {name}", f"    path: {split}/*.parquet"]
    lines.append("---")
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def sample_labelled_rows(split_dir: Path, count: int, seed: int, prefix=None,
                         keep_unlabelled=False):
    """`count` random labelled rows with images, reading only the row groups they land in."""
    import pyarrow.parquet as pq

    label_columns = ["episode_index", "frame_index", "seconds_to_grasp", "target_uv",
                     "target_range_m", "grasp_axis_rad", "finger", "close_now",
                     "target_present", "holding", "state"]

    candidates = []
    for path in sorted(split_dir.glob(f"{prefix}-*.parquet" if prefix else "*.parquet")):
        table = pq.read_table(path, columns=["target_uv"])
        uv = table.column("target_uv").to_pylist()
        # Negative and false-grab rows have no position label, so don't require one.
        candidates += [(path, i) for i, value in enumerate(uv)
                       if value is not None or keep_unlabelled]

    chosen = random.Random(seed).sample(candidates, min(count, len(candidates)))

    by_file: dict[Path, list[int]] = {}
    for path, index in chosen:
        by_file.setdefault(path, []).append(index)

    rows = []
    for path, indices in by_file.items():
        reader = pq.ParquetFile(path)
        # row group boundaries, so only the groups holding a chosen row get read
        bounds, total = [], 0
        for g in range(reader.num_row_groups):
            bounds.append(total)
            total += reader.metadata.row_group(g).num_rows
        wanted: dict[int, list[int]] = {}
        for index in indices:
            g = max(i for i, start in enumerate(bounds) if start <= index)
            wanted.setdefault(g, []).append(index - bounds[g])
        for g, offsets in wanted.items():
            table = reader.read_row_group(g, columns=["image"] + label_columns)
            batch = table.to_pylist()
            rows += [batch[o] for o in offsets]
    return rows


def write_preview(split_dir: Path, preview_dir: Path, count: int, seed: int,
                  group: int = 20, columns: int = 4, prefix=None, keep_unlabelled=False):
    render_preview(sample_labelled_rows(split_dir, count, seed, prefix, keep_unlabelled),
                   preview_dir, group, columns)


def render_preview(chosen, preview_dir: Path, group: int = 20, columns: int = 4):
    """Draw labelled rows as annotated frames plus contact sheets, at twice their stored size."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in list(preview_dir.glob("*.jpg")) + list(preview_dir.glob("*.png")):
        old.unlink()

    annotated = []
    for sample in chosen:
        img = cv2.imdecode(np.frombuffer(sample["image"], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        has_target = sample["target_uv"] is not None
        u, v = ((sample["target_uv"][0] * w, sample["target_uv"][1] * h)
                if has_target else (w / 2, h / 2))
        theta = sample["grasp_axis_rad"] or 0.0

        # A canvas big enough for the whole -0.25..1.25 range, so off-edge targets show.
        pad_x, pad_y = int(w * 0.25), int(h * 0.25)
        canvas = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                    cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cx, cy = int(round(u + pad_x)), int(round(v + pad_y))
        cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + w, pad_y + h), (90, 90, 90), 1)

        # The bar is rotated by the grasp axis, showing the jaw line the operator used.
        if has_target:
            length = 40
            dx, dy = math.cos(theta) * length, math.sin(theta) * length
            cv2.line(canvas, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                     (0, 200, 255), 3)
            cv2.circle(canvas, (cx, cy), 14, (0, 255, 0), 2)
            cv2.drawMarker(canvas, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
        else:
            # Say why there is no crosshair: an empty-floor negative or a masked label.
            banner = "NOTHING HERE" if sample['target_present'] == 0 else "NO TARGET LABEL"
            cv2.putText(canvas, banner, (pad_x + 10, pad_y + h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(canvas, banner, (pad_x + 10, pad_y + h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 200, 255), 2)

        # Every label is nullable, so check before formatting.
        finger = ('none' if sample['finger'] is None else f"{sample['finger']:+.2f}")
        lines = [
            f"ep{sample['episode_index']} f{sample['frame_index']}" + (
                "  no grasp" if sample['seconds_to_grasp'] is None
                else f"  t-{sample['seconds_to_grasp']:.2f}s"),
            (f"uv {sample['target_uv'][0]:+.3f},{sample['target_uv'][1]:+.3f}  "
             f"range {sample['target_range_m']:.3f}m") if has_target
            else f"target_present {sample['target_present']}",
            (f"axis {math.degrees(theta):+.1f}deg  " if has_target else "")
            + f"finger {finger}  close {sample['close_now']}  holding {sample['holding']}",
            f"laser {sample['state']['laser_rangefinder']:.3f}  fingerang {sample['state']['finger_angle']:.1f}"
            f"  force {sample['state']['target_force']:.3f}",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 26
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        name = f"ep{sample['episode_index']:04d}_f{sample['frame_index']:05d}.jpg"
        cv2.imwrite(str(preview_dir / name), canvas)
        annotated.append(canvas)

    for start in range(0, len(annotated), group):
        cells = annotated[start:start + group]
        h, w = cells[0].shape[:2]
        blank = np.full((h, w, 3), 25, dtype=np.uint8)
        cells = cells + [blank] * (-len(cells) % columns)
        sheet = np.vstack([np.hstack(cells[r:r + columns]) for r in range(0, len(cells), columns)])
        name = f"_sheet_{start // group + 1:02d}.png"
        cv2.imwrite(str(preview_dir / name), sheet)

    logging.info(
        f"wrote {len(chosen)} preview frames and "
        f"{-(-len(annotated) // group)} contact sheets to {preview_dir}"
    )


def main():
    # force=True because importing lerobot/transformers installs a root handler.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, nargs="+",
                        help="Source teleop dataset(s). Several are mined into one split.")
    parser.add_argument("--root", default=None, nargs="+",
                        help="Their roots on disk, in the same order (defaults to the HF cache)")
    parser.add_argument("--check", action="store_true",
                        help="Say whether each --repo_id can be mined, and exit. Reads "
                             "metadata only, so it costs nothing against the hub.")
    parser.add_argument("--output_root", required=False,
                        help="Where the mined dataset is written; its split directory is replaced")
    parser.add_argument("--split", default=POOL_SPLIT, choices=[POOL_SPLIT, "train", "eval"],
                        help=f"Where the shards land. The default is the {POOL_SPLIT}/ pool, "
                             f"which split_pool deals into train and eval afterwards")
    parser.add_argument("--preview_dir", default=None, help="Write annotated sample frames here")
    parser.add_argument("--preview_only", action="store_true",
                        help="Label the sources as usual but write only the preview: no "
                             "frame is decoded except the ones it draws, and no shard is "
                             "written or replaced. The loop to iterate on labelling in - "
                             "pair it with --limit to cut it further. Needs --preview_dir "
                             "and ignores --output_root.")
    parser.add_argument("--preview_count", type=int, default=100)
    parser.add_argument("--preview_group", type=int, default=20,
                        help="Frames per contact sheet")
    parser.add_argument("--preview_seed", type=int, default=0)
    add_uv_arguments(parser)
    parser.add_argument("--approach_seconds", type=float, default=APPROACH_SECONDS)
    parser.add_argument("--carry_seconds", type=float, default=CARRY_SECONDS)
    parser.add_argument("--rise_m", type=float, default=RISE_M)
    parser.add_argument("--limit", type=int, default=None, help="Only mine this many episodes")
    what = parser.add_mutually_exclusive_group()
    what.add_argument(
        "--negatives", action="store_true",
        help="The source is a recording of empty floor, so every frame is a "
             "target_present=0 row with no position labels. Writes negative-*.parquet "
             "beside the positives rather than replacing them. Point it only at a "
             "recording that really is empty - nothing here can check that for you.")
    what.add_argument(
        "--false_grabs", action="store_true",
        help="The source is a recording in which closing the jaws would catch nothing, "
             "so every frame is a close_now=0, holding=0 row with every other label "
             "masked. Writes false_grab-*.parquet beside the positives. Point it only at "
             "a recording where the fingers really did stay empty throughout.")
    parser.add_argument("--negative_stride", "--false_grab_stride", dest="stride",
                        type=int, default=SWEEP_STRIDE,
                        help="With --negatives or --false_grabs, keep one frame in this many")
    parser.add_argument("--image_size", type=int, nargs=2, default=list(IMAGE_SIZE),
                        metavar=("WIDTH", "HEIGHT"),
                        help="Frame size to store, which has to suit the backbone the "
                             "dataset is for: 448 256 for DINOv3 at /16, 448 252 for "
                             "DINOv2 at /14. A split that already holds frames refuses a "
                             "run that disagrees with it. Default %(default)s.")
    args = parser.parse_args()

    roots = args.root or []
    if roots and len(roots) != len(args.repo_id):
        parser.error(f"got {len(args.repo_id)} --repo_id but {len(roots)} --root")

    if args.check:
        report_sources(roots or args.repo_id)
        return
    if args.preview_only and not args.preview_dir:
        parser.error("--preview_only needs --preview_dir to write to")
    if not (args.output_root or args.preview_only):
        parser.error("--output_root is required unless --check or --preview_only")

    sources = []
    for i, repo_id in enumerate(args.repo_id):
        if roots:
            sources.append((repo_id, Path(roots[i])))
        else:
            sources.append((repo_id, Path(hub_root(repo_id))))

    mode = (MODE_NEGATIVES if args.negatives
            else MODE_FALSE_GRABS if args.false_grabs
            else MODE_GRASPS)

    if args.preview_only:
        rows = mine_preview(
            sources, args.approach_seconds, args.carry_seconds, args.rise_m, args.limit,
            args.preview_count, args.preview_seed, mode=mode, stride=args.stride,
            image_size=tuple(args.image_size),
            uv_method=args.uv_method, jaw_uv=tuple(args.jaw_uv) if args.jaw_uv else None,
        )
        render_preview(rows, Path(args.preview_dir), args.preview_group)
        return

    total, split_dir = mine(
        sources, Path(args.output_root), args.split,
        args.approach_seconds, args.carry_seconds, args.rise_m, args.limit,
        mode=mode, stride=args.stride,
        image_size=tuple(args.image_size),
        uv_method=args.uv_method, jaw_uv=tuple(args.jaw_uv) if args.jaw_uv else None,
    )
    if args.preview_dir and total:
        write_preview(split_dir, Path(args.preview_dir),
                      args.preview_count, args.preview_seed, args.preview_group,
                      prefix=shard_prefix(mode), keep_unlabelled=mode != MODE_GRASPS)


if __name__ == "__main__":
    main()
