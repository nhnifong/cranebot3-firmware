#!/usr/bin/env python

"""Mine visual-servoing labels out of teleop recordings.

The trick is to run time backwards. At the moment of the grasp we know exactly where
the object was - directly under the gripper camera, at the distance the rangefinder was
reading. Projecting that one room point back into the camera for each of the preceding
frames labels a whole approach with no object detector involved, and labels it with the
spot a human chose to grab rather than an object's visual centroid. On a towel those are
different places and only the first one is a good grasp.

Note that the anchor point is *not* the recorded gripper_pos. That is the gripper body
origin, which sits about a centimetre from the camera, so projecting it produces a point
effectively inside the lens and a garbage pixel. The object is a jaw-length further down,
so the anchor is the gripper position at the grasp dropped straight down by whatever the
rangefinder was reading.

Getting that point into the frame is the rotated contact vector that
lerobot_label_contact_actions already builds - the room-frame vector from the gripper to
the target, with its horizontal part rotated by `spin` into the gripper frame - followed
by the fixed camera mount and config.camera_cal_wide's 684x384 intrinsics. See
geometry.point_in_camera for the mount, which is two sign flips once the gripper is
assumed level; the robot's inverse of it lives beside it, so the two cannot drift apart.

The camera's 9.06 degree backward tilt and its 2.7cm offset toward the nose are both
accounted for. Swing of the gripper away from vertical is still ignored - the recorded
6D rotation carries it, but nothing here reads it yet.

Output is parquet shards of a few hundred MB, one row per frame with the frame itself in
an `image` column as JPEG bytes at the model's input resolution. One file per frame is
the obvious layout and the wrong one: mining the full source list is on the order of
400k frames, and the hub charges per file, not per byte. Each run replaces its split
directory outright.

What comes out, per frame, in the row format readme.md describes:

    target_uv        the grasp point in normalized frame coordinates, where 0..1 is the
                     visible frame and the kept range is -0.25..1.25 - so a target just
                     off the bottom edge keeps a real position instead of being clamped
                     to the edge or thrown away. That case is the whole point.
    target_range_m   distance from the camera to the grasp point, the third dimension
    grasp_axis_rad   how much further the wrist turned before grasping, pi-wrapped
    finger           commanded finger speed / 90, in -1..1
    holding          0 before the grasp, 1 after it
    target_present   1 (an approach always has one)

With --negatives the source is instead a recording of flying over empty floor, and every
frame becomes a target_present=0 row with no position labels at all. That mode exists
because a head trained only on synthetic negatives learns "nothing here" as a property of
composited images: measured on a real checkpoint, it fires low on half of the synthetic
bare-floor frames and never once on a real one. Negatives have to arrive through the same
camera and the same pipeline as the positives to mean anything at deploy time.

After the grasp the object rides in the jaws, so the static room point stops describing
where it is. Those frames therefore carry only `holding` and `finger`, and every other
label is null - which the row format spells "mask this head's loss here" rather than
"the answer is zero".

Only successful grasps are mined. A grasp that closed on nothing puts the label
somewhere the object never was, which is worse than no label at all; the rise test in
lerobot_trim_to_grasp is exactly that success filter and is reused here.

Usage:
    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/empty-floor-sweep --negatives \
        --output_root datasets/visual_servoing

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/bedroom-laundry-aug7-2 \
        --root datasets/bedroom-laundry-aug7-2 \
        --output_root data/visual_servoing \
        --preview_dir data/visual_servoing/preview
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

from nf_robot.common.config_loader import create_default_config
from nf_robot.ml.stringman_lerobot import rotate_vector
from nf_robot.ml.visual_servoing.geometry import CAMERA_POS_BODY, point_in_camera
from nf_robot.ml.lerobot_trim_to_grasp import (
    MIN_GRASP_SECONDS,
    PRESSURE_THRESHOLD,
    RISE_M,
    find_grasp,
)

# (seconds) how much of the approach before the grasp to label. Long enough to cover the
# final descent and the corrections in it, short enough that the object is plausibly the
# one being approached rather than whatever the gripper happened to fly over earlier.
APPROACH_SECONDS = 10.0
# (seconds) how much of the carry after the grasp to keep, for the holding head.
CARRY_SECONDS = 3.0
# The canvas the target head predicts over, as a fraction of the frame. 1.5 means
# coordinates run -0.25..1.25 and a target a quarter-frame off the edge still has a cell.
CANVAS_SCALE = 1.5
# How far outside the visible frame a target may be and still be worth predicting, as a
# fraction of the frame. A tenth is about 45px of the 448 wide input.
#
# A little way out is the case the oversized canvas exists for: the object is still in
# shot, only the spot to grab it by has slipped past the edge, and "down there, just off
# the bottom" is an answer the image supports. Further out there is nothing in the picture
# to point at, and asking for a position anyway trains the head to invent one from
# whatever the floor happens to look like. Those rows keep their frame and lose their
# position labels.
#
# Measured on the combined_targets eval split, this masks 15% of the labelled rows; 0.05
# would mask 20% and 0.20 only 4%, with the off-screen ones spread evenly out to the
# canvas edge at 0.25.
OFF_SCREEN_MARGIN = 0.10
# What negative shards are called, so mining an empty-floor recording into a split that
# already holds positives replaces only its own output.
NEGATIVE_PREFIX = "negative"
# Keep one frame in this many when mining negatives. A recording of flying over empty
# floor is negative in every frame, and at 30fps thirty of them a second are the same
# picture; six a second is plenty of variety and keeps an hour of flying from burying the
# positives it is meant to balance.
NEGATIVE_STRIDE = 5
# (metres) minimum distance in front of the camera for a projection to mean anything.
MIN_DEPTH_M = 0.02
# Full-scale commanded finger speed, used to normalize the finger label into -1..1.
FINGER_SPEED_FULL_SCALE = 90.0
# Frames are stored at the model's input resolution. Labels are normalized coordinates,
# so they survive the resize untouched, and the stored frame is then exactly what the
# model sees - nothing is gained by keeping pixels the training loader would throw away.
IMAGE_SIZE = (448, 256)
JPEG_QUALITY = 90
# Roughly how much image data goes in one parquet shard. The point of shards is file
# count: a few hundred large files upload and download from the hub in a way that
# hundreds of thousands of small ones do not.
SHARD_TARGET_BYTES = 512 * 1024 * 1024
# Rows per parquet row group. Small groups let the preview pull a handful of scattered
# images back without reading whole shards.
ROW_GROUP_SIZE = 256

STATE_NEEDED = (
    "gripper_pos_x", "gripper_pos_y", "gripper_pos_z",
    "spin", "finger_pressure", "wrist_angle", "finger_angle",
    "laser_rangefinder", "target_force",
)


def row_schema():
    """Parquet schema for one labelled frame.

    Mirrors the row format in readme.md: every label is nullable, and null means "mask
    this head's loss for this row" rather than "the answer is zero". The frame travels
    in the row as JPEG bytes, which is what keeps the file count down without a second
    copy of the pixels living outside the table.
    """
    import pyarrow as pa

    return pa.schema([
        ("image", pa.binary()),
        ("split_source", pa.string()),
        ("source_repo_id", pa.string()),
        ("episode_index", pa.int32()),
        ("frame_index", pa.int32()),
        ("seconds_to_grasp", pa.float32()),
        # a plain list rather than a fixed-size one: parquet cannot store a null in a
        # fixed-size list, and null is exactly what an unlabelled row needs here
        ("target_uv", pa.list_(pa.float32())),
        ("target_range_m", pa.float32()),
        ("grasp_axis_rad", pa.float32()),
        ("finger", pa.float32()),
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

    # What the miner's own shards are called. The compositor passes its own prefix, and
    # each producer only ever deletes files carrying its own.
    DEFAULT_PREFIX = "shard"

    def __init__(self, split_dir: Path, target_bytes: int = SHARD_TARGET_BYTES,
                 prefix: str = DEFAULT_PREFIX):
        self.split_dir = split_dir
        # Shards are named by producer so the synthetic compositor can write into the
        # same split as the miner without either overwriting the other's files.
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


def encode_frame(bgr):
    """One frame as JPEG bytes at the model's input resolution."""
    resized = cv2.resize(bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def gripper_camera_calibration():
    """Gripper camera intrinsics as fractions of the frame: (fx, fy), (cx, cy).

    The wide calibration rather than camera_cal: the gripper streams the full-sensor
    16:9 field of view, which is what camera_cal_wide was chessboard-calibrated for.

    Normalized, because that makes the labels independent of what resolution the frames
    happen to be stored at - a resize moves every pixel coordinate and leaves every
    normalized one alone. A *crop* does not, which is why the recipe that builds the
    source dataset sets center_crop and pad_clamp false.
    """
    cal = create_default_config().camera_cal_wide
    K = np.array(cal.intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    width, height = cal.resolution.width, cal.resolution.height
    return (K[0, 0] / width, K[1, 1] / height), (K[0, 2] / width, K[1, 2] / height)


def read_columns(root: Path):
    """Per-episode state and action rows, straight from the parquets.

    Read as columns rather than through LeRobotDataset because this pass wants a handful
    of components for every frame and none of the video; decoding a frame per row to
    find the grasps would dominate the runtime.
    """
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
            })
    for rows in episodes.values():
        rows.sort(key=lambda r: r["frame_index"])
    return episodes, float(info["fps"])


def project(point_room, gripper_pos, spin, calibration):
    """A room point as normalized (u, v) in the gripper camera, plus its distance.

    0..1 spans the visible frame whatever resolution it is stored at. Returns None for
    anything at or behind the lens, where the projection is meaningless but still
    numerically produces a plausible looking coordinate.

    Pinhole only, no distortion: the wide calibration's coefficients are small (k1 is
    -0.026) next to the approximations above, and the distortion polynomial diverges
    wildly outside the field of view - which is exactly where this has to stay sane,
    since the whole point is labelling targets past the frame edge.
    """
    (fx, fy), (cx, cy) = calibration
    p_cam = point_in_camera(point_room, gripper_pos, spin)
    if p_cam[2] < MIN_DEPTH_M:
        return None
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return float(u), float(v), float(np.linalg.norm(p_cam))


def grasp_point_room(row):
    """Where the object was, in the room, at the instant of the grasp.

    Straight down from the *rangefinder* by whatever it read. The rangefinder sits
    beside the lens (see the measure_hover comment in observer.py), so the drop hangs
    from the camera position, not from the recorded gripper position - those differ by
    the 2.7cm the camera sits toward the nose, which is a systematic error in every
    label if it is charged to the wrong point.

    Down rather than along the optical axis because the beam points down the body axis;
    the lens is what is tilted, not the sensor.
    """
    body_offset = np.asarray(CAMERA_POS_BODY, dtype=np.float64)
    # body -> room is a rotation by -spin, the inverse of the room -> body above
    horizontal = rotate_vector(body_offset[:2], -float(row["spin"]))
    camera_room = np.asarray(row["gripper_pos"], dtype=np.float64) + np.array(
        [horizontal[0], horizontal[1], body_offset[2]])
    return camera_room + np.array([0.0, 0.0, -float(row["laser_rangefinder"])])


def wrap_pi(radians):
    """Fold an angle into [-pi/2, pi/2), the range a pi-periodic grasp axis lives in."""
    return (radians + math.pi / 2) % math.pi - math.pi / 2


def in_view(u, v, margin=OFF_SCREEN_MARGIN):
    """Whether a projected target is close enough to the frame to be worth predicting."""
    return -margin <= u <= 1 + margin and -margin <= v <= 1 + margin


def mine_episode(rows, fps, calibration, approach_seconds, carry_seconds, rise_m,
                 margin=OFF_SCREEN_MARGIN):
    """Labelled rows for one episode, or (None, reason, 0) if it is not a usable grasp.

    Returns (rows, dropped, blind): dropped fell off the canvas entirely, blind kept their
    frame but lost their position labels for being too far outside it to see.
    """
    pressure = np.array([r["pressure"] for r in rows], dtype=np.float32)
    grasp = find_grasp(pressure, fps, PRESSURE_THRESHOLD, MIN_GRASP_SECONDS)
    if grasp is None:
        return None, "no_grasp", 0

    heights = np.array([r["gripper_pos"][2] for r in rows])
    if not np.any(heights[grasp:] >= heights[grasp] + rise_m):
        # closed on nothing, or on something it could not pick up
        return None, "no_rise", 0

    target_room = grasp_point_room(rows[grasp])
    wrist_at_grasp = rows[grasp]["wrist_angle"]

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
            "target_present": 1 if i <= grasp else None,
            "holding": 0 if i < grasp else 1,
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "target_force": round(float(r["target_force"]), 4),
            },
        }

        # Only up to the grasp. After it the object rides in the jaws and the static room
        # point no longer says where it is.
        if i <= grasp:
            projected = project(target_room, r["gripper_pos"], r["spin"], calibration)
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
                # The frame is kept and its position labels are not: nothing in it shows
                # where this object is, so every position head is masked here.
                #
                # target_present is masked rather than set to 0. All that is known is that
                # the object being approached is out of shot - not that the picture is
                # empty, and in a room with laundry over the floor it usually is not.
                # Teaching "nothing here" off that would be a lie the model can see
                # through. The honest negatives are the synthetic bare-floor frames, which
                # are empty by construction.
                blind += 1
                sample["target_present"] = None

        out.append(sample)
    return out, dropped, blind


def mine_negative_episode(rows, fps, stride=NEGATIVE_STRIDE, rise_m=RISE_M):
    """Rows for one episode of an empty-floor recording, or (None, reason, 0).

    The mirror image of mine_episode: no grasp to run time backwards from, so nothing is
    labelled about *where* anything is - only that there was nothing there to go to.
    Every frame qualifies, which is the point, and the stride is what stops an hour of
    flying from contributing a hundred thousand near-identical rows.

    What each row carries, and what it deliberately does not:

        target_present  0, the label this whole mode exists to produce
        target_uv       null, along with range and axis. "Nothing is there" says nothing
        target_range_m  about where it would have been, and a zero would be a position
        grasp_axis_rad  claim rather than an absence of one
        finger          the recorded finger speed, same as any mined row. An operator
                        flying over bare floor is commanding no grip, which is exactly
                        what the finger head should answer here
        holding         0 while the pressure says the hand is empty, null if it is not -
                        an operator who picked something up mid-recording is no longer
                        describing empty floor, and guessing would be worse than masking

    Refuses an episode that contains a grasp. Pointing this mode at an ordinary grasping
    dataset would label every frame of every approach "nothing here", which is not a
    mislabelled row but a poisoned head, and it is an easy mistake to make from the
    command line. The test is the same held-pressure-then-lift the positive path uses to
    decide a grasp succeeded, so it costs nothing and catches exactly that.
    """
    pressure = np.array([r["pressure"] for r in rows], dtype=np.float32)
    grasp = find_grasp(pressure, fps, PRESSURE_THRESHOLD, MIN_GRASP_SECONDS)
    if grasp is not None:
        heights = np.array([r["gripper_pos"][2] for r in rows])
        if np.any(heights[grasp:] >= heights[grasp] + rise_m):
            return None, "has_grasp", 0

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
            "target_present": 0,
            "holding": 0 if empty else None,
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "target_force": round(float(r["target_force"]), 4),
            },
        })
    return out, 0, 0


def source_episode_count(root: Path) -> int:
    """Episodes in a source, read from its metadata."""
    return int(json.loads((root / "meta" / "info.json").read_text())["total_episodes"])


IMAGE_KEY = "observation.images.gripper_camera"


def check_source(source):
    """Whether a teleop dataset can be mined, from its metadata alone.

    Worth having as its own thing because the answer is a property of how the robot was
    recorded, not of the frames: a dataset that never logged the rangefinder cannot be
    mined however many good grasps are in it, and finding that out by downloading a few
    hundred GB of video first is the expensive way round.

    `source` is a directory or a hub dataset repo id. Returns a dict; `ok` is whether
    every requirement is met and `missing` says which are not.
    """
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
    """A hub dataset's local root, downloading it if it is not already there.

    Wrapped for the error: lerobot resolves a dataset by a git tag named after the
    codebase_version in its meta/info.json, and a repo published by a plain folder upload
    has no such tag. What comes back then is a TypeError raised while raising
    RevisionNotFoundError, naming neither the repo nor the tag - so say it here instead.
    """
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


def mine_source(writer: ShardWriter, root: Path, repo_id: str, approach_seconds: float,
                carry_seconds: float, rise_m: float, limit: int | None, progress=None,
                negatives: bool = False, stride: int = NEGATIVE_STRIDE):
    """Mine one teleop dataset into an open shard writer.

    `negatives` treats the whole recording as empty floor: see mine_negative_episode.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    calibration = gripper_camera_calibration()
    episodes, fps = read_columns(root)

    dataset = LeRobotDataset(repo_id, root=root)
    starts = {
        int(r["episode_index"]): int(r["dataset_from_index"])
        for r in dataset.meta.episodes.select_columns(
            ["episode_index", "dataset_from_index"]).to_list()
    }
    image_key = "observation.images.gripper_camera"
    if image_key not in dataset.meta.video_keys:
        raise ValueError(f"{repo_id} has no {image_key}; present: {dataset.meta.video_keys}")
    src_h, src_w = dataset.meta.features[image_key]["shape"][:2]
    if progress is not None:
        progress.set_description(f"{repo_id.split('/')[-1]} {src_w}x{src_h}")

    mined, skipped = 0, {"no_grasp": 0, "no_rise": 0, "has_grasp": 0}
    dropped_total, blind_total = 0, 0
    considered = 0
    for n, ep in enumerate(sorted(episodes)):
        if limit and n >= limit:
            break
        considered += 1
        if progress is not None:
            progress.update(1)
        if negatives:
            result, info, blind = mine_negative_episode(episodes[ep], fps, stride, rise_m)
        else:
            result, info, blind = mine_episode(episodes[ep], fps, calibration,
                                               approach_seconds, carry_seconds, rise_m)
        if result is None:
            skipped[info] += 1
            continue
        dropped_total += info
        blind_total += blind
        base = starts[ep]
        for sample in result:
            frame = dataset[base + sample["frame_index"]][image_key]
            bgr = cv2.cvtColor(
                (frame.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
            sample["image"] = encode_frame(bgr)
            sample["episode_index"] = ep
            sample["source_repo_id"] = repo_id
            writer.add(sample)
            mined += 1
        if progress is not None:
            progress.set_postfix(frames=writer.total, refresh=False)

    summary = (f"{repo_id}: mined {mined} frames from "
               f"{considered - sum(skipped.values())}/{considered} episodes "
               f"(skipped {skipped}), {dropped_total} frames dropped as off-canvas "
               f"or behind the camera, {blind_total} kept with no target in view "
               f"({blind_total / max(mined, 1) * 100:.0f}% of what was mined)")
    if progress is not None:
        progress.write(summary)
    else:
        logging.info(summary)
    return mined


def mine(sources, output_root: Path, split: str, approach_seconds: float,
         carry_seconds: float, rise_m: float, limit: int | None,
         negatives: bool = False, stride: int = NEGATIVE_STRIDE):
    """Replace one split of the mined dataset with the given (repo_id, root) sources.

    Only this producer's shards go: mining is deterministic given its inputs, so a rerun
    should leave no trace of the previous one - appending instead is how a dataset ends up
    with rows for frames that are no longer produced, or two rows for the same frame. The
    synthetic compositor writes its own shards into this same split and they are not ours
    to delete, which emptying the whole directory used to do silently.
    """
    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    # Negatives are their own producer, written beside the positives rather than over
    # them: a split wants both, and each rerun should replace only what it wrote.
    prefix = NEGATIVE_PREFIX if negatives else ShardWriter.DEFAULT_PREFIX
    for stale in split_dir.glob(f"{prefix}-*.parquet"):
        stale.unlink()

    from tqdm import tqdm

    total = sum(min(source_episode_count(root), limit or 1 << 30) for _, root in sources)
    writer = ShardWriter(split_dir, prefix=prefix)
    with tqdm(total=total, unit="ep", dynamic_ncols=True) as progress:
        for repo_id, root in sources:
            mine_source(writer, root, repo_id, approach_seconds, carry_seconds, rise_m,
                        limit, progress, negatives=negatives, stride=stride)
    writer.flush()

    write_dataset_card(output_root)
    logging.info(f"{writer.total} rows in {writer.shards} shard(s) under {split_dir}")
    return writer.total, split_dir


def write_dataset_card(output_root: Path):
    """The YAML header that makes the parquet files load as a hub dataset."""
    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for split, name in (("train", "train"), ("eval", "test")):
        if any((output_root / split).glob("*.parquet")):
            lines += [f"  - split: {name}", f"    path: {split}/*.parquet"]
    lines.append("---")
    (output_root / "README.md").write_text("\n".join(lines) + "\n")


def sample_labelled_rows(split_dir: Path, count: int, seed: int, prefix=None,
                         negatives=False):
    """`count` random labelled rows, images included, read back out of the shards.

    Two passes so that previewing a large dataset does not mean reading it: the first
    reads only the label columns to find candidates, the second pulls just the row
    groups the chosen rows landed in.
    """
    import pyarrow.parquet as pq

    label_columns = ["episode_index", "frame_index", "seconds_to_grasp", "target_uv",
                     "target_range_m", "grasp_axis_rad", "finger", "holding", "state"]

    candidates = []
    for path in sorted(split_dir.glob(f"{prefix}-*.parquet" if prefix else "*.parquet")):
        table = pq.read_table(path, columns=["target_uv"])
        uv = table.column("target_uv").to_pylist()
        # A negative row has no position label by construction, so requiring one would
        # preview nothing at all - and "is this really empty floor" is the check that
        # matters most for a mode whose whole job is to assert emptiness.
        candidates += [(path, i) for i, value in enumerate(uv)
                       if value is not None or negatives]

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
                  group: int = 20, columns: int = 4, prefix=None, negatives=False):
    """A folder of annotated frames plus contact sheets, for eyeballing the labels.

    A sign error in the projection produces perfectly plausible numbers and an obviously
    wrong crosshair, so this is the check that actually catches things. The frames are
    drawn at twice their stored size, and the sheets do not shrink them to fit, because
    text scaled down to fit a grid cell cannot be read - which defeats the point.
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in list(preview_dir.glob("*.jpg")) + list(preview_dir.glob("*.png")):
        old.unlink()

    chosen = sample_labelled_rows(split_dir, count, seed, prefix, negatives)

    annotated = []
    for sample in chosen:
        img = cv2.imdecode(np.frombuffer(sample["image"], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        has_target = sample["target_uv"] is not None
        u, v = ((sample["target_uv"][0] * w, sample["target_uv"][1] * h)
                if has_target else (w / 2, h / 2))
        theta = sample["grasp_axis_rad"] or 0.0

        # Draw on a canvas big enough to hold the whole -0.25..1.25 range, so a target
        # off the edge is visible instead of silently clipped away.
        pad_x, pad_y = int(w * 0.25), int(h * 0.25)
        canvas = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                    cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cx, cy = int(round(u + pad_x)), int(round(v + pad_y))
        cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + w, pad_y + h), (90, 90, 90), 1)

        # The grasp axis is how much further the wrist turns before the grasp, so the bar
        # is drawn rotated by it: it shows the jaw line the operator ended up using.
        if has_target:
            length = 40
            dx, dy = math.cos(theta) * length, math.sin(theta) * length
            cv2.line(canvas, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                     (0, 200, 255), 3)
            cv2.circle(canvas, (cx, cy), 14, (0, 255, 0), 2)
            cv2.drawMarker(canvas, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
        else:
            # No crosshair to draw, and saying so beats an unmarked frame that could just
            # as easily be a preview bug.
            cv2.putText(canvas, "NOTHING HERE", (pad_x + 10, pad_y + h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(canvas, "NOTHING HERE", (pad_x + 10, pad_y + h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 200, 255), 2)

        lines = [
            f"ep{sample['episode_index']} f{sample['frame_index']}" + (
                "  no grasp" if sample['seconds_to_grasp'] is None
                else f"  t-{sample['seconds_to_grasp']:.2f}s"),
            (f"uv {sample['target_uv'][0]:+.3f},{sample['target_uv'][1]:+.3f}  "
             f"range {sample['target_range_m']:.3f}m") if has_target else "target_present 0",
            (f"axis {math.degrees(theta):+.1f}deg  " if has_target else "")
            + f"finger {sample['finger']:+.2f}  holding {sample['holding']}",
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
    # force=True: importing lerobot/transformers installs a root handler, which makes a
    # later basicConfig a silent no-op and drops every info line this tool logs.
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
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    parser.add_argument("--preview_dir", default=None, help="Write annotated sample frames here")
    parser.add_argument("--preview_count", type=int, default=100)
    parser.add_argument("--preview_group", type=int, default=20,
                        help="Frames per contact sheet")
    parser.add_argument("--preview_seed", type=int, default=0)
    parser.add_argument("--approach_seconds", type=float, default=APPROACH_SECONDS)
    parser.add_argument("--carry_seconds", type=float, default=CARRY_SECONDS)
    parser.add_argument("--rise_m", type=float, default=RISE_M)
    parser.add_argument("--limit", type=int, default=None, help="Only mine this many episodes")
    parser.add_argument(
        "--negatives", action="store_true",
        help="The source is a recording of empty floor, so every frame is a "
             "target_present=0 row with no position labels. Writes negative-*.parquet "
             "beside the positives rather than replacing them. Episodes that turn out to "
             "contain a successful grasp are skipped, since one of those mined this way "
             "would teach that an object in the jaws is nothing at all.")
    parser.add_argument("--negative_stride", type=int, default=NEGATIVE_STRIDE,
                        help="With --negatives, keep one frame in this many")
    args = parser.parse_args()

    roots = args.root or []
    if roots and len(roots) != len(args.repo_id):
        parser.error(f"got {len(args.repo_id)} --repo_id but {len(roots)} --root")

    if args.check:
        report_sources(roots or args.repo_id)
        return
    if not args.output_root:
        parser.error("--output_root is required unless --check")

    sources = []
    for i, repo_id in enumerate(args.repo_id):
        if roots:
            sources.append((repo_id, Path(roots[i])))
        else:
            sources.append((repo_id, Path(hub_root(repo_id))))

    total, split_dir = mine(
        sources, Path(args.output_root), args.split,
        args.approach_seconds, args.carry_seconds, args.rise_m, args.limit,
        negatives=args.negatives, stride=args.negative_stride,
    )
    if args.preview_dir and total:
        write_preview(split_dir, Path(args.preview_dir),
                      args.preview_count, args.preview_seed, args.preview_group,
                      prefix=NEGATIVE_PREFIX if args.negatives else ShardWriter.DEFAULT_PREFIX,
                      negatives=args.negatives)


if __name__ == "__main__":
    main()
