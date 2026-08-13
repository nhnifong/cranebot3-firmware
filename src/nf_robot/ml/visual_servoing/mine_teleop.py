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
point_in_camera for the mount, which is two sign flips once the gripper is assumed level.

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
    target_present   1 (mined frames always have one; negatives come from synthetic data)

After the grasp the object rides in the jaws, so the static room point stops describing
where it is. Those frames therefore carry only `holding` and `finger`, and every other
label is null - which the row format spells "mask this head's loss here" rather than
"the answer is zero".

Only successful grasps are mined. A grasp that closed on nothing puts the label
somewhere the object never was, which is worse than no label at all; the rise test in
lerobot_trim_to_grasp is exactly that success filter and is reused here.

Usage:
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

from scipy.spatial.transform import Rotation

import nf_robot.common.definitions as definitions
from nf_robot.common.config_loader import create_default_config
from nf_robot.ml.stringman_lerobot import rotate_vector
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

# The camera mount, moved from the CAD y-up gripper frame (grommet at +y, nose at -z)
# into the z-up body frame the rest of the system uses (pole up +z, nose at +y). Rx(90)
# is the same seam arp_gripper_client.measure_gantry_minus_card crosses.
_YUP_TO_ZUP = Rotation.from_euler("x", 90, degrees=True)
# Comes out as Rx(180 - 9.06 deg): looking down, tilted back away from the nose.
CAMERA_ROT_BODY = _YUP_TO_ZUP * Rotation.from_rotvec(definitions.gripper_camera[0])
# Comes out as (0, +0.027, +0.006): 2.7cm toward the nose, 6mm up from the body origin.
CAMERA_POS_BODY = _YUP_TO_ZUP.apply(definitions.gripper_camera[1])

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

    def __init__(self, split_dir: Path, target_bytes: int = SHARD_TARGET_BYTES,
                 prefix: str = "shard"):
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


def point_in_camera(point_room, gripper_pos, spin):
    """A room point in the gripper camera's optical frame, assuming the gripper hangs level.

    Step one is the rotated contact vector that lerobot_label_contact_actions already
    builds: the room-frame vector from the gripper to the target, turned into the gripper
    frame by rotating its horizontal part by `spin`. get_spin is a clockwise bearing, so
    gripper->room is a rotation by -spin and room->gripper is +spin, which is the
    direction rotate_vector takes here and there.

    Step two is the fixed camera mount, taken from definitions.gripper_camera rather
    than idealised: the lens sits 2.7cm toward the nose and 6mm up from the body origin,
    and looks 9.06 degrees back from straight down. Both matter at grasping range, where
    the object is only a few centimetres away and 2.7cm is a large part of the frame.

    Still ignored: any swing of the gripper away from vertical.
    """
    delta = np.asarray(point_room, dtype=np.float64) - np.asarray(gripper_pos, dtype=np.float64)
    horizontal = rotate_vector(delta[:2], spin)
    in_body = np.array([horizontal[0], horizontal[1], delta[2]])
    return CAMERA_ROT_BODY.inv().apply(in_body - CAMERA_POS_BODY)


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


def mine_episode(rows, fps, calibration, approach_seconds, carry_seconds, rise_m):
    """Labelled rows for one episode, or (None, reason) if it is not a usable grasp."""
    pressure = np.array([r["pressure"] for r in rows], dtype=np.float32)
    grasp = find_grasp(pressure, fps, PRESSURE_THRESHOLD, MIN_GRASP_SECONDS)
    if grasp is None:
        return None, "no_grasp"

    heights = np.array([r["gripper_pos"][2] for r in rows])
    if not np.any(heights[grasp:] >= heights[grasp] + rise_m):
        # closed on nothing, or on something it could not pick up
        return None, "no_rise"

    target_room = grasp_point_room(rows[grasp])
    wrist_at_grasp = rows[grasp]["wrist_angle"]

    first = max(0, grasp - int(round(approach_seconds * fps)))
    last = min(len(rows) - 1, grasp + int(round(carry_seconds * fps)))

    out, dropped = [], 0
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
            sample["target_uv"] = [round(u, 5), round(v, 5)]
            sample["target_range_m"] = round(distance, 4)
            sample["grasp_axis_rad"] = round(
                wrap_pi(math.radians(wrist_at_grasp - r["wrist_angle"])), 5)

        out.append(sample)
    return out, dropped


def mine_source(writer: ShardWriter, root: Path, repo_id: str, approach_seconds: float,
                carry_seconds: float, rise_m: float, limit: int | None):
    """Mine one teleop dataset into an open shard writer."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    calibration = gripper_camera_calibration()
    episodes, fps = read_columns(root)
    logging.info(f"{root}: {len(episodes)} episodes at {fps} fps")

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
    logging.info(f"{repo_id}: {image_key} at {src_w}x{src_h}")

    mined, skipped = 0, {"no_grasp": 0, "no_rise": 0}
    dropped_total = 0
    considered = 0
    for n, ep in enumerate(sorted(episodes)):
        if limit and n >= limit:
            break
        considered += 1
        result, info = mine_episode(episodes[ep], fps, calibration,
                                    approach_seconds, carry_seconds, rise_m)
        if result is None:
            skipped[info] += 1
            continue
        dropped_total += info
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
        logging.info(f"episode {ep}: {len(result)} frames")

    logging.info(
        f"{repo_id}: mined {mined} frames from {considered - sum(skipped.values())}"
        f"/{considered} episodes (skipped {skipped}), "
        f"{dropped_total} frames dropped as off-canvas or behind the camera"
    )
    return mined


def mine(sources, output_root: Path, split: str, approach_seconds: float,
         carry_seconds: float, rise_m: float, limit: int | None):
    """Replace one split of the mined dataset with the given (repo_id, root) sources.

    The split directory is emptied first. Mining is deterministic given its inputs, so
    a rerun should leave no trace of the previous one - appending instead is how a
    dataset ends up with rows for frames that are no longer produced, or two rows for
    the same frame.
    """
    split_dir = output_root / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)

    writer = ShardWriter(split_dir)
    for repo_id, root in sources:
        mine_source(writer, root, repo_id, approach_seconds, carry_seconds, rise_m, limit)
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


def sample_labelled_rows(split_dir: Path, count: int, seed: int):
    """`count` random labelled rows, images included, read back out of the shards.

    Two passes so that previewing a large dataset does not mean reading it: the first
    reads only the label columns to find candidates, the second pulls just the row
    groups the chosen rows landed in.
    """
    import pyarrow.parquet as pq

    label_columns = ["episode_index", "frame_index", "seconds_to_grasp", "target_uv",
                     "target_range_m", "grasp_axis_rad", "finger", "holding", "state"]

    candidates = []
    for path in sorted(split_dir.glob("*.parquet")):
        table = pq.read_table(path, columns=["target_uv"])
        uv = table.column("target_uv").to_pylist()
        candidates += [(path, i) for i, value in enumerate(uv) if value is not None]

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
                  group: int = 20, columns: int = 4):
    """A folder of annotated frames plus contact sheets, for eyeballing the labels.

    A sign error in the projection produces perfectly plausible numbers and an obviously
    wrong crosshair, so this is the check that actually catches things. The frames are
    drawn at twice their stored size, and the sheets do not shrink them to fit, because
    text scaled down to fit a grid cell cannot be read - which defeats the point.
    """
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in list(preview_dir.glob("*.jpg")) + list(preview_dir.glob("*.png")):
        old.unlink()

    chosen = sample_labelled_rows(split_dir, count, seed)

    annotated = []
    for sample in chosen:
        img = cv2.imdecode(np.frombuffer(sample["image"], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        u, v = sample["target_uv"][0] * w, sample["target_uv"][1] * h
        theta = sample["grasp_axis_rad"]

        # Draw on a canvas big enough to hold the whole -0.25..1.25 range, so a target
        # off the edge is visible instead of silently clipped away.
        pad_x, pad_y = int(w * 0.25), int(h * 0.25)
        canvas = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x,
                                    cv2.BORDER_CONSTANT, value=(40, 40, 40))
        cx, cy = int(round(u + pad_x)), int(round(v + pad_y))
        cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + w, pad_y + h), (90, 90, 90), 1)

        # The grasp axis is how much further the wrist turns before the grasp, so the bar
        # is drawn rotated by it: it shows the jaw line the operator ended up using.
        length = 40
        dx, dy = math.cos(theta) * length, math.sin(theta) * length
        cv2.line(canvas, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                 (0, 200, 255), 3)
        cv2.circle(canvas, (cx, cy), 14, (0, 255, 0), 2)
        cv2.drawMarker(canvas, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 26, 2)

        lines = [
            f"ep{sample['episode_index']} f{sample['frame_index']}  t-{sample['seconds_to_grasp']:.2f}s",
            f"uv {sample['target_uv'][0]:+.3f},{sample['target_uv'][1]:+.3f}  range {sample['target_range_m']:.3f}m",
            f"axis {math.degrees(theta):+.1f}deg  finger {sample['finger']:+.2f}  holding {sample['holding']}",
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
    parser.add_argument("--output_root", required=True,
                        help="Where the mined dataset is written; its split directory is replaced")
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    parser.add_argument("--preview_dir", default=None, help="Write annotated sample frames here")
    parser.add_argument("--preview_count", type=int, default=20)
    parser.add_argument("--preview_group", type=int, default=20,
                        help="Frames per contact sheet")
    parser.add_argument("--preview_seed", type=int, default=0)
    parser.add_argument("--approach_seconds", type=float, default=APPROACH_SECONDS)
    parser.add_argument("--carry_seconds", type=float, default=CARRY_SECONDS)
    parser.add_argument("--rise_m", type=float, default=RISE_M)
    parser.add_argument("--limit", type=int, default=None, help="Only mine this many episodes")
    args = parser.parse_args()

    roots = args.root or []
    if roots and len(roots) != len(args.repo_id):
        parser.error(f"got {len(args.repo_id)} --repo_id but {len(roots)} --root")

    sources = []
    for i, repo_id in enumerate(args.repo_id):
        if roots:
            sources.append((repo_id, Path(roots[i])))
        else:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            sources.append((repo_id, Path(LeRobotDataset(repo_id).root)))

    total, split_dir = mine(
        sources, Path(args.output_root), args.split,
        args.approach_seconds, args.carry_seconds, args.rise_m, args.limit,
    )
    if args.preview_dir and total:
        write_preview(split_dir, Path(args.preview_dir),
                      args.preview_count, args.preview_seed, args.preview_group)


if __name__ == "__main__":
    main()
