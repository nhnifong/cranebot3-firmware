#!/usr/bin/env python

"""Extract placer training data from teleop recordings, in the formats dataset.md defines.

Usage:
    python -m nf_robot.ml.placer.mine_teleop --repo_id naavox/nick-sep14 \
        --output_root datasets/drop_pairs --preview_dir datasets/drop_pairs/previews

    python -m nf_robot.ml.placer.mine_teleop --mode carry --repo_id naavox/nick-sep14 \
        --preview_only --preview_dir placer_previews --preview_episodes 12
"""

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.lerobot.trim_to_grasp import MIN_GRASP_SECONDS, PRESSURE_THRESHOLD, find_grasp
from nf_robot.ml.ortho_target.model import ORTHO_EXTENT_M, room_to_ortho_px
from nf_robot.ml.visual_servoing.geometry import rotate_about_vertical
from nf_robot.ml.visual_servoing.mine_teleop import (
    FINGER_SPEED_FULL_SCALE, IMAGE_SIZE, JPEG_QUALITY, POOL_SPLIT, ShardWriter, close_onset,
    find_lift, read_columns)
from nf_robot.ml.visual_servoing.uv_methods import gripper_camera_calibration, project_delta

GRIPPER_KEY = "observation.images.gripper_camera"
OVERHEAD_KEY = "observation.images.overhead_camera"
OVERHEAD_SIZE = (448, 448)
MODE_PAIRS = "pairs"
MODE_CARRY = "carry"
SHARD_PREFIX = {MODE_PAIRS: "drop_pairs", MODE_CARRY: "placer"}

# Open commands this many frames apart are one opening.
OPEN_GAP_FRAMES = 10
# How far pressure must fall across an opening for it to have let go of something.
RELEASE_DROP = 0.02
# How far pressure may rise after the release and it still be the last one.
RELEASE_RISE = 0.02

POST_SECONDS = 1.0
MAX_CARRY_SECONDS = 10.0

OFFSET_METHODS = ("room-delta", "dead-reckon")
DEFAULT_OFFSET_METHOD = "room-delta"

# Seconds from the release onset at which each previewed episode is drawn.
PREVIEW_OFFSETS_S = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0)

# Pairs: gripper snapshots of the item taken while the laser reads this range, before the
# close begins - close enough to fill the frame, far enough that nothing covers it yet.
SNAPSHOT_RANGE_M = (0.12, 0.25)
SNAPSHOTS_PER_EPISODE = 4
SNAPSHOT_SEARCH_SECONDS = 10.0
# The empty gripper's view of the drop point, this long after the opening ends.
DROP_VIEW_DELAY_S = 0.3


def row_schema():
    import pyarrow as pa

    vector = pa.list_(pa.float32())
    return pa.schema([
        ("image", pa.binary()),
        ("overhead_image", pa.binary()),
        ("source_repo_id", pa.string()),
        ("episode_index", pa.int32()),
        ("frame_index", pa.int32()),
        ("task", pa.string()),
        ("seconds_to_release", pa.float32()),
        ("open_now", pa.int8()),
        ("holding", pa.int8()),
        ("release_offset_m", vector),
        ("release_height_m", pa.float32()),
        ("release_uv", vector),
        ("release_ortho_uv", vector),
        ("gripper_ortho_uv", vector),
        ("finger", pa.float32()),
        ("offset_method", pa.string()),
        ("state", pa.struct([
            ("laser_rangefinder", pa.float32()),
            ("finger_angle", pa.float32()),
            ("target_force", pa.float32()),
            ("finger_pressure", pa.float32()),
            ("wrist_angle", pa.float32()),
            ("gripper_z", pa.float32()),
        ])),
    ])


def pair_schema():
    import pyarrow as pa

    vector = pa.list_(pa.float32())
    return pa.schema([
        ("image", pa.binary()),
        ("snapshot_overhead_image", pa.binary()),
        ("release_overhead_image", pa.binary()),
        ("drop_view_image", pa.binary()),
        ("source_repo_id", pa.string()),
        ("episode_index", pa.int32()),
        ("frame_index", pa.int32()),
        ("release_frame_index", pa.int32()),
        ("task", pa.string()),
        ("seconds_before_grasp", pa.float32()),
        ("release_room_xy", vector),
        ("release_ortho_uv", vector),
        ("release_height_m", pa.float32()),
        ("pickup_room_xy", vector),
        ("pickup_ortho_uv", vector),
        ("state", pa.struct([
            ("laser_rangefinder", pa.float32()),
            ("finger_angle", pa.float32()),
            ("wrist_angle", pa.float32()),
            ("gripper_z", pa.float32()),
        ])),
    ])


IMAGE_COLUMNS = ("image", "overhead_image", "snapshot_overhead_image",
                 "release_overhead_image", "drop_view_image")


class PlacerShardWriter(ShardWriter):
    def __init__(self, split_dir: Path, mode: str):
        super().__init__(split_dir, prefix=SHARD_PREFIX[mode])
        self.schema = pair_schema() if mode == MODE_PAIRS else row_schema()

    def add(self, row: dict):
        super().add(row)
        self.pending += sum(len(row.get(k) or b"") for k in IMAGE_COLUMNS if k != "image")


def open_runs(rows, gap=OPEN_GAP_FRAMES):
    """(first, last) frame of every run of open commands, merging runs `gap` frames apart."""
    runs, first, last = [], None, None
    for i, r in enumerate(rows):
        if r["finger_speed"] < 0:
            if first is not None and i - last > gap + 1:
                runs.append((first, last))
                first = None
            if first is None:
                first = i
            last = i
    if first is not None:
        runs.append((first, last))
    return runs


def find_release(rows):
    """(onset, end) of the last opening that let go of something, or None.

    Letting go means pressure falls across the opening and never rises again afterwards,
    which is what separates the release from openings followed by a regrasp.
    """
    pressure = np.array([r["pressure"] for r in rows], dtype=np.float64)
    found = None
    for first, last in open_runs(rows):
        before = pressure[max(0, first - 3):first + 1].max()
        tail = pressure[last:]
        rise = float((tail - np.minimum.accumulate(tail)).max())
        if before - tail.min() >= RELEASE_DROP and rise < RELEASE_RISE:
            found = (first, last)
    return found


def last_grasp(pressure, before, fps):
    """The last grasp that starts before frame `before`, or None."""
    grasp, start = None, 0
    while start < before:
        k = find_grasp(pressure[start:before], fps, PRESSURE_THRESHOLD, MIN_GRASP_SECONDS)
        if k is None:
            break
        grasp = start + k
        start = grasp + 1
        while start < before and pressure[start] > PRESSURE_THRESHOLD:
            start += 1
    return grasp


def room_deltas(rows, onset, method):
    """Room-frame vector from the jaws at each frame to the jaws at the release onset."""
    if method == "room-delta":
        target = rows[onset]["gripper_pos"]
        return [target - r["gripper_pos"] for r in rows]
    if rows[0]["vel_cmd"] is None:
        raise ValueError("dead-reckon needs vel_x/vel_y/vel_z in the recorded action")
    # The commanded velocity is in the gripper's frame, so it is turned by -spin.
    steps = [rotate_about_vertical(rows[k]["vel_cmd"], -float(rows[k]["spin"]))
             * float(rows[k + 1]["timestamp"] - rows[k]["timestamp"])
             for k in range(len(rows) - 1)]
    deltas = [None] * len(rows)
    deltas[onset] = np.zeros(3)
    for k in range(onset - 1, -1, -1):
        deltas[k] = deltas[k + 1] + steps[k]
    for k in range(onset + 1, len(rows)):
        deltas[k] = deltas[k - 1] - steps[k - 1]
    return deltas


def ortho_uv(x, y):
    return [round(float(c), 5) for c in room_to_ortho_px(x, y, 1.0, 1.0, ORTHO_EXTENT_M)]


def mine_episode(rows, fps, calibration, method=DEFAULT_OFFSET_METHOD, post_seconds=POST_SECONDS,
                 max_carry_seconds=MAX_CARRY_SECONDS, stride=1):
    """Label rows for one episode's carry and release, or (None, reason) if unusable."""
    release = find_release(rows)
    if release is None:
        return None, "no_release"
    onset, end = release
    pressure = np.array([r["pressure"] for r in rows], dtype=np.float64)
    grasp = last_grasp(pressure, onset, fps)
    if grasp is None:
        return None, "no_grasp"
    lift = find_lift(rows, grasp, fps)
    if lift is None or lift >= onset:
        return None, "no_lift"

    deltas = room_deltas(rows, onset, method)
    floor_z = float(rows[grasp]["gripper_pos"][2])
    release_pos = rows[onset]["gripper_pos"]
    release_ortho = ortho_uv(release_pos[0], release_pos[1])
    height = round(float(release_pos[2]) - floor_z, 4)

    first = max(lift, onset - int(round(max_carry_seconds * fps)))
    last = min(len(rows) - 1, end + int(round(post_seconds * fps)))
    out = []
    for i in range(first, last + 1, stride):
        r = rows[i]
        spin = float(r["spin"])
        delta = deltas[i]
        offset = None
        if i <= end:
            offset = [round(float(c), 4) for c in rotate_about_vertical(delta, spin)]
        # the floor straight below the release jaws, as seen from this frame
        below = delta + np.array([0.0, 0.0, floor_z - float(release_pos[2])])
        projected = project_delta(below, spin, calibration)
        out.append({
            "frame_index": r["frame_index"],
            "timestamp": r["timestamp"],
            "seconds_to_release": round(rows[onset]["timestamp"] - r["timestamp"], 3),
            "open_now": 1 if i >= onset else 0,
            "holding": 1 if i < onset else (0 if i > end else None),
            "release_offset_m": offset,
            "release_height_m": height,
            "release_uv": None if projected is None else [round(projected[0], 5),
                                                         round(projected[1], 5)],
            "release_ortho_uv": release_ortho,
            "gripper_ortho_uv": ortho_uv(r["gripper_pos"][0], r["gripper_pos"][1]),
            "finger": round(float(r["finger_speed"]) / FINGER_SPEED_FULL_SCALE, 4),
            "offset_method": method,
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "target_force": round(float(r["target_force"]), 4),
                "finger_pressure": round(float(r["pressure"]), 4),
                "wrist_angle": round(float(r["wrist_angle"]), 3),
                "gripper_z": round(float(r["gripper_pos"][2]), 4),
            },
        })
    return out, None


def mine_pair_episode(rows, fps, snapshots=SNAPSHOTS_PER_EPISODE,
                      snapshot_range=SNAPSHOT_RANGE_M):
    """Item snapshots from before the grasp, each paired with where that item was dropped.

    Returns (samples, None), or (None, reason) if the episode has no usable pair.
    """
    release = find_release(rows)
    if release is None:
        return None, "no_release"
    onset, end = release
    pressure = np.array([r["pressure"] for r in rows], dtype=np.float64)
    grasp = last_grasp(pressure, onset, fps)
    if grasp is None:
        return None, "no_grasp"
    lift = find_lift(rows, grasp, fps)
    if lift is None or lift >= onset:
        return None, "no_lift"

    # before the close starts, so the fingers are not yet across the item
    close = close_onset(rows, grasp) or grasp
    first = max(0, grasp - int(round(SNAPSHOT_SEARCH_SECONDS * fps)))
    low, high = snapshot_range
    candidates = [i for i in range(first, close)
                  if low <= rows[i]["laser_rangefinder"] <= high]
    if not candidates:
        return None, "no_snapshot"
    picks = sorted({candidates[int(round(k))]
                    for k in np.linspace(0, len(candidates) - 1, min(snapshots, len(candidates)))})

    release_pos, pickup_pos = rows[onset]["gripper_pos"], rows[grasp]["gripper_pos"]
    drop_view = min(len(rows) - 1, end + int(round(DROP_VIEW_DELAY_S * fps)))
    shared = {
        "release_frame_index": rows[onset]["frame_index"],
        "release_room_xy": [round(float(c), 4) for c in release_pos[:2]],
        "release_ortho_uv": ortho_uv(release_pos[0], release_pos[1]),
        "release_height_m": round(float(release_pos[2] - pickup_pos[2]), 4),
        "pickup_room_xy": [round(float(c), 4) for c in pickup_pos[:2]],
        "pickup_ortho_uv": ortho_uv(pickup_pos[0], pickup_pos[1]),
        "_release_timestamp": rows[onset]["timestamp"],
        "_drop_view_timestamp": rows[drop_view]["timestamp"],
    }
    out = []
    for i in picks:
        r = rows[i]
        out.append({
            **shared,
            "frame_index": r["frame_index"],
            "timestamp": r["timestamp"],
            "seconds_before_grasp": round(rows[grasp]["timestamp"] - r["timestamp"], 3),
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "wrist_angle": round(float(r["wrist_angle"]), 3),
                "gripper_z": round(float(r["gripper_pos"][2]), 4),
            },
        })
    return out, None


class FrameReader:
    """Decodes frames of one dataset's videos by episode and timestamp."""

    def __init__(self, root: Path):
        import pyarrow.parquet as pq

        self.root = root
        self.videos, self.tasks = {}, {}
        for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
            for row in pq.read_table(path).to_pylist():
                ep = row["episode_index"]
                tasks = row.get("tasks") or []
                self.tasks[ep] = tasks[0] if tasks else ""
                for key in (GRIPPER_KEY, OVERHEAD_KEY):
                    chunk = row.get(f"videos/{key}/chunk_index")
                    if chunk is None:
                        continue
                    file = row[f"videos/{key}/file_index"]
                    video = root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
                    self.videos[(ep, key)] = (video, row[f"videos/{key}/from_timestamp"])

    def has(self, ep, key):
        return (ep, key) in self.videos and self.videos[(ep, key)][0].exists()

    def frames(self, ep, key, timestamps, fps, batch=64):
        """BGR uint8 frames at `timestamps` (seconds into the episode)."""
        from lerobot.datasets.video_utils import decode_video_frames

        video, start = self.videos[(ep, key)]
        out = []
        for k in range(0, len(timestamps), batch):
            chunk = [start + t for t in timestamps[k:k + batch]]
            decoded = decode_video_frames(video, chunk, 0.5 / fps, backend="pyav",
                                          return_uint8=True)
            out += [cv2.cvtColor(f.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR) for f in decoded]
        return out


def encode(bgr, size):
    resized = cv2.resize(bgr, tuple(size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def attach_images(samples, reader, ep, fps, overhead=True):
    times = [s["timestamp"] for s in samples]
    grip = reader.frames(ep, GRIPPER_KEY, times, fps)
    over = (reader.frames(ep, OVERHEAD_KEY, times, fps)
            if overhead and reader.has(ep, OVERHEAD_KEY) else [None] * len(samples))
    for s, g, o in zip(samples, grip, over):
        s["image"] = encode(g, IMAGE_SIZE)
        s["overhead_image"] = None if o is None else encode(o, OVERHEAD_SIZE)


def attach_pair_images(samples, reader, ep, fps):
    times = [s["timestamp"] for s in samples]
    once = [samples[0]["_release_timestamp"]]
    has_overhead = reader.has(ep, OVERHEAD_KEY)
    snapshot_overheads = (reader.frames(ep, OVERHEAD_KEY, times, fps) if has_overhead
                          else [None] * len(samples))
    release_overhead = (encode(reader.frames(ep, OVERHEAD_KEY, once, fps)[0], OVERHEAD_SIZE)
                        if has_overhead else None)
    drop_view = encode(reader.frames(ep, GRIPPER_KEY, [samples[0]["_drop_view_timestamp"]], fps)[0],
                       IMAGE_SIZE)
    for s, g, o in zip(samples, reader.frames(ep, GRIPPER_KEY, times, fps), snapshot_overheads):
        s["image"] = encode(g, IMAGE_SIZE)
        s["snapshot_overhead_image"] = None if o is None else encode(o, OVERHEAD_SIZE)
        s["release_overhead_image"] = release_overhead
        s["drop_view_image"] = drop_view


def source_root(repo_id, root=None):
    """A local copy holding only what mining reads: metadata, parquets and two video feeds."""
    if root:
        return Path(root)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns=[
        "meta/**", "data/**", f"videos/{GRIPPER_KEY}/**", f"videos/{OVERHEAD_KEY}/**"]))


def mine(sources, output_root=None, split=POOL_SPLIT, mode=MODE_PAIRS,
         method=DEFAULT_OFFSET_METHOD, post_seconds=POST_SECONDS,
         max_carry_seconds=MAX_CARRY_SECONDS, stride=1, limit=None, overhead=True,
         snapshots=SNAPSHOTS_PER_EPISODE, preview_dir=None, preview_episodes=8, preview_seed=0):
    """Mine every source; write shards unless output_root is None, then draw previews."""
    calibration = gripper_camera_calibration()
    writer = None
    if output_root is not None:
        split_dir = Path(output_root) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for old in split_dir.glob(f"{SHARD_PREFIX[mode]}-*.parquet"):
            old.unlink()
        writer = PlacerShardWriter(split_dir, mode)

    mined, skipped, rows_total = [], {}, 0
    for repo_id, root in sources:
        episodes, fps = read_columns(root)
        reader = FrameReader(root)
        for ep in sorted(episodes)[:limit]:
            if mode == MODE_PAIRS:
                samples, reason = mine_pair_episode(episodes[ep], fps, snapshots)
            else:
                samples, reason = mine_episode(episodes[ep], fps, calibration, method,
                                               post_seconds, max_carry_seconds, stride)
            if samples is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            for s in samples:
                s.update(source_repo_id=repo_id, episode_index=ep, task=reader.tasks.get(ep, ""))
            rows_total += len(samples)
            mined.append((repo_id, ep, reader, fps, samples))
            if writer is not None:
                if mode == MODE_PAIRS:
                    attach_pair_images(samples, reader, ep, fps)
                else:
                    attach_images(samples, reader, ep, fps, overhead)
                for s in samples:
                    writer.add({k: v for k, v in s.items()
                                if k != "timestamp" and not k.startswith("_")})
                    for k in IMAGE_COLUMNS:
                        s.pop(k, None)
    if writer is not None:
        writer.flush()
    logging.info(f"{len(mined)} episode(s) mined into {rows_total} row(s); skipped {skipped or 'none'}")

    if preview_dir is not None:
        chosen = random.Random(preview_seed).sample(mined, min(preview_episodes, len(mined)))
        if mode == MODE_PAIRS:
            write_pair_previews(sorted(chosen, key=lambda m: (m[0], m[1])), Path(preview_dir))
        else:
            write_previews(chosen, Path(preview_dir), overhead)
    return mined


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

GREEN, BLUE, GREY, WHITE = (60, 220, 60), (255, 160, 40), (140, 140, 140), (255, 255, 255)


def _text(img, lines, origin=(8, 20), scale=0.5):
    for k, line in enumerate(lines):
        y = origin[1] + k * int(26 * scale / 0.5 * 0.8)
        cv2.putText(img, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
        cv2.putText(img, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1)


def _gripper_panel(bgr, sample):
    w, h = IMAGE_SIZE
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    pad_x, pad_y = w // 4, h // 4
    canvas = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT,
                                value=(40, 40, 40))
    cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + w, pad_y + h), (90, 90, 90), 1)
    uv = sample["release_uv"]
    if uv is not None:
        x = int(np.clip(uv[0] * w + pad_x, 4, canvas.shape[1] - 5))
        y = int(np.clip(uv[1] * h + pad_y, 4, canvas.shape[0] - 5))
        cv2.drawMarker(canvas, (x, y), GREEN, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(canvas, (x, y), 12, GREEN, 2)
    else:
        _text(canvas, ["release point behind the lens"], (pad_x + 8, pad_y + h - 10))
    return canvas


def _overhead_panel(bgr, sample, size):
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    now = [int(c * size) for c in sample["gripper_ortho_uv"]]
    target = [int(c * size) for c in sample["release_ortho_uv"]]
    cv2.line(img, tuple(now), tuple(target), GREY, 1)
    cv2.circle(img, tuple(now), 7, BLUE, 2)
    cv2.drawMarker(img, tuple(target), GREEN, cv2.MARKER_CROSS, 18, 2)
    return img


def _cell(sample, grip_bgr, over_bgr):
    grip = _gripper_panel(grip_bgr, sample)
    parts = [grip]
    if over_bgr is not None:
        parts.append(_overhead_panel(over_bgr, sample, grip.shape[0]))
    body = np.hstack(parts)
    band = np.full((78, body.shape[1], 3), 25, dtype=np.uint8)
    off = sample["release_offset_m"]
    state = sample["state"]
    _text(band, [
        f"ep{sample['episode_index']} f{sample['frame_index']}  t{-sample['seconds_to_release']:+.2f}s"
        f"  open_now {sample['open_now']}  holding {sample['holding']}  {sample['task']}",
        ("offset none" if off is None else f"offset x {off[0]:+.2f} y {off[1]:+.2f} z {off[2]:+.2f} m")
        + f"  release height {sample['release_height_m']:.2f} m",
        f"laser {state['laser_rangefinder']:.2f}  pressure {state['finger_pressure']:.3f}"
        f"  finger {state['finger_angle']:.0f}deg  z {state['gripper_z']:+.2f}",
    ])
    return np.vstack([band, body])


def write_previews(chosen, preview_dir: Path, overhead=True):
    """One tall image per episode: the carry at fixed times before and after the release."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in preview_dir.glob("ep*.jpg"):
        old.unlink()
    for repo_id, ep, reader, fps, samples in chosen:
        picked = []
        for offset in PREVIEW_OFFSETS_S:
            nearest = min(samples, key=lambda s: abs(-s["seconds_to_release"] - offset))
            if abs(-nearest["seconds_to_release"] - offset) < 0.5 / fps + 1e-6 and nearest not in picked:
                picked.append(nearest)
        times = [s["timestamp"] for s in picked]
        grips = reader.frames(ep, GRIPPER_KEY, times, fps)
        overs = (reader.frames(ep, OVERHEAD_KEY, times, fps)
                 if overhead and reader.has(ep, OVERHEAD_KEY) else [None] * len(picked))
        cells = [_cell(s, g, o) for s, g, o in zip(picked, grips, overs)]
        name = f"ep{ep:04d}_{repo_id.replace('/', '_')}.jpg"
        cv2.imwrite(str(preview_dir / name), np.vstack(cells), [cv2.IMWRITE_JPEG_QUALITY, 88])
    logging.info(f"wrote {len(chosen)} episode preview(s) to {preview_dir}")


PAIR_PANEL_H = 336
PAIRS_PER_SHEET = 6


def _fit(bgr, height):
    return cv2.resize(bgr, (int(round(bgr.shape[1] * height / bgr.shape[0])), height),
                      interpolation=cv2.INTER_AREA)


def _mark(img, uv, colour, marker):
    x, y = int(uv[0] * img.shape[1]), int(uv[1] * img.shape[0])
    if marker == "cross":
        cv2.drawMarker(img, (x, y), colour, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(img, (x, y), 12, colour, 2)
    else:
        cv2.circle(img, (x, y), 8, colour, 2)


def _pair_row(sample, snapshot, snapshot_overhead, release_overhead, drop_view):
    h = PAIR_PANEL_H
    panels = [_fit(snapshot, h)]
    if snapshot_overhead is not None:
        before = _fit(snapshot_overhead, h)
        _mark(before, sample["pickup_ortho_uv"], BLUE, "circle")
        after = _fit(release_overhead, h)
        _mark(after, sample["pickup_ortho_uv"], BLUE, "circle")
        _mark(after, sample["release_ortho_uv"], GREEN, "cross")
        panels += [before, after]
    panels.append(_fit(drop_view, h))
    body = np.hstack(panels)
    band = np.full((56, body.shape[1], 3), 25, dtype=np.uint8)
    rx, ry = sample["release_room_xy"]
    _text(band, [
        f"ep{sample['episode_index']} f{sample['frame_index']}  {sample['task']}"
        f"  snapshot {sample['seconds_before_grasp']:.1f}s before grasp,"
        f" laser {sample['state']['laser_rangefinder']:.2f}m",
        f"dropped at room ({rx:+.2f}, {ry:+.2f}) m, {sample['release_height_m']:.2f} m above pickup"
        f"   panels: item snapshot | overhead then (blue: pickup) | overhead at drop (green: drop)"
        f" | gripper after drop",
    ])
    return np.vstack([band, body])


def write_pair_previews(chosen, preview_dir: Path):
    """Contact sheets, one line per episode: the item, the room then, the drop, the drop point."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in preview_dir.glob("pairs_*.jpg"):
        old.unlink()
    lines = []
    for repo_id, ep, reader, fps, samples in chosen:
        sample = samples[len(samples) // 2]
        t = sample["timestamp"]
        has_overhead = reader.has(ep, OVERHEAD_KEY)
        lines.append(_pair_row(
            sample,
            reader.frames(ep, GRIPPER_KEY, [t], fps)[0],
            reader.frames(ep, OVERHEAD_KEY, [t], fps)[0] if has_overhead else None,
            (reader.frames(ep, OVERHEAD_KEY, [sample["_release_timestamp"]], fps)[0]
             if has_overhead else None),
            reader.frames(ep, GRIPPER_KEY, [sample["_drop_view_timestamp"]], fps)[0],
        ))
    width = max(line.shape[1] for line in lines)
    lines = [cv2.copyMakeBorder(line, 0, 4, 0, width - line.shape[1], cv2.BORDER_CONSTANT,
                                value=(0, 0, 0)) for line in lines]
    for k in range(0, len(lines), PAIRS_PER_SHEET):
        name = preview_dir / f"pairs_{k // PAIRS_PER_SHEET + 1:02d}.jpg"
        cv2.imwrite(str(name), np.vstack(lines[k:k + PAIRS_PER_SHEET]),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    logging.info(f"wrote {len(chosen)} episode(s) of pair previews to {preview_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", nargs="+", required=True, help="teleop dataset(s) to mine")
    parser.add_argument("--root", default=None,
                        help="local copy of a single --repo_id (default: download what is needed)")
    parser.add_argument("--output_root", default=None, help="dataset directory to write shards into")
    parser.add_argument("--split", default=POOL_SPLIT)
    parser.add_argument("--mode", choices=(MODE_PAIRS, MODE_CARRY), default=MODE_PAIRS,
                        help="pairs: item snapshots paired with their drop point. carry: every "
                             "frame of the carry with release labels. The two have different "
                             "schemas, so give each its own --output_root")
    parser.add_argument("--snapshots", type=int, default=SNAPSHOTS_PER_EPISODE,
                        help="pairs: item snapshots per episode")
    parser.add_argument("--offset_method", choices=OFFSET_METHODS, default=DEFAULT_OFFSET_METHOD,
                        help="carry: how release_offset_m is computed")
    parser.add_argument("--post_seconds", type=float, default=POST_SECONDS,
                        help="carry: frames kept after the opening ends")
    parser.add_argument("--max_carry_seconds", type=float, default=MAX_CARRY_SECONDS,
                        help="carry: earliest frame kept, before the release onset")
    parser.add_argument("--stride", type=int, default=1, help="carry: keep every Nth frame")
    parser.add_argument("--limit", type=int, default=None, help="first N episodes of each source")
    parser.add_argument("--no_overhead", action="store_true", help="carry: leave overhead_image null")
    parser.add_argument("--preview_dir", default=None)
    parser.add_argument("--preview_episodes", type=int, default=8)
    parser.add_argument("--preview_seed", type=int, default=0)
    parser.add_argument("--preview_only", action="store_true",
                        help="label and preview without writing shards")
    args = parser.parse_args()

    if args.root and len(args.repo_id) > 1:
        parser.error("--root names one dataset; pass a single --repo_id with it")
    if not args.preview_only and args.output_root is None:
        parser.error("--output_root is required unless --preview_only")
    if args.preview_only and args.preview_dir is None:
        parser.error("--preview_only needs --preview_dir")

    sources = [(repo_id, source_root(repo_id, args.root)) for repo_id in args.repo_id]
    mine(sources, None if args.preview_only else args.output_root, args.split, args.mode,
         args.offset_method, args.post_seconds, args.max_carry_seconds, args.stride, args.limit,
         not args.no_overhead, args.snapshots, args.preview_dir, args.preview_episodes,
         args.preview_seed)


if __name__ == "__main__":
    main()
