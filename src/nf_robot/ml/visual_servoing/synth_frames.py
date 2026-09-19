#!/usr/bin/env python

"""Composite exactly labelled synthetic visual servoing frames from floor, object and finger plates.

Usage:
    python -m nf_robot.ml.visual_servoing.synth_frames \
        --plates plates --output_root datasets/visual_servoing --count 20000
"""

import argparse
from datetime import datetime
import json
import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np

try:
    # Unix only; absent on Windows, where the peak-memory line is simply skipped.
    import resource
except ImportError:
    resource = None

from nf_robot.ml.visual_servoing.mine_teleop import (
    CANVAS_SCALE, IMAGE_SIZE, POOL_SPLIT, ShardWriter, encode_frame,
)
from nf_robot.ml.visual_servoing.object_matte import read_objects
from nf_robot.ml.visual_servoing.plates import iter_run, read_manifest
from nf_robot.ml.visual_servoing import white_balance

# Simulated camera heights, sampled log-uniformly when no real distribution is supplied.
RANGE_MIN_M = 0.12
RANGE_MAX_M = 1.10
# How many objects land in the canvas; zero gives the target-present head its negatives.
OBJECT_COUNT_WEIGHTS = {0: 0.12, 1: 0.45, 2: 0.25, 3: 0.12, 4: 0.06}
# Where the jaws are in the frame, the same point mine_teleop's anchor projects to.
JAW_REF_UV = (0.5, 0.308)
# Finger apertures to sample, in degrees; -90 is fully open.
FINGER_ANGLE_RANGE = (-90.0, 90.0)
# Extra inward zoom a floor plate may get on top of the scale its range implies.
FLOOR_ZOOM_MAX = 1.12
# Sign taking a cutout's wrist offset to the image-plane grasp axis; flip it if annotated
# bars look mirrored.
AXIS_FROM_WRIST_SIGN = -1.0
# (MB) decoded floor plate pool budget, this tool's high water mark.
FLOOR_CACHE_MB = 2048
# (MB) image data a synthetic shard buffers before it is written.
SHARD_MB = 256
# (metres) the simulated heights this tool will composite, excluding sensor-in-contact
# readings.
SIM_RANGE_M = (0.05, 1.5)
# Hard cap on magnifying any plate or cutout, a backstop against bad ranges.
MAX_MAGNIFICATION = 6.0


def neutralize_plates(entries, what):
    """Remove one capture's colour cast from its plates in place, returning the illuminant
    removed."""
    if not entries:
        return np.ones(3)
    illuminant = white_balance.estimate_illuminant(e["image"] for e in entries)
    gains = white_balance.neutralize_gains(illuminant)
    for entry in entries:
        entry["image"] = white_balance.apply_gains(entry["image"], gains)
    logging.info(f"{what}: lit at RGB {np.round(illuminant, 3)}, "
                 f"neutralized by {np.round(gains, 3)}")
    return illuminant


def load_floorplates(plate_dir, limit_runs=None, budget_mb=FLOOR_CACHE_MB, seed=0):
    """A memory-bounded, reservoir-sampled pool of decoded floorplate frames with their
    capture ranges."""
    runs = [r for r in read_manifest(plate_dir) if r["kind"] == "floorplates"]
    if limit_runs:
        runs = runs[-limit_runs:]
    if not runs:
        return []

    # Decode one frame up front to learn the per-frame cost.
    probe = iter_run(plate_dir, runs[0]["run_id"])
    try:
        first = next(probe, None)
    finally:
        probe.close()
    if first is None:
        return []
    frame_bytes = first["image"].nbytes

    capacity = max(len(runs), int(budget_mb * 1e6 // frame_bytes))
    quota = max(1, capacity // len(runs))
    rng = random.Random(seed)

    plates, seen_total, illuminants = [], 0, {}
    for run in runs:
        kept, seen = [], 0
        for row in iter_run(plate_dir, run["run_id"]):
            range_m = row["laser_rangefinder"] or row["attrs"].get("target_range_m")
            if not range_m:
                continue
            seen += 1
            entry = {"image": row["image"], "range_m": float(range_m),
                     "run_id": run["run_id"]}
            if len(kept) < quota:
                kept.append(entry)
            else:
                # Standard reservoir replacement.
                index = rng.randrange(seen)
                if index < quota:
                    kept[index] = entry
        # White balance per run, since each run is its own lighting.
        if kept:
            illuminants[run["run_id"]] = neutralize_plates(kept, run["run_id"])
        plates += kept
        seen_total += seen

    resident = len(plates) * frame_bytes / 1e6
    logging.info(f"{len(plates)} floor plates held from {seen_total} frames across "
                 f"{len(runs)} run(s), {resident:.0f} MB resident "
                 f"(budget {budget_mb} MB, {quota} per run)")
    if len(plates) < seen_total:
        logging.info("raise --floor_cache_mb for more background variety if the machine "
                     "has the memory for it")
    return plates, illuminants


def run_time(run_id):
    """The capture time a run id carries, or None if it is not shaped like one."""
    parts = str(run_id).split("-")
    try:
        return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
    except (IndexError, ValueError):
        return None


def nearest_illuminant(run_id, measured):
    """The illuminant of the measured capture closest in time to run_id."""
    when = run_time(run_id)
    dated = {other: value for other, value in measured.items() if run_time(other)}
    if not dated:
        return np.ones(3)
    if when is None:
        return next(iter(dated.values()))
    return dated[min(dated, key=lambda other: abs(run_time(other) - when))]


def cutout_gains(entries, floor_illuminants):
    """Neutralizing gains per objectplates run, borrowed from the nearest floorplates run."""
    gains = {}
    for run_id in sorted({entry.get("run_id", "") for entry in entries}):
        illuminant = nearest_illuminant(run_id, floor_illuminants)
        gains[run_id] = white_balance.neutralize_gains(illuminant)
        logging.info(f"cutouts from {run_id or 'unknown run'}: neutralized by "
                     f"{np.round(gains[run_id], 3)}, borrowed from the nearest floor capture")
    return gains


def load_finger_plates(finger_dir, neutralize=True):
    """RGBA finger plates grouped by capture, as [[(finger_angle, rgba), ...], ...]."""
    finger_dir = Path(finger_dir)
    manifest = finger_dir / "mattes.jsonl"
    if not manifest.exists():
        logging.warning(f"no finger mattes at {manifest}; frames will have no fingers")
        return []
    by_run = {}
    for line in open(manifest):
        entry = json.loads(line)
        bgra = cv2.imread(str(finger_dir / entry["file"]), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.shape[2] != 4:
            continue
        # Downscaled once to frame size, the only size they are pasted at.
        if (bgra.shape[1], bgra.shape[0]) != IMAGE_SIZE:
            bgra = cv2.resize(bgra, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        by_run.setdefault(entry.get("run_id", ""), []).append(
            (float(entry["finger_angle"]), bgra[:, :, [2, 1, 0, 3]]))
    for run_id, plates in sorted(by_run.items()):
        # White balance per run, since finger sets really are different colours.
        if neutralize:
            illuminant = white_balance.estimate_illuminant(rgba for _, rgba in plates)
            gains = white_balance.neutralize_gains(illuminant)
            plates[:] = [(angle, white_balance.apply_gains(rgba, gains)) for angle, rgba in plates]
            logging.info(f"{len(plates)} finger plates from {run_id or finger_dir}: "
                         f"lit at RGB {np.round(illuminant, 3)}, neutralized by {np.round(gains, 3)}")
        else:
            logging.info(f"{len(plates)} finger plates from {run_id or finger_dir}")
    return [sorted(plates) for _, plates in sorted(by_run.items())]


def capped_scale(scale, what):
    """A rescale factor clamped to MAX_SCALE, logged when it binds."""
    if scale <= MAX_MAGNIFICATION:
        return scale
    logging.warning(f"{what}: magnification {scale:.1f}x capped at {MAX_MAGNIFICATION}x; "
                    f"check the capture range on this plate")
    return MAX_MAGNIFICATION


def sample_ranges(dataset_root, count, rng):
    """Simulated heights, drawn from real teleop ranges when a mined dataset is given."""
    if dataset_root:
        import pyarrow.parquet as pq

        values = []
        for shard in sorted(Path(dataset_root).glob("*.parquet")):
            table = pq.read_table(shard, columns=["state"])
            values += [row["laser_rangefinder"] for row in table.column("state").to_pylist()
                       if row and row["laser_rangefinder"]]
        if values:
            logging.info(f"sampling ranges from {len(values)} mined rows "
                         f"({np.percentile(values, 5):.2f}-{np.percentile(values, 95):.2f}m)")
            drawn = [float(rng.choice(values)) for _ in range(count)]
            return clamp_ranges(drawn)
        logging.warning(f"no ranges found under {dataset_root}; falling back to log-uniform")
    low, high = math.log(RANGE_MIN_M), math.log(RANGE_MAX_M)
    return clamp_ranges([math.exp(rng.uniform(low, high)) for _ in range(count)])


def clamp_ranges(values):
    """Simulated heights clamped into SIM_RANGE_M, reporting how many moved."""
    low, high = SIM_RANGE_M
    clamped = [min(max(v, low), high) for v in values]
    moved = sum(1 for v, c in zip(values, clamped) if v != c)
    if moved:
        logging.info(f"{moved}/{len(values)} simulated heights clamped into "
                     f"{low}-{high}m; readings outside it are the sensor in contact or "
                     f"out of range rather than a view of a floor")
    return clamped


def floor_canvas(plate, target_range, canvas_size, rng):
    """A floor plate rescaled to the simulated height and cropped to the canvas, zooming in
    rather than tiling to fill the frame."""
    image = plate["image"]
    frame_w, frame_h = IMAGE_SIZE
    canvas_w, canvas_h = canvas_size

    physical = (frame_w / image.shape[1]) * (plate["range_m"] / target_range)
    # the least magnification that still fills the frame on both axes
    cover = max(frame_w / image.shape[1], frame_h / image.shape[0])
    scale = capped_scale(max(physical, cover) * rng.uniform(1.0, FLOOR_ZOOM_MAX),
                         f"floor plate {plate['run_id']} at {plate['range_m']:.2f}m")

    # ceil, and never below the frame, so rounding cannot leave a row of pixels missing
    width = max(frame_w, math.ceil(image.shape[1] * scale))
    height = max(frame_h, math.ceil(image.shape[0] * scale))
    scaled = cv2.resize(image, (width, height),
                        interpolation=cv2.INTER_AREA if width < image.shape[1] else cv2.INTER_LINEAR)

    x = rng.randint(0, width - frame_w)
    y = rng.randint(0, height - frame_h)
    crop = scaled[y:y + frame_h, x:x + frame_w]

    pad_x, pad_y = (canvas_w - frame_w) // 2, (canvas_h - frame_h) // 2
    return cv2.copyMakeBorder(crop, pad_y, canvas_h - frame_h - pad_y,
                              pad_x, canvas_w - frame_w - pad_x, cv2.BORDER_REPLICATE)


def paste_rgba(canvas, rgba, top_left):
    """Alpha-composite an RGBA patch onto a canvas in place, clipped to it."""
    x, y = int(round(top_left[0])), int(round(top_left[1]))
    h, w = rgba.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x0 >= x1 or y0 >= y1:
        return
    patch = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = (patch[:, :, :3] * alpha + region * (1 - alpha)).astype(np.uint8)


def photometric(image, rng):
    """Colour temperature, exposure, white balance, noise, motion blur and JPEG quality."""
    image = white_balance.apply_gains(image, white_balance.random_illuminant_gains(rng))
    out = image.astype(np.float32)
    out *= rng.uniform(0.75, 1.3)
    out *= np.array([rng.uniform(0.94, 1.06) for _ in range(3)], dtype=np.float32)
    if rng.random() < 0.4:
        length = rng.randrange(3, 11)
        angle = rng.uniform(0, math.pi)
        kernel = np.zeros((length, length), np.float32)
        cv2.line(kernel, (0, length // 2), (length - 1, length // 2), 1.0, 1)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(
            (length / 2 - 0.5, length / 2 - 0.5), math.degrees(angle), 1.0), (length, length))
        total = kernel.sum()
        if total > 0:
            out = cv2.filter2D(out, -1, kernel / total)
    out += np.random.default_rng(rng.randrange(1 << 31)).normal(0, rng.uniform(0.5, 4.0), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    quality = rng.randrange(55, 96)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if ok else out


def axis_from_wrist_offset(offset_deg):
    """A cutout's wrist offset as the image-plane grasp axis in radians, with None passing
    through."""
    if offset_deg is None:
        return None
    return math.radians(AXIS_FROM_WRIST_SIGN * float(offset_deg))


def compose(floor_plates, objects, object_dir, finger_plates, target_range, rng,
            object_gains=None):
    """One synthetic frame and its labels, targeting the candidate nearest the jaws."""
    frame_w, frame_h = IMAGE_SIZE
    canvas_w, canvas_h = int(frame_w * CANVAS_SCALE), int(frame_h * CANVAS_SCALE)
    offset_x, offset_y = (canvas_w - frame_w) // 2, (canvas_h - frame_h) // 2

    # Prefer a plate captured no closer than the simulated height.
    higher = [p for p in floor_plates if p["range_m"] >= target_range]
    pool = higher or floor_plates
    plate = min(pool, key=lambda p: abs(math.log(p["range_m"] / target_range)))
    canvas = floor_canvas(plate, target_range, (canvas_w, canvas_h), rng)

    count = rng.choices(list(OBJECT_COUNT_WEIGHTS), weights=list(OBJECT_COUNT_WEIGHTS.values()))[0]
    candidates = []
    for _ in range(count) if objects else ():
        entry = rng.choice(objects)
        bgra = cv2.imread(str(Path(object_dir) / entry["file"]), cv2.IMREAD_UNCHANGED)
        if bgra is None:
            continue
        rgba = bgra[:, :, [2, 1, 0, 3]]
        if object_gains:
            rgba = white_balance.apply_gains(rgba, object_gains[entry.get("run_id", "")])
        scale = capped_scale(
            (IMAGE_SIZE[0] / entry.get("capture_width", IMAGE_SIZE[0]))
            * float(entry["range_m"]) / target_range,
            f"cutout {entry['file']} at {entry['range_m']}m")
        rgba = cv2.resize(rgba, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        grasp = (entry["grasp_x"] * scale, entry["grasp_y"] * scale)
        top_left = (rng.uniform(-rgba.shape[1] * 0.5, canvas_w - rgba.shape[1] * 0.5),
                    rng.uniform(-rgba.shape[0] * 0.5, canvas_h - rgba.shape[0] * 0.5))
        paste_rgba(canvas, rgba, top_left)
        candidates.append({
            "uv": ((top_left[0] + grasp[0] - offset_x) / frame_w,
                   (top_left[1] + grasp[1] - offset_y) / frame_h),
            # The wrist offset at capture is the object's rotation from upright in the
            # image.
            "axis": axis_from_wrist_offset(entry.get("wrist_offset_deg")),
            "label": entry.get("label", ""),
        })

    frame = canvas[offset_y:offset_y + frame_h, offset_x:offset_x + frame_w].copy()

    finger_angle = rng.uniform(*FINGER_ANGLE_RANGE)
    if finger_plates:
        # Pick a set of fingers uniformly, then the nearest aperture within it.
        chosen = rng.choice(finger_plates)
        finger_angle, plate_rgba = min(chosen, key=lambda p: abs(p[0] - finger_angle))
        if (plate_rgba.shape[1], plate_rgba.shape[0]) != IMAGE_SIZE:
            plate_rgba = cv2.resize(plate_rgba, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        paste_rgba(frame, plate_rgba, (0, 0))

    frame = photometric(frame, rng)

    half = (CANVAS_SCALE - 1.0) / 2.0
    inside = [c for c in candidates
              if -half <= c["uv"][0] <= 1 + half and -half <= c["uv"][1] <= 1 + half]
    winner = min(inside, key=lambda c: math.dist(c["uv"], JAW_REF_UV)) if inside else None

    row = {
        "split_source": "synth",
        "source_repo_id": plate["run_id"],
        "episode_index": 0,
        "frame_index": 0,
        "seconds_to_grasp": None,
        "target_uv": [round(winner["uv"][0], 5), round(winner["uv"][1], 5)] if winner else None,
        # Simulated camera height, ignoring the object's own height.
        "target_range_m": round(target_range, 4) if winner else None,
        "grasp_axis_rad": (round(wrap_half_pi(winner["axis"]), 5)
                           if winner and winner["axis"] is not None else None),
        # No finger, close or pressure labels: compositing can't know them.
        "finger": None,
        "close_now": None,
        "grasp_pressure": None,
        "target_present": 1 if winner else 0,
        "holding": None,
        "state": {
            "laser_rangefinder": round(target_range, 4),
            "finger_angle": round(float(finger_angle), 3),
            "target_force": 0.0,
        },
    }
    return frame, row, candidates


def wrap_half_pi(radians):
    """Fold an angle into -pi/2..pi/2, where a pi-periodic grasp axis lives."""
    return (radians + math.pi / 2) % math.pi - math.pi / 2


def annotate(frame, row, candidates):
    """The frame with its labels drawn on, for eyeballing a compositing sign error."""
    canvas = cv2.copyMakeBorder(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                int(frame.shape[0] * 0.25), int(frame.shape[0] * 0.25),
                                int(frame.shape[1] * 0.25), int(frame.shape[1] * 0.25),
                                cv2.BORDER_CONSTANT, value=(40, 40, 40))
    pad_x, pad_y = int(frame.shape[1] * 0.25), int(frame.shape[0] * 0.25)
    cv2.rectangle(canvas, (pad_x, pad_y), (pad_x + frame.shape[1], pad_y + frame.shape[0]),
                  (90, 90, 90), 1)
    for candidate in candidates:
        x = int(candidate["uv"][0] * frame.shape[1] + pad_x)
        y = int(candidate["uv"][1] * frame.shape[0] + pad_y)
        cv2.drawMarker(canvas, (x, y), (0, 160, 255), cv2.MARKER_TILTED_CROSS, 12, 1)
    if row["target_uv"]:
        x = int(row["target_uv"][0] * frame.shape[1] + pad_x)
        y = int(row["target_uv"][1] * frame.shape[0] + pad_y)
        angle = row["grasp_axis_rad"]
        if angle is None:
            # No bar at all, so a missing axis isn't mistaken for zero.
            cv2.putText(canvas, "no axis", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 200, 255), 1)
        else:
            length = 26
            cv2.line(canvas,
                     (int(x - math.cos(angle) * length), int(y - math.sin(angle) * length)),
                     (int(x + math.cos(angle) * length), int(y + math.sin(angle) * length)),
                     (0, 200, 255), 2)
        cv2.drawMarker(canvas, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    text = (f"range {row['state']['laser_rangefinder']:.2f}  fing "
            f"{row['state']['finger_angle']:+.0f}  present {row['target_present']}  "
            f"cands {len(candidates)}")
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


def generate(plate_dir, output_root, split, count, seed, object_dir=None, finger_dir=None,
             ranges_from=None, annotate_dir=None, annotate_count=40,
             floor_cache_mb=FLOOR_CACHE_MB, shard_mb=SHARD_MB):
    # Each ingredient is white-balanced on load; photometric re-lights the finished frame.
    floor_plates, floor_illuminants = load_floorplates(plate_dir, budget_mb=floor_cache_mb, seed=seed)
    if not floor_plates:
        raise ValueError(f"no floorplates runs in {plate_dir}; nothing to build a background from")

    object_dir = Path(object_dir or Path(plate_dir) / "objects")
    objects = read_objects(object_dir)
    if not objects:
        logging.warning(f"no object cutouts in {object_dir}; every frame will be bare floor")
    object_gains = cutout_gains(objects, floor_illuminants)
    finger_plates = load_finger_plates(finger_dir or Path(plate_dir) / "fingers")
    if len(finger_plates) > 1:
        logging.info(f"{len(finger_plates)} sets of fingers; each frame picks one")

    rng = random.Random(seed)
    ranges = sample_ranges(ranges_from, count, rng)

    # Count frames above the tallest plate, where the floor texture looks nearer than
    # labelled.
    tallest = max(p["range_m"] for p in floor_plates)
    stretched = sum(1 for r in ranges if r > tallest)
    if stretched:
        logging.warning(f"{stretched}/{count} frames simulate a height above the tallest floor "
                        f"plate ({tallest:.2f}m); their floor is magnified past its true scale. "
                        f"Capture floorplates higher up to remove the approximation.")

    split_dir = Path(output_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    # Only this producer's shards are replaced; the miner's are left where they are.
    for stale in split_dir.glob("synth-*.parquet"):
        stale.unlink()

    if annotate_dir:
        Path(annotate_dir).mkdir(parents=True, exist_ok=True)
        for old in Path(annotate_dir).glob("*.jpg"):
            old.unlink()

    writer = ShardWriter(split_dir, prefix="synth", target_bytes=int(shard_mb * 1e6))
    present = 0
    for index, target_range in enumerate(ranges):
        frame, row, candidates = compose(floor_plates, objects, object_dir,
                                         finger_plates, target_range, rng, object_gains)
        row["image"] = encode_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        row["frame_index"] = index
        writer.add(row)
        present += row["target_present"]
        if annotate_dir and index < annotate_count:
            cv2.imwrite(str(Path(annotate_dir) / f"synth{index:04d}.jpg"),
                        annotate(frame, row, candidates))
        if (index + 1) % 2000 == 0:
            logging.info(f"{index + 1}/{count} frames")
    writer.flush()

    logging.info(f"{writer.total} synthetic frames in {writer.shards} shard(s) under "
                 f"{split_dir}; {present} with a target, {writer.total - present} without")
    # Measured peak memory, the number that decides whether the run survives.
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        logging.info(f"peak resident memory {peak:.1f} GB")
    return writer.total


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plates", default="plates", help="Directory of capture runs")
    parser.add_argument("--output_root", required=True,
                        help="Mined dataset root; synthetic shards join its pool")
    parser.add_argument("--split", default=POOL_SPLIT, choices=[POOL_SPLIT, "train", "eval"],
                        help=f"Where the shards land. The default is the {POOL_SPLIT}/ pool "
                             f"the miner also writes to, which split_pool deals into train "
                             f"and eval afterwards")
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--object_dir", default=None, help="Cutouts (default <plates>/objects)")
    parser.add_argument("--finger_dir", default=None, help="Finger plates (default <plates>/fingers)")
    parser.add_argument("--ranges_from", default=None,
                        help=f"A directory of mined shards to draw simulated heights from, "
                             f"normally the same {POOL_SPLIT}/ pool these join")
    parser.add_argument("--annotate_dir", default=None,
                        help="Write annotated sample frames here, to check the labels")
    parser.add_argument("--annotate_count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--floor_cache_mb", type=float, default=FLOOR_CACHE_MB,
                        help="Memory budget for the decoded floor plate pool, which is "
                             "this tool's high water mark. More is more background "
                             "variety; the full set of captures is tens of GB.")
    parser.add_argument("--shard_mb", type=float, default=SHARD_MB,
                        help="Image bytes buffered before a shard is written; the arrow "
                             "copy on the way out costs about the same again")
    args = parser.parse_args()

    generate(args.plates, args.output_root, args.split, args.count, args.seed,
             args.object_dir, args.finger_dir, args.ranges_from,
             args.annotate_dir, args.annotate_count,
             args.floor_cache_mb, args.shard_mb)


if __name__ == "__main__":
    main()
