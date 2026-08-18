#!/usr/bin/env python

"""Composite synthetic visual servoing frames from captured plates.

Offline, no robot. Three ingredients, each captured by a motion task in observer.py and
turned into usable pieces by its own extractor:

    floorplates    raw frames of bare floor at a spread of heights - the background
    objects        RGBA cutouts keyed off the green board (object_matte.py)
    fingers        RGBA plates of the gripper's own hardware (finger_matte.py)

and the labels come from the compositing rather than from looking at the result: we
know where we pasted the object, at what scale and at what orientation, so the target
position, its range and its grasp axis are exact by construction. That is the whole
reason for building frames this way instead of labelling more teleop.

Objects are pasted into a canvas 1.5x the frame, so a good fraction land partly or
wholly off the visible edge. That case - an object just past the bottom edge, which the
model must still point at - is the one this whole model exists to get right, so it has
to be common in training rather than a rare corner.

Fingers go on top, and objects that end up behind one are kept rather than avoided: that
occlusion is a large part of why the live frames are hard.

Output is parquet shards in the row format readme.md describes, written into the same
split directory the miner writes to and named apart from its shards, so training on the
mix needs nothing but generating some.

Usage:
    python -m nf_robot.ml.visual_servoing.synth_frames \
        --plates plates --output_root datasets/visual_servoing --count 20000
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.visual_servoing.mine_teleop import (
    CANVAS_SCALE,
    IMAGE_SIZE,
    ShardWriter,
    encode_frame,
)
from nf_robot.ml.visual_servoing.object_matte import read_objects
from nf_robot.ml.visual_servoing.plates import iter_run, read_manifest

# Simulated camera heights, in metres, when no real distribution is supplied. Sampled
# log-uniformly because what matters to the image is the ratio between heights, not the
# difference: the step from 0.2m to 0.3m changes the view far more than 0.9m to 1.0m.
RANGE_MIN_M = 0.12
RANGE_MAX_M = 1.10
# How many objects land in the canvas. Zero is deliberate and not rare: those frames are
# the only negatives the target-present head ever sees, since mined teleop rows are all
# positives by construction.
OBJECT_COUNT_WEIGHTS = {0: 0.12, 1: 0.45, 2: 0.25, 3: 0.12, 4: 0.06}
# Where the jaws are in the frame: straight down from the camera, which the 9.06 degree
# backward tilt puts above centre. Same point mine_teleop's anchor projects to, so
# "nearest the jaws" means the same thing in both halves of the dataset.
JAW_REF_UV = (0.5, 0.308)
# Finger apertures to sample, in degrees; -90 is fully open.
FINGER_ANGLE_RANGE = (-90.0, 90.0)


def load_floorplates(plate_dir, limit_runs=None):
    """Every floorplate frame in a directory, with the range it was captured at.

    Held in memory: a floor plate run is tens of frames, and the compositor picks a
    different one for every synthetic frame, so a random read per frame would spend the
    whole run seeking.
    """
    runs = [r for r in read_manifest(plate_dir) if r["kind"] == "floorplates"]
    if limit_runs:
        runs = runs[-limit_runs:]
    plates = []
    for run in runs:
        for row in iter_run(plate_dir, run["run_id"]):
            range_m = row["laser_rangefinder"] or row["attrs"].get("target_range_m")
            if range_m:
                plates.append({"image": row["image"], "range_m": float(range_m),
                               "run_id": run["run_id"]})
    logging.info(f"{len(plates)} floor plates from {len(runs)} run(s)")
    return plates


def load_finger_plates(finger_dir):
    """RGBA finger plates grouped by capture, as [[(finger_angle, rgba), ...], ...].

    One list per fingerplates run rather than one flat list, because a run is one physical
    set of fingers - they get swapped, and they are not all the same colour. Keeping them
    apart lets a frame pick a set and then an aperture within it, so the model sees each
    set at every aperture instead of a chimera that is blue at one angle and white at the
    next.
    """
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
        by_run.setdefault(entry.get("run_id", ""), []).append(
            (float(entry["finger_angle"]), bgra[:, :, [2, 1, 0, 3]]))
    for run_id, plates in sorted(by_run.items()):
        logging.info(f"{len(plates)} finger plates from {run_id or finger_dir}")
    return [sorted(plates) for _, plates in sorted(by_run.items())]


def sample_ranges(dataset_root, count, rng):
    """Simulated heights, drawn from real teleop ranges when a mined dataset is given.

    Matching the real distribution matters more than covering the span evenly: the model
    spends its time where the gripper spends its time, and a uniform sample over-trains
    the heights an operator flies through quickly.
    """
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
            return [float(rng.choice(values)) for _ in range(count)]
        logging.warning(f"no ranges found under {dataset_root}; falling back to log-uniform")
    low, high = math.log(RANGE_MIN_M), math.log(RANGE_MAX_M)
    return [math.exp(rng.uniform(low, high)) for _ in range(count)]


def floor_canvas(plate, target_range, canvas_size, rng):
    """A floor plate rescaled to the simulated height and cropped or tiled to the canvas.

    Two scalings, and forgetting either puts the floor at the wrong size. The plate was
    captured at some resolution of the same field of view the model input covers, so it
    first has to be scaled by the ratio between them; then a plate captured at r0 and
    viewed from r covers r0/r as much floor per pixel.

    Picking a plate captured at or above the target range keeps the second factor at or
    above 1, which is what keeps the tiling below out of the visible frame - it only
    covers the canvas margin, which is label space and never rendered.
    """
    image = plate["image"]
    scale = (IMAGE_SIZE[0] / image.shape[1]) * (plate["range_m"] / target_range)
    scaled = cv2.resize(image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    width, height = canvas_size
    if scaled.shape[1] < width or scaled.shape[0] < height:
        reps = (int(np.ceil(height / scaled.shape[0])), int(np.ceil(width / scaled.shape[1])), 1)
        scaled = np.tile(scaled, reps)
    x = rng.randrange(0, max(1, scaled.shape[1] - width + 1))
    y = rng.randrange(0, max(1, scaled.shape[0] - height + 1))
    return scaled[y:y + height, x:x + width].copy()


def paste_rgba(canvas, rgba, top_left):
    """Alpha-composite an RGBA patch onto a canvas, clipped to it. In place."""
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
    """Exposure, white balance, noise, motion blur and JPEG quality.

    Motion blur especially: live frames have it and no captured plate does, so without
    it the model can key on sharpness to tell synthetic from real - which it cannot do
    at eval, where everything is real and half of it is blurred.
    """
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


def compose(floor_plates, objects, object_dir, finger_plates, target_range, rng):
    """One synthetic frame and its labels.

    The winner - the object the target head is trained to point at - is whichever
    candidate lands nearest the jaws in the image plane. Across many random
    arrangements that teaches the softmax to put a mode on every candidate while the
    cross-entropy target names one, which is the same argument the ortho targeting model
    makes about several objects on a floor.
    """
    frame_w, frame_h = IMAGE_SIZE
    canvas_w, canvas_h = int(frame_w * CANVAS_SCALE), int(frame_h * CANVAS_SCALE)
    offset_x, offset_y = (canvas_w - frame_w) // 2, (canvas_h - frame_h) // 2

    # Prefer a plate captured no closer than the simulated height, so its floor is
    # being magnified rather than tiled; fall back to the nearest if there is none.
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
        scale = ((IMAGE_SIZE[0] / entry.get("capture_width", IMAGE_SIZE[0]))
                 * float(entry["range_m"]) / target_range)
        rgba = cv2.resize(rgba, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        grasp = (entry["grasp_x"] * scale, entry["grasp_y"] * scale)
        top_left = (rng.uniform(-rgba.shape[1] * 0.5, canvas_w - rgba.shape[1] * 0.5),
                    rng.uniform(-rgba.shape[0] * 0.5, canvas_h - rgba.shape[0] * 0.5))
        paste_rgba(canvas, rgba, top_left)
        candidates.append({
            "uv": ((top_left[0] + grasp[0] - offset_x) / frame_w,
                   (top_left[1] + grasp[1] - offset_y) / frame_h),
            # The wrist offset the object was photographed at is how far it is rotated
            # from the ideal grasping angle, so it is the angle to turn back. The sign
            # is the one thing here a real objectplates capture has to confirm.
            # None when the cutout carries no offset - a capture that recorded no ideal
            # angle to measure against - which masks the axis loss rather than teaching
            # the head that every object in the world is aligned.
            "axis": (None if entry.get("wrist_offset_deg") is None
                     else math.radians(float(entry["wrist_offset_deg"]))),
            "label": entry.get("label", ""),
        })

    frame = canvas[offset_y:offset_y + frame_h, offset_x:offset_x + frame_w].copy()

    finger_angle = rng.uniform(*FINGER_ANGLE_RANGE)
    if finger_plates:
        # a set of fingers first, then the aperture nearest the one drawn. Uniform over
        # sets, not over plates, so a capture that swept more angles does not crowd out
        # the colour of the fingers in another.
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
        # The simulated camera height, ignoring the object's own height above the floor.
        # A real approximation, and it biases tall objects; recording object height at
        # capture time and subtracting it is the fix if it shows up in eval.
        "target_range_m": round(target_range, 4) if winner else None,
        "grasp_axis_rad": (round(wrap_half_pi(winner["axis"]), 5)
                           if winner and winner["axis"] is not None else None),
        # No finger label: what a human does with the fingers is not something the
        # compositing knows, and teleop is where that signal lives.
        "finger": None,
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
            # no bar at all, rather than one lying flat: a flat bar is what an axis of
            # exactly zero looks like, and telling those apart is the whole point here
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
             ranges_from=None, annotate_dir=None, annotate_count=40):
    floor_plates = load_floorplates(plate_dir)
    if not floor_plates:
        raise ValueError(f"no floorplates runs in {plate_dir}; nothing to build a background from")
    object_dir = Path(object_dir or Path(plate_dir) / "objects")
    objects = read_objects(object_dir)
    if not objects:
        logging.warning(f"no object cutouts in {object_dir}; every frame will be bare floor")
    finger_plates = load_finger_plates(finger_dir or Path(plate_dir) / "fingers")
    if len(finger_plates) > 1:
        logging.info(f"{len(finger_plates)} sets of fingers; each frame picks one")

    rng = random.Random(seed)
    ranges = sample_ranges(ranges_from, count, rng)

    split_dir = Path(output_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)
    # Only this producer's shards are replaced; the miner's are left where they are.
    for stale in split_dir.glob("synth-*.parquet"):
        stale.unlink()

    if annotate_dir:
        Path(annotate_dir).mkdir(parents=True, exist_ok=True)
        for old in Path(annotate_dir).glob("*.jpg"):
            old.unlink()

    writer = ShardWriter(split_dir, prefix="synth")
    present = 0
    for index, target_range in enumerate(ranges):
        frame, row, candidates = compose(floor_plates, objects, object_dir,
                                         finger_plates, target_range, rng)
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
    return writer.total


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plates", default="plates", help="Directory of capture runs")
    parser.add_argument("--output_root", required=True,
                        help="Mined dataset root; synthetic shards join its split")
    parser.add_argument("--split", default="train", choices=["train", "eval"])
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--object_dir", default=None, help="Cutouts (default <plates>/objects)")
    parser.add_argument("--finger_dir", default=None, help="Finger plates (default <plates>/fingers)")
    parser.add_argument("--ranges_from", default=None,
                        help="A mined split directory to draw simulated heights from")
    parser.add_argument("--annotate_dir", default=None,
                        help="Write annotated sample frames here, to check the labels")
    parser.add_argument("--annotate_count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate(args.plates, args.output_root, args.split, args.count, args.seed,
             args.object_dir, args.finger_dir, args.ranges_from,
             args.annotate_dir, args.annotate_count)


if __name__ == "__main__":
    main()
