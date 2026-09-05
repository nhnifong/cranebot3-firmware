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
# How much extra zoom a floor plate may be given on top of the scale its range implies.
# Inwards only, and only a little. Zooming out would need floor outside what was
# captured - which is what the tiling used to invent - and a large zoom in would tell the
# model the floor is nearer than the range label it is being trained against says.
FLOOR_ZOOM_MAX = 1.12
# Sign taking a cutout's measured wrist offset to the grasp axis in the image.
#
# The axis label is an image-plane direction - the line the jaws close along - because
# that is the only thing a network looking at one frame can be asked for. It is drawn
# perpendicular to the object's long axis, and it reads zero (a horizontal bar) when the
# object stands upright in frame, which is the orientation the servoing is steering to.
#
# The offset it comes from is a wrist angle, and the camera turns with the wrist: a wrist
# turned +30 degrees off ideal photographs the object rotated 30 degrees the *other* way
# in the image. So the image-plane angle is the negative of the wrist offset. Flip this
# to +1 if annotated frames show the bar mirrored about the horizontal - it is a fact
# about the mount, and one look at a long object settles it.
AXIS_FROM_WRIST_SIGN = -1.0
# (MB) how much decoded floor plate to hold. This is the tool's high water mark by a wide
# margin - every other structure here is either streaming or a manifest - so it is the
# number to move when the machine is smaller or larger, not a thing to discover by
# watching the OOM killer.
FLOOR_CACHE_MB = 2048
# (MB) how much image data a synthetic shard buffers before it is written. The buffer is
# copied into an arrow table on the way out, so the transient cost is about twice this.
SHARD_MB = 256
# (metres) the simulated heights this tool is willing to composite, whatever a recording
# says. The mined rangefinder column bottoms out at 0.001m - 1.5% of rows are under 1cm -
# which is the sensor in contact rather than a camera looking at a floor. Composited
# literally, 0.001m asks for a floor plate magnified 283x: a 271936x152964 image, 125GB in
# one allocation, and the end of the run. The ceiling is the same idea from the other end,
# where a plate shrinks past the point of carrying any texture.
SIM_RANGE_M = (0.05, 1.5)
# Hard cap on magnifying any plate or cutout, as a backstop that bounds the largest
# intermediate array whatever ranges arrive. With SIM_RANGE_M in force the worst real case
# is about 5.7x, so this never binds in normal operation - it exists so that a bad capture
# range cannot allocate the machine.
MAX_MAGNIFICATION = 6.0


def neutralize_plates(entries, what):
    """Take one capture's colour cast out of its plates, in place.

    Returns the illuminant that was removed, so a caller holding sources that cannot be
    measured on their own - a cutout is mostly object, and the object's colour is not the
    light's - can borrow it.
    """
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
    """A bounded pool of floorplate frames, with the range each was captured at.

    Kept decoded, because the compositor picks a different plate for every synthetic frame
    and seeking into an h264 capture per frame would cost more than the composite does.

    Bounded, because decoded frames are enormous and the pool is the peak of this whole
    tool: one 960x540 frame is 1.56MB, and eleven runs of ~1,700 frames each is 28GB of
    resident numpy - enough to have the OOM killer end a 40,000 frame run before a single
    shard is written, which is exactly what it did. The budget caps that at something a
    machine can hold, and everything else here is streaming.

    Which frames make the cut matters as much as how many. A run steps through heights and
    turns the wrist at each one, so the first N frames of a run are all one height: taking
    the front of the run would quietly narrow the background distribution to the lowest
    capture. Each run is reservoir sampled instead, so every frame of the sweep has the
    same chance of being in the pool however long the run turns out to be - no frame count
    is needed in advance, which is just as well, since the manifest's `samples` undercounts
    the frames a video run decodes to by over 10%.
    """
    runs = [r for r in read_manifest(plate_dir) if r["kind"] == "floorplates"]
    if limit_runs:
        runs = runs[-limit_runs:]
    if not runs:
        return []

    # One frame decoded up front, only to find out what a frame costs here; capture
    # resolution is a property of the run, not something this tool should assume.
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
                # standard reservoir replacement; anything not kept is freed as the
                # generator moves on, so the high water mark is the pool itself
                index = rng.randrange(seen)
                if index < quota:
                    kept[index] = entry
        # Per run, because a run is one session under one set of room lights, and two runs
        # shot at different times of day are not the same yellow.
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
    """The illuminant of whichever measured capture sits closest in time to run_id.

    For sources whose own pixels cannot be measured: the light in the house is what these
    all have in common, and the capture nearest in time is the best record of it there is.
    """
    when = run_time(run_id)
    dated = {other: value for other, value in measured.items() if run_time(other)}
    if not dated:
        return np.ones(3)
    if when is None:
        return next(iter(dated.values()))
    return dated[min(dated, key=lambda other: abs(run_time(other) - when))]


def cutout_gains(entries, floor_illuminants):
    """Neutralizing gains per objectplates run, borrowed from the floorplates runs.

    The cutouts cannot be measured on their own. A cutout is object and nothing else, and
    assorted objects do not average to grey the way a room does: measured that way this
    set asks for a 4.6x blue gain, which is the toys being warm-coloured rather than the
    light being warm. The floor captured nearest in time is the same house under the same
    pinned preset, and it has a room's worth of surfaces to average over.
    """
    gains = {}
    for run_id in sorted({entry.get("run_id", "") for entry in entries}):
        illuminant = nearest_illuminant(run_id, floor_illuminants)
        gains[run_id] = white_balance.neutralize_gains(illuminant)
        logging.info(f"cutouts from {run_id or 'unknown run'}: neutralized by "
                     f"{np.round(gains[run_id], 3)}, borrowed from the nearest floor capture")
    return gains


def load_finger_plates(finger_dir, neutralize=True):
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
        # Downscaled once here rather than per composite: a finger plate is always pasted
        # over the whole frame at exactly this size, so anything larger is memory that is
        # thrown away every time it is used. A capture-resolution set of plates is 2MB
        # each; at frame size they are a quarter of that.
        if (bgra.shape[1], bgra.shape[0]) != IMAGE_SIZE:
            bgra = cv2.resize(bgra, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        by_run.setdefault(entry.get("run_id", ""), []).append(
            (float(entry["finger_angle"]), bgra[:, :, [2, 1, 0, 3]]))
    for run_id, plates in sorted(by_run.items()):
        # Per run again: a run is one set of fingers under one set of lights, and the sets
        # really are different colours, so pooling them would read a white set as the
        # light and turn a blue set bluer.
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
    """A rescale factor, clamped to something that cannot allocate the machine.

    Logged when it binds, because it should not: SIM_RANGE_M is what keeps the inputs
    sane, and this firing means a plate or a cutout carries a range that the range clamp
    did not cover.
    """
    if scale <= MAX_MAGNIFICATION:
        return scale
    logging.warning(f"{what}: magnification {scale:.1f}x capped at {MAX_MAGNIFICATION}x; "
                    f"check the capture range on this plate")
    return MAX_MAGNIFICATION


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
            drawn = [float(rng.choice(values)) for _ in range(count)]
            return clamp_ranges(drawn)
        logging.warning(f"no ranges found under {dataset_root}; falling back to log-uniform")
    low, high = math.log(RANGE_MIN_M), math.log(RANGE_MAX_M)
    return clamp_ranges([math.exp(rng.uniform(low, high)) for _ in range(count)])


def clamp_ranges(values):
    """Simulated heights held inside SIM_RANGE_M, reporting how many had to move.

    A recorded rangefinder reading is not automatically a height worth simulating: the
    sensor reads 0.001m while the fingers are closing on something, and that number
    composites into an allocation no machine has. Clamping rather than dropping keeps the
    height distribution the mined data asked for everywhere it is meaningful.
    """
    low, high = SIM_RANGE_M
    clamped = [min(max(v, low), high) for v in values]
    moved = sum(1 for v, c in zip(values, clamped) if v != c)
    if moved:
        logging.info(f"{moved}/{len(values)} simulated heights clamped into "
                     f"{low}-{high}m; readings outside it are the sensor in contact or "
                     f"out of range rather than a view of a floor")
    return clamped


def floor_canvas(plate, target_range, canvas_size, rng):
    """A floor plate rescaled to the simulated height and cropped to the canvas.

    Two scalings, and forgetting either puts the floor at the wrong size. The plate was
    captured at some resolution of the same field of view the model input covers, so it
    first has to be scaled by the ratio between them; then a plate captured at r0 and
    viewed from r covers r0/r as much floor per pixel.

    That product regularly lands short of the frame - whenever nothing was captured above
    the simulated height, and by a couple of percent anyway because the plate's aspect
    ratio is not exactly the model input's. Filling the shortfall by tiling is what put a
    seam through the middle of every frame: repeated floor is not something any camera can
    see, and the model would have been free to learn the repeat as a feature. So the scale
    is floored at what covers the frame and then given a small random zoom that only ever
    goes inwards, and the crop moves within the slack that zoom leaves. Every pixel of the
    result is floor that was photographed exactly once.

    Being magnified past r0/r does mean the texture looks nearer than the range label
    says. The alternative is inventing floor; generate() counts how often it happens,
    because the fix is capturing floorplates from higher up rather than anything here.

    The canvas margin outside the frame is edge-replicated. Objects are pasted in canvas
    coordinates and the frame is cut out of the middle, so nothing out there is ever
    rendered - the margin exists to give an off-frame object somewhere to land.
    """
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
    """Colour temperature, exposure, white balance, noise, motion blur and JPEG quality.

    Motion blur especially: live frames have it and no captured plate does, so without
    it the model can key on sharpness to tell synthetic from real - which it cannot do
    at eval, where everything is real and half of it is blurred.

    The temperature comes first because it is the light in the room, not something the
    camera did: the ingredients have each been neutralized on the way in, so this is what
    puts a cast back, and it re-lights the whole frame at once the way a room does. The
    per-channel jitter below stays, on top of it, for everything that is the camera.
    """
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
    """A cutout's measured wrist offset as the grasp axis in the image, in radians.

    None passes through as None - an unlabelled capture masks the axis loss instead of
    claiming the object is upright.
    """
    if offset_deg is None:
        return None
    return math.radians(AXIS_FROM_WRIST_SIGN * float(offset_deg))


def compose(floor_plates, objects, object_dir, finger_plates, target_range, rng,
            object_gains=None):
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

    # Prefer a plate captured no closer than the simulated height. Magnifying it by r0/r
    # is then the honest transform, and floor_canvas has to magnify further than that -
    # putting the texture at the wrong scale - only when nothing was captured high enough.
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
            # The cutout was photographed with the wrist this far off ideal, which is the
            # same thing as the object being that far off upright in the image.
            "axis": axis_from_wrist_offset(entry.get("wrist_offset_deg")),
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
        # compositing knows, and teleop is where that signal lives. Same for when a close
        # would have started, how hard it would have ended up squeezing, and how wide the
        # jaws would have been spread to go around the object.
        "finger": None,
        "close_now": None,
        "grasp_pressure": None,
        "open_angle": None,
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
             ranges_from=None, annotate_dir=None, annotate_count=40,
             floor_cache_mb=FLOOR_CACHE_MB, shard_mb=SHARD_MB):
    # Every ingredient is neutralized as it loads, each measured against its own capture,
    # so the floor, the objects and the fingers in a composite agree on what white is.
    # photometric then re-lights the finished frame at one random colour temperature, the
    # way a room lights everything in it at once.
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

    # Above the tallest plate there is no floor captured wide enough to fill the frame, so
    # floor_canvas magnifies past r0/r and the texture comes out looking nearer than the
    # label says. Worth knowing how much of the output that is, since the fix is a capture
    # run from higher up rather than anything this file can do.
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
    # Measured rather than predicted, because the budget above only bounds the plate pool
    # and this is the number that decides whether the run survives on this machine.
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
