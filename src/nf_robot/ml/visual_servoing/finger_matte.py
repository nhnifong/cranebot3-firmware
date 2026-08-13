#!/usr/bin/env python

"""Extract the gripper's fingers, with alpha, from a fingerplates capture.

The capture holds the fingers still and turns the world behind them: the camera is in
the palm and rotates with the wrist, so across one wrist turn the fingers stay on the
same pixels while everything else moves. That makes the separation a per-pixel question
about time rather than a question about colour - no chroma key, no backdrop, no manual
masking:

    median over the turn   what was there when nothing moved -> finger colour
    variance over the turn low where the fingers are, high where the world rotated past

Which is only as good as the ground the capture was taken over. A patch of floor with no
texture, or a soft shadow directly under the lens, does not move in any way the variance
can see, and comes out of this indistinguishable from hardware. The tool reports the
fraction of the frame it called static so that failure is visible rather than silent.

Threshold, morphology and the border test are all offline decisions, revisable against
the same capture - which is why the capture stores raw frames and nothing else.

Usage:
    python -m nf_robot.ml.visual_servoing.finger_matte --dir plates
    python -m nf_robot.ml.visual_servoing.finger_matte --dir plates --contrast_margin 0.25
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.visual_servoing.plates import iter_run, read_manifest

# How much stiller a pixel has to be in the raw stack than in the rotation-aligned one
# to count as hardware, as a fraction of the two added together. Scale-free on purpose:
# an absolute threshold on the raw variation fails at both ends - it keeps floor that
# happens to look the same from every angle, and it throws away the parts of a finger
# whose brightness changes as the gripper turns under a fixed light.
DEFAULT_CONTRAST_MARGIN = 0.15
# Optical axis as a fraction of the frame, from camera_cal_wide: rotation is about the
# lens centre, which is where the principal point is, not the middle of the image.
PRINCIPAL_NORM = (342.0 / 684.0, 192.0 / 384.0)
# Fraction of the frame above which a "static" region is not plausibly the gripper, and
# almost certainly means the background did not move enough to be separable.
IMPLAUSIBLE_STATIC_FRACTION = 0.40
# Connected components smaller than this fraction of the frame are speckle.
MIN_COMPONENT_FRACTION = 0.002
# Radius, in pixels, of the alpha feather. Enough to hide the stair-stepped edge a hard
# threshold leaves, not enough to smear the finger outline.
FEATHER_PX = 2

MANIFEST_NAME = "mattes.jsonl"


def rotate_about(image, degrees, center):
    """Rotate an image about the optical axis, with a validity mask for what fell in.

    Rotating the raw frame is exact rather than approximate: lens distortion here is
    radial, and a radial function is unchanged by rotation about its own centre. So the
    background of one frame really does map onto the background of another, without
    undistorting first.
    """
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)
    warped = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.warpAffine(np.ones((height, width), np.float32), matrix, (width, height),
                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0)
    return warped, valid > 0.5


def aligned_variance(stack, wrist_angles, center, sign):
    """Per-pixel variance after rotating every frame back to the first one's heading.

    The background is what this makes still: rotate each frame by the wrist's change and
    the world lines up, while the fingers - which never moved in the raw frames - are
    smeared around the circle instead. So low variance here means background, which is
    the opposite of what low variance in the raw stack means, and comparing the two
    separates "held still because it is bolted to the camera" from "held still because
    that patch of floor looks the same everywhere".

    Returns (variance, count) with count the number of frames that actually covered each
    pixel; rotation pushes the corners out of frame, so some pixels have few samples.
    """
    total = np.zeros(stack.shape[1:3], np.float64)
    total_sq = np.zeros_like(total)
    count = np.zeros_like(total)
    for frame, angle in zip(stack, wrist_angles):
        delta = sign * (angle - wrist_angles[0])
        warped, valid = rotate_about(frame, delta, center)
        grey = warped.mean(axis=2)
        total += np.where(valid, grey, 0.0)
        total_sq += np.where(valid, grey * grey, 0.0)
        count += valid
    safe = np.maximum(count, 1)
    mean = total / safe
    variance = np.maximum(total_sq / safe - mean * mean, 0.0)
    return variance, count


def alignment_sign(stack, wrist_angles, center):
    """Which rotation direction lines the background up, decided from the frames.

    The wrist angle's sign convention against the image is a fact about the mount that
    is easier to measure here than to derive: whichever direction makes the background
    agree with itself is the right one, by a wide margin.
    """
    scores = {}
    for sign in (1.0, -1.0):
        variance, count = aligned_variance(stack, wrist_angles, center, sign)
        usable = count >= max(3, len(stack) // 2)
        scores[sign] = float(np.sqrt(variance[usable]).mean()) if usable.any() else np.inf
    best = min(scores, key=scores.get)
    return best, scores


def finger_mask(stack, wrist_angles, sign, contrast_margin=DEFAULT_CONTRAST_MARGIN,
                border_only=True, min_component_fraction=MIN_COMPONENT_FRACTION,
                max_brightness=None):
    """Boolean mask of the gripper's own hardware in a stack of (N, H, W, 3) frames.

    The test is a comparison, not a threshold. Each pixel is measured twice: how much it
    varies across the raw frames, and how much it varies once every frame has been
    rotated back to a common heading. Hardware is stiller raw than aligned, background is
    stiller aligned than raw, and

        contrast = (aligned - raw) / (aligned + raw)

    is how much stiller, on a scale that does not care about absolute brightness. That
    matters for both ways an absolute threshold fails here: floor that looks the same
    from every angle never varies much raw, and a finger's brightness changes as the
    gripper turns under a fixed light, so parts of it vary quite a lot.

    Returns (mask, diagnostics). The morphology is deliberately mild: an opening to drop
    speckle, a closing to bridge the gaps that JPEG noise leaves inside a finger, then a
    hole fill, so a highlight in the middle of the hardware does not punch through it.
    """
    raw_std = stack.std(axis=0).mean(axis=2)
    height, width = raw_std.shape
    centre = (width * PRINCIPAL_NORM[0], height * PRINCIPAL_NORM[1])
    variance, count = aligned_variance(stack, wrist_angles, centre, sign)
    aligned_std = np.sqrt(variance)

    contrast = (aligned_std - raw_std) / (aligned_std + raw_std + 1e-3)
    # Rotation pushes the corners out of frame, so some pixels are covered by too few
    # frames for the comparison to mean anything. Those are left out rather than guessed.
    comparable = count >= 3
    raw = comparable & (contrast > contrast_margin)
    if max_brightness is not None:
        # A prior about one capture, not part of the method: over a pale floor with dark
        # hardware, brightness alone separates them, and saying so beats leaving the
        # floor in. Off by default because it is exactly the shortcut the rotation test
        # exists to avoid needing.
        raw &= np.median(stack, axis=0).mean(axis=2) < max_brightness
    area = height * width

    mask = raw.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    components = 0
    for label in range(1, count):
        x, y, w, h, size = stats[label]
        if size < min_component_fraction * area:
            continue
        if border_only:
            # The fingers are attached to a gripper that is itself at the edge of the
            # frame, so anything floating in the middle is scene, not hardware.
            touches = x == 0 or y == 0 or x + w == width or y + h == height
            if not touches:
                continue
        kept[labels == label] = 1
        components += 1

    # Fill enclosed holes, so a specular highlight inside a finger stays part of it. A
    # hole is a component of the *background* that does not reach the frame edge; the
    # obvious flood fill from a corner is wrong here, because a finger that reaches the
    # corner makes the flood a no-op and the whole background reads as one giant hole.
    filled = kept.copy()
    holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        (kept == 0).astype(np.uint8), connectivity=8)
    for label in range(1, holes):
        x, y, w, h, _ = hole_stats[label]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        filled[hole_labels == label] = 1

    diagnostics = {
        "raw_std_median": float(np.median(raw_std)),
        "aligned_std_median": float(np.median(aligned_std[comparable])) if comparable.any() else 0.0,
        "contrast_p90": float(np.percentile(contrast[comparable], 90)) if comparable.any() else 0.0,
        "selected_fraction": float(raw.mean()),
        "kept_fraction": float(filled.mean()),
        "components": components,
    }
    return filled.astype(bool), diagnostics


def build_matte(stack, wrist_angles, sign, contrast_margin=DEFAULT_CONTRAST_MARGIN,
                border_only=True, feather_px=FEATHER_PX, max_brightness=None):
    """One RGBA plate from a stack of (N, H, W, 3) frames of one finger angle."""
    mask, diagnostics = finger_mask(stack, wrist_angles, sign, contrast_margin,
                                    border_only, max_brightness=max_brightness)

    # Median rather than mean: a mean over a rotating background bleeds the world into
    # every edge pixel, where a median throws out anything that was only there part of
    # the time - which is the whole background, by construction.
    colour = np.median(stack, axis=0)

    # Sample colour from inside the mask, so the feathered edge fades out finger colour
    # rather than the average of finger and whatever rotated behind it.
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    if eroded.any():
        for channel in range(3):
            plane = colour[:, :, channel]
            plane[mask & (eroded == 0)] = cv2.blur(plane, (5, 5))[mask & (eroded == 0)]

    alpha = mask.astype(np.float32)
    if feather_px:
        size = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (size, size), 0)

    rgba = np.dstack([
        np.clip(colour, 0, 255).astype(np.uint8),
        np.clip(alpha * 255, 0, 255).astype(np.uint8),
    ])
    return rgba, diagnostics


def group_by_finger_angle(plate_dir, run_id):
    """Frames of one run grouped by finger angle, as {angle: (images, wrist_angles)}.

    Keyed on the commanded angle rather than the measured one so that a group is exactly
    one aperture; the measured value wanders by a fraction of a degree and would split
    every group into singletons.
    """
    groups = defaultdict(lambda: ([], []))
    for row in iter_run(plate_dir, run_id):
        key = row["attrs"].get("commanded_finger_angle")
        if key is None:
            key = round(row["finger_angle"]) if row["finger_angle"] is not None else 0
        images, angles = groups[float(key)]
        images.append(row["image"])
        angles.append(float(row["wrist_angle"] if row["wrist_angle"] is not None else 0.0))
    return dict(sorted(groups.items()))


def extract_mattes(plate_dir, run_id, output_dir, contrast_margin=DEFAULT_CONTRAST_MARGIN,
                   border_only=True, feather_px=FEATHER_PX, max_brightness=None):
    """Write one RGBA plate per finger angle, plus a manifest line each."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_by_finger_angle(plate_dir, run_id)
    logging.info(f"{run_id}: {len(groups)} finger angles, "
                 f"{sum(len(v[0]) for v in groups.values())} frames")

    # One direction for the whole run: it is a property of the mount, and deciding it per
    # group would let a bad group flip it and quietly matte the background instead.
    first = next(iter(groups.values()))
    probe = np.stack(first[0]).astype(np.float32)
    height, width = probe.shape[1:3]
    sign, scores = alignment_sign(
        probe, first[1], (width * PRINCIPAL_NORM[0], height * PRINCIPAL_NORM[1]))
    logging.info(f"alignment sign {sign:+.0f} (aligned std {scores[sign]:.1f} vs "
                 f"{scores[-sign]:.1f} the other way)")

    entries = []
    for finger_angle, (frames, wrist_angles) in groups.items():
        if len(frames) < 3:
            logging.warning(f"finger {finger_angle:+.0f}: only {len(frames)} frames, skipping")
            continue
        stack = np.stack(frames).astype(np.float32)
        rgba, diagnostics = build_matte(stack, wrist_angles, sign, contrast_margin,
                                        border_only, feather_px, max_brightness)

        name = f"finger{finger_angle:+04.0f}.png"
        # cv2 writes BGRA; the plates are RGB
        cv2.imwrite(str(output_dir / name), rgba[:, :, [2, 1, 0, 3]])

        entry = {
            "file": name, "run_id": run_id, "finger_angle": finger_angle,
            "frames": len(frames), "width": rgba.shape[1], "height": rgba.shape[0],
            "contrast_margin": contrast_margin, "border_only": border_only,
            **diagnostics,
        }
        entries.append(entry)
        flag = ""
        if diagnostics["kept_fraction"] > IMPLAUSIBLE_STATIC_FRACTION:
            flag = "  <- implausible, the background cannot have moved enough"
        elif diagnostics["components"] == 0:
            flag = "  <- nothing kept"
        logging.info(
            f"finger {finger_angle:+4.0f}: {len(frames):3d} frames, selected "
            f"{diagnostics['selected_fraction'] * 100:5.1f}% -> kept "
            f"{diagnostics['kept_fraction'] * 100:5.1f}% in {diagnostics['components']} "
            f"component(s), std raw {diagnostics['raw_std_median']:.1f} vs aligned "
            f"{diagnostics['aligned_std_median']:.1f}{flag}")

    with open(output_dir / MANIFEST_NAME, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return entries


def write_preview(output_dir, entries, cell_width=480, columns=5):
    """Contact sheet of the mattes over a checkerboard, so the alpha is visible."""
    output_dir = Path(output_dir)
    cells = []
    for entry in entries:
        bgra = cv2.imread(str(output_dir / entry["file"]), cv2.IMREAD_UNCHANGED)
        scale = cell_width / bgra.shape[1]
        bgra = cv2.resize(bgra, (cell_width, int(round(bgra.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
        height, width = bgra.shape[:2]

        square = 16
        ys, xs = np.mgrid[0:height, 0:width]
        board = np.where(((ys // square) + (xs // square)) % 2 == 0, 110, 70).astype(np.uint8)
        board = np.dstack([board] * 3)

        alpha = (bgra[:, :, 3:4].astype(np.float32) / 255.0)
        cell = (bgra[:, :, :3].astype(np.float32) * alpha
                + board.astype(np.float32) * (1 - alpha)).astype(np.uint8)

        text = (f"{entry['finger_angle']:+.0f}deg  {entry['kept_fraction'] * 100:.1f}% "
                f"{entry['components']}c")
        cv2.putText(cell, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(cell, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cells.append(cell)

    if not cells:
        return None
    h, w = cells[0].shape[:2]
    blank = np.full((h, w, 3), 25, dtype=np.uint8)
    cells = cells + [blank] * (-len(cells) % columns)
    sheet = np.vstack([np.hstack(cells[r:r + columns]) for r in range(0, len(cells), columns)])
    path = output_dir / "_mattes.png"
    cv2.imwrite(str(path), sheet)
    logging.info(f"wrote {path}")
    return path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="plates", help="Directory of capture runs")
    parser.add_argument("--run_id", default=None, help="Which run; defaults to the newest")
    parser.add_argument("--output_dir", default=None,
                        help="Where the RGBA plates go (default <dir>/fingers)")
    parser.add_argument("--contrast_margin", type=float, default=DEFAULT_CONTRAST_MARGIN,
                        help="How much stiller raw than aligned a pixel must be, 0-1")
    parser.add_argument("--max_brightness", type=float, default=None,
                        help="Reject pixels brighter than this (0-255). A prior about "
                             "one capture - use it when the floor is pale and the "
                             "hardware dark - not part of the method.")
    parser.add_argument("--keep_floating", action="store_true",
                        help="Keep static regions that do not touch the frame edge")
    parser.add_argument("--feather_px", type=int, default=FEATHER_PX)
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    runs = [r for r in read_manifest(args.dir) if r["kind"] == "fingerplates"]
    if not runs:
        parser.error(f"no fingerplates runs in {args.dir}")
    run_id = args.run_id or runs[-1]["run_id"]
    output_dir = Path(args.output_dir or Path(args.dir) / "fingers")

    entries = extract_mattes(args.dir, run_id, output_dir, args.contrast_margin,
                             not args.keep_floating, args.feather_px, args.max_brightness)
    if entries and not args.no_preview:
        write_preview(output_dir, entries)


if __name__ == "__main__":
    main()
