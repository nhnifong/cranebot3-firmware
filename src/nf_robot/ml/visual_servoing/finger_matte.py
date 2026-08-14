#!/usr/bin/env python

"""Extract the gripper's fingers, with alpha, from a fingerplates capture.

A chroma key against the green backdrop the capture is taken over, the same keyer the
object cutouts use: greenness is G - max(R, B), which needs no colour space conversion
and holds up in shadow, where a hue threshold in HSV gets unreliable as saturation falls.

The wrist turn in the capture is still doing work. Each frame is keyed on its own and the
per-pixel median taken across the turn, so the fingers - the only ungreen thing that is in
the same place in every frame - survive, while anything that rotated past underneath is
outvoted rather than matted in.

One plate per finger angle. Threshold, morphology and the border test are all offline
decisions, revisable against the same capture, which is why the capture stores raw frames
and nothing else.

Usage:
    python -m nf_robot.ml.visual_servoing.finger_matte --dir plates
    python -m nf_robot.ml.visual_servoing.finger_matte --dir plates --green_low 6
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.visual_servoing.object_matte import GREEN_HIGH, GREEN_LOW, chroma_key
from nf_robot.ml.visual_servoing.plates import (
    VIDEO_FPS, checkerboard, iter_run, over_checkerboard, read_manifest, write_video)

# Connected components smaller than this fraction of the frame are speckle.
MIN_COMPONENT_FRACTION = 0.002
# Fraction of the frame above which what was kept is not plausibly the gripper, and almost
# certainly means the backdrop was not green enough to key.
IMPLAUSIBLE_KEPT_FRACTION = 0.40
# Fraction of the frame that has to be decisively green before a capture is worth keying
# at all. Below it the backdrop was missing or badly lit, which is worth saying out loud.
MIN_GREEN_FRACTION = 0.20

MANIFEST_NAME = "mattes.jsonl"


def green_fraction(image, green_high=GREEN_HIGH):
    """Fraction of a frame that is decisively green, which is how a backdrop is checked."""
    image = image.astype(np.float32)
    greenness = image[:, :, 1] - np.maximum(image[:, :, 0], image[:, :, 2])
    return float((greenness > green_high).mean())


def clean_mask(raw, border_only=True, min_component_fraction=MIN_COMPONENT_FRACTION):
    """Speckle, floaters and holes.

    Returns (filled, kept, components): kept is what survived the component tests, filled
    is that with enclosed holes closed, so a caller can tell a hole from hardware.

    The morphology is deliberately mild - an opening to drop speckle, a closing to bridge
    the gaps JPEG noise leaves inside a finger - and holes are found as components of the
    background rather than by a flood from a corner, which does nothing when a finger
    reaches that corner and reads the whole background as one giant hole.
    """
    height, width = raw.shape
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

    filled = kept.copy()
    holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(
        (kept == 0).astype(np.uint8), connectivity=8)
    for label in range(1, holes):
        x, y, w, h, _ = hole_stats[label]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        filled[hole_labels == label] = 1

    return filled.astype(bool), kept.astype(bool), components


def build_matte(stack, green_low=GREEN_LOW, green_high=GREEN_HIGH, border_only=True,
                min_component_fraction=MIN_COMPONENT_FRACTION):
    """One RGBA plate from a stack of (N, H, W, 3) frames of one finger angle."""
    alphas, colours, greens = [], [], []
    for frame in stack:
        alpha, colour = chroma_key(frame, green_low=green_low, green_high=green_high)
        alphas.append(alpha)
        colours.append(colour)
        # measured on the way in: chroma_key's despill has taken the cast out of what it
        # returns, so the backdrop is no longer green by the time the colour comes back
        greens.append(green_fraction(frame, green_high))
    # Median over the turn, so a pixel has to have been ungreen most of the time. Mean
    # would let one frame's intruder leave a ghost at a third of its opacity.
    alpha = np.median(np.stack(alphas), axis=0)
    colour = np.median(np.stack(colours), axis=0)

    filled, kept, components = clean_mask(alpha > 0.5, border_only, min_component_fraction)
    # Keep the ramp wherever it is - it is what puts a soft edge on fluff and on the frayed
    # rubber of a finger pad - but lift enclosed holes to solid, since a green-lit
    # highlight inside a finger keys as backdrop and is not one.
    out = alpha * np.maximum(filled, (alpha > 0) & (alpha <= 0.5))
    out = np.where(filled & ~kept, 1.0, out)

    rgba = np.dstack([
        np.clip(colour, 0, 255).astype(np.uint8),
        np.clip(out * 255, 0, 255).astype(np.uint8),
    ])
    diagnostics = {
        "green_fraction": float(np.mean(greens)),
        "selected_fraction": float((alpha > 0.5).mean()),
        "kept_fraction": float(filled.mean()),
        "components": components,
    }
    return rgba, diagnostics


def group_by_finger_angle(plate_dir, run_id):
    """Frames of one run grouped by finger angle, as {angle: [images]}.

    Keyed on the commanded angle rather than the measured one so that a group is exactly
    one aperture; the measured value wanders by a fraction of a degree and would split
    every group into singletons.
    """
    groups = defaultdict(list)
    for row in iter_run(plate_dir, run_id):
        key = row["attrs"].get("commanded_finger_angle")
        if key is None:
            key = round(row["finger_angle"]) if row["finger_angle"] is not None else 0
        groups[float(key)].append(row["image"])
    return dict(sorted(groups.items()))


def extract_mattes(plate_dir, run_id, output_dir, green_low=GREEN_LOW,
                   green_high=GREEN_HIGH, border_only=True):
    """Write one RGBA plate per finger angle, plus a manifest line each."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_by_finger_angle(plate_dir, run_id)
    logging.info(f"{run_id}: {len(groups)} finger angles, "
                 f"{sum(len(v) for v in groups.values())} frames")

    entries = []
    for finger_angle, frames in groups.items():
        if not frames:
            continue
        stack = np.stack(frames).astype(np.float32)
        rgba, diagnostics = build_matte(stack, green_low, green_high, border_only)

        name = f"finger{finger_angle:+04.0f}.png"
        # cv2 writes BGRA; the plates are RGB
        cv2.imwrite(str(output_dir / name), rgba[:, :, [2, 1, 0, 3]])

        entries.append({
            "file": name, "run_id": run_id, "finger_angle": finger_angle,
            "frames": len(frames), "width": rgba.shape[1], "height": rgba.shape[0],
            "green_low": green_low, "green_high": green_high, "border_only": border_only,
            **diagnostics,
        })
        flag = ""
        if diagnostics["green_fraction"] < MIN_GREEN_FRACTION:
            flag = "  <- barely any green; was this shot over the backdrop?"
        elif diagnostics["kept_fraction"] > IMPLAUSIBLE_KEPT_FRACTION:
            flag = "  <- implausible, too much of the frame keyed as hardware"
        elif diagnostics["components"] == 0:
            flag = "  <- nothing kept"
        logging.info(
            f"finger {finger_angle:+4.0f}: {len(frames):3d} frames, "
            f"{diagnostics['green_fraction'] * 100:5.1f}% green, keyed "
            f"{diagnostics['selected_fraction'] * 100:5.1f}% -> kept "
            f"{diagnostics['kept_fraction'] * 100:5.1f}% in "
            f"{diagnostics['components']} component(s){flag}")

    with open(output_dir / MANIFEST_NAME, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return entries


def write_video_preview(output_dir, entries, fps=VIDEO_FPS):
    """The plates as an mp4, in finger angle order: the aperture sweep as it will look.

    A still contact sheet hides the failure this catches - a plate whose matte is a few
    pixels different from its neighbours' reads as a flicker in motion and as nothing at
    all side by side.
    """
    output_dir = Path(output_dir)
    board = None

    def frames():
        nonlocal board
        for entry in sorted(entries, key=lambda e: e["finger_angle"]):
            bgra = cv2.imread(str(output_dir / entry["file"]), cv2.IMREAD_UNCHANGED)
            if bgra is None:
                continue
            if board is None or board.shape[:2] != bgra.shape[:2]:
                board = checkerboard(bgra.shape[0], bgra.shape[1])
            cell = over_checkerboard(bgra, board)
            text = f"{entry['finger_angle']:+.0f}deg"
            cv2.putText(cell, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(cell, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            yield cell

    return write_video(output_dir / "_mattes.mp4", frames(), fps)


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
    parser.add_argument("--green_low", type=float, default=GREEN_LOW,
                        help="Greenness below which a pixel is certainly not backdrop")
    parser.add_argument("--green_high", type=float, default=GREEN_HIGH,
                        help="Greenness above which a pixel is certainly backdrop")
    parser.add_argument("--keep_floating", action="store_true",
                        help="Keep kept regions that do not touch the frame edge")
    parser.add_argument("--fps", type=int, default=VIDEO_FPS,
                        help="Playback rate of the mp4 written beside the plates")
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    runs = [r for r in read_manifest(args.dir) if r["kind"] == "fingerplates"]
    if not runs:
        parser.error(f"no fingerplates runs in {args.dir}")
    run_id = args.run_id or runs[-1]["run_id"]
    output_dir = Path(args.output_dir or Path(args.dir) / "fingers")

    entries = extract_mattes(args.dir, run_id, output_dir, args.green_low,
                             args.green_high, not args.keep_floating)
    if entries and not args.no_preview:
        write_preview(output_dir, entries)
        write_video_preview(output_dir, entries, args.fps)


if __name__ == "__main__":
    main()
