#!/usr/bin/env python

"""Chroma-key objectplates captures into RGBA cutouts with grasp point and axis labels.

Usage:
    python -m nf_robot.ml.visual_servoing.object_matte --dir plates
    python -m nf_robot.ml.visual_servoing.object_matte --dir plates --run_id objectplates-...
"""

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.visual_servoing.plates import (
    VIDEO_FPS, checkerboard, iter_run, over_checkerboard, read_manifest, write_video)

# Greenness thresholds between which the alpha ramps for a soft edge.
GREEN_HIGH = 40.0
GREEN_LOW = 10.0
# Connected components smaller than this fraction of the frame are speckle.
MIN_COMPONENT_FRACTION = 0.0008
# Margin in pixels left around the object's bounding box when cropping.
CROP_MARGIN = 8
# Optical axis as a fraction of the frame, where the operator centred the grasp point.
PRINCIPAL_NORM = (342.0 / 684.0, 192.0 / 384.0)

# Floor diameter around the grasp point beyond which keyed pixels are board edge or floor,
# not object.
VIGNETTE_DIAMETER_M = 0.5
# Normalized focal length from camera_cal_wide, valid at any capture resolution.
FOCAL_NORM = (439.31834658631243 / 684.0, 461.5621083718772 / 384.0)

MANIFEST_NAME = "objects.jsonl"


def chroma_key(rgb, green_low=GREEN_LOW, green_high=GREEN_HIGH):
    """Alpha from greenness (G - max(R, B)), plus the colour with green spill pulled out."""
    image = rgb.astype(np.float32)
    red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    greenness = green - np.maximum(red, blue)

    alpha = (green_high - greenness) / max(green_high - green_low, 1e-6)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Clamp G to the average of R and B to remove the backdrop's green cast.
    despilled = image.copy()
    spill = greenness > 0
    despilled[:, :, 1] = np.where(spill, np.minimum(green, (red + blue) / 2.0), green)
    return alpha, np.clip(despilled, 0, 255).astype(np.uint8)


def clean_alpha(alpha, min_component_fraction=MIN_COMPONENT_FRACTION):
    """Drop speckle, keep everything else - holes included."""
    solid = (alpha > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    area = alpha.shape[0] * alpha.shape[1]
    keep = np.zeros_like(solid)
    for label in range(1, count):
        if stats[label][4] >= min_component_fraction * area:
            keep[labels == label] = 1
    # Only zero the rejected specks; the ramp on every kept edge is left alone.
    return alpha * np.maximum(keep, (alpha > 0) & (alpha <= 0.5))


def vignette_axes(shape, range_m, diameter_m=VIGNETTE_DIAMETER_M):
    """Pixel semi-axes that VIGNETTE_DIAMETER_M projects to at range_m (fx and fy differ)."""
    height, width = shape[:2]
    radius = diameter_m / 2.0 / max(range_m, 1e-6)
    return FOCAL_NORM[0] * width * radius, FOCAL_NORM[1] * height * radius


def apply_vignette(alpha, range_m, diameter_m=VIGNETTE_DIAMETER_M):
    """Zero alpha outside the keep-region centred on the grasp point."""
    height, width = alpha.shape
    ax, ay = vignette_axes(alpha.shape, range_m, diameter_m)
    cx, cy = PRINCIPAL_NORM[0] * width, PRINCIPAL_NORM[1] * height
    ys, xs = np.ogrid[:height, :width]
    inside = ((xs - cx) / ax) ** 2 + ((ys - cy) / ay) ** 2 <= 1.0
    return np.where(inside, alpha, 0.0)


def extract_cutout(rgb, margin=CROP_MARGIN, range_m=None,
                   diameter_m=VIGNETTE_DIAMETER_M, **key_kwargs):
    """One frame as a tight RGBA cutout, returning (rgba, grasp_xy, coverage) or None."""
    alpha, colour = chroma_key(rgb, **key_kwargs)
    if range_m is not None:
        # Before the speckle pass, so leftover board edge is judged at its final size.
        alpha = apply_vignette(alpha, range_m, diameter_m)
    alpha = clean_alpha(alpha)
    if not (alpha > 0.5).any():
        return None

    ys, xs = np.nonzero(alpha > 0.5)
    height, width = alpha.shape
    x0 = max(0, xs.min() - margin)
    x1 = min(width, xs.max() + 1 + margin)
    y0 = max(0, ys.min() - margin)
    y1 = min(height, ys.max() + 1 + margin)

    rgba = np.dstack([colour[y0:y1, x0:x1],
                      (alpha[y0:y1, x0:x1] * 255).astype(np.uint8)])
    grasp = (PRINCIPAL_NORM[0] * width - x0, PRINCIPAL_NORM[1] * height - y0)
    return rgba, grasp, float((alpha > 0.5).mean())


def wrist_offset_deg(wrist_angle, grasp_axis_deg):
    """Degrees a frame's view of the object is turned from the ideal grasping angle, or None
    if unknown."""
    if wrist_angle is None or grasp_axis_deg is None:
        return None
    return (float(wrist_angle) - float(grasp_axis_deg) + 180.0) % 360.0 - 180.0


def extract_run(plate_dir, run_id, output_dir, label=None,
                diameter_m=VIGNETTE_DIAMETER_M, vignette=True, **key_kwargs):
    """Write one RGBA cutout per frame of an objectplates run, plus a manifest line each."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # The starting wrist angle, set by the operator to the ideal grasping angle.
    run = next((e for e in read_manifest(plate_dir) if e["run_id"] == run_id), {})
    grasp_axis = run.get("grasp_axis_wrist_angle", run.get("start_wrist_angle"))
    if grasp_axis is None:
        logging.warning(f"{run_id}: no grasp_axis_wrist_angle; every cutout's grasp axis "
                        f"will be unlabelled and synthetic frames built from them will "
                        f"train the axis head on nothing")

    entries, skipped = [], 0
    for index, row in enumerate(iter_run(plate_dir, run_id)):
        attrs = row["attrs"]
        # Measured range where there is one, else the sweep's target height.
        range_m = row["laser_rangefinder"] or attrs.get("target_range_m")
        result = extract_cutout(row["image"], diameter_m=diameter_m,
                                range_m=range_m if vignette else None, **key_kwargs)
        if result is None:
            skipped += 1
            continue
        rgba, grasp, coverage = result
        name = f"{run_id}-{index:04d}.png"
        cv2.imwrite(str(output_dir / name), rgba[:, :, [2, 1, 0, 3]])

        entries.append({
            "file": name,
            "run_id": run_id,
            "label": label or attrs.get("label", ""),
            # Measured per frame from wrist telemetry against the run's zero.
            "wrist_offset_deg": attrs.get(
                "wrist_offset_deg", wrist_offset_deg(row.get("wrist_angle"), grasp_axis)),
            "range_m": range_m,
            "grasp_x": round(float(grasp[0]), 2),
            "grasp_y": round(float(grasp[1]), 2),
            "width": int(rgba.shape[1]),
            "height": int(rgba.shape[0]),
            # The source frame size, needed to scale the cutout at composite time.
            "capture_width": int(row["image"].shape[1]),
            "capture_height": int(row["image"].shape[0]),
            # The vignette size in pixels, for spotting over-cropped runs.
            "vignette_px": (None if not (vignette and range_m) else
                            [round(2 * a, 1) for a in
                             vignette_axes(row["image"].shape, range_m, diameter_m)]),
            "coverage": round(coverage, 4),
        })

    manifest = output_dir / MANIFEST_NAME
    existing = [json.loads(line) for line in open(manifest)] if manifest.exists() else []
    existing = [e for e in existing if e["run_id"] != run_id]
    with open(manifest, "w") as f:
        for entry in existing + entries:
            f.write(json.dumps(entry) + "\n")

    if not entries:
        logging.info(f"{run_id}: nothing extracted")
        return entries

    # Report the axis label spread, since one constant value would train the axis head to
    # answer it.
    offsets = [e["wrist_offset_deg"] for e in entries if e["wrist_offset_deg"] is not None]
    axis = (f"grasp axis {min(offsets):+.0f}..{max(offsets):+.0f} deg" if offsets
            else "grasp axis UNLABELLED")
    if offsets and max(offsets) - min(offsets) < 1.0:
        axis += "  <- the wrist barely moved; the axis head has nothing to learn here"
    logging.info(f"{run_id}: {len(entries)} cutouts, {skipped} frames keyed to nothing; "
                 f"median coverage {np.median([e['coverage'] for e in entries]) * 100:.1f}%, "
                 f"{axis}")
    return entries


def read_objects(output_dir):
    """Every cutout in a directory, as manifest rows."""
    manifest = Path(output_dir) / MANIFEST_NAME
    if not manifest.exists():
        return []
    return [json.loads(line) for line in open(manifest) if line.strip()]


def write_video_preview(output_dir, entries, fps=VIDEO_FPS):
    """The cutouts as an mp4, each pasted back where it sat in its capture frame."""
    output_dir = Path(output_dir)

    def frames():
        board = None
        for entry in entries:
            bgra = cv2.imread(str(output_dir / entry["file"]), cv2.IMREAD_UNCHANGED)
            if bgra is None:
                continue
            capture = (int(entry["capture_height"]), int(entry["capture_width"]))
            if board is None or board.shape[:2] != capture:
                board = checkerboard(*capture)
            canvas = np.zeros((*capture, 4), np.uint8)
            # The crop's offset in the capture frame, recovered from the grasp point's
            # position in the cutout.
            x0 = int(round(PRINCIPAL_NORM[0] * capture[1] - entry["grasp_x"]))
            y0 = int(round(PRINCIPAL_NORM[1] * capture[0] - entry["grasp_y"]))
            x0 = max(0, min(x0, capture[1] - bgra.shape[1]))
            y0 = max(0, min(y0, capture[0] - bgra.shape[0]))
            canvas[y0:y0 + bgra.shape[0], x0:x0 + bgra.shape[1]] = bgra

            cell = over_checkerboard(canvas, board)
            text = f"{entry['label']}  {entry['range_m'] or 0:.2f}m"
            cv2.putText(cell, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(cell, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            yield cell

    return write_video(output_dir / "_objects.mp4", frames(), fps)


def write_preview(output_dir, entries, cell_width=240, columns=8, limit=32):
    """Contact sheet over a checkerboard, so the keyed alpha is visible."""
    output_dir = Path(output_dir)
    cells = []
    for entry in entries[:limit]:
        bgra = cv2.imread(str(output_dir / entry["file"]), cv2.IMREAD_UNCHANGED)
        scale = cell_width / max(bgra.shape[1], 1)
        bgra = cv2.resize(bgra, (cell_width, max(1, int(round(bgra.shape[0] * scale)))))
        height, width = bgra.shape[:2]
        ys, xs = np.mgrid[0:height, 0:width]
        board = np.dstack([np.where(((ys // 12) + (xs // 12)) % 2 == 0, 120, 70).astype(np.uint8)] * 3)
        alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
        cell = (bgra[:, :, :3] * alpha + board * (1 - alpha)).astype(np.uint8)
        cv2.drawMarker(cell, (int(entry["grasp_x"] * scale), int(entry["grasp_y"] * scale)),
                       (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cells.append(cell)
    if not cells:
        return None
    tallest = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, tallest - c.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                value=(25, 25, 25)) for c in cells]
    blank = np.full_like(cells[0], 25)
    cells += [blank] * (-len(cells) % columns)
    sheet = np.vstack([np.hstack(cells[r:r + columns]) for r in range(0, len(cells), columns)])
    path = output_dir / "_objects.png"
    cv2.imwrite(str(path), sheet)
    logging.info(f"wrote {path}")
    return path


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="plates", help="Directory of capture runs")
    parser.add_argument("--run_id", default=None,
                        help="Which objectplates run; defaults to every one in the directory")
    parser.add_argument("--output_dir", default=None,
                        help="Where the cutouts go (default <dir>/objects)")
    parser.add_argument("--label", default=None, help="Override the object name")
    parser.add_argument("--green_low", type=float, default=GREEN_LOW)
    parser.add_argument("--green_high", type=float, default=GREEN_HIGH)
    parser.add_argument("--vignette_m", type=float, default=VIGNETTE_DIAMETER_M,
                        help="Diameter on the floor, in metres, outside which nothing is object")
    parser.add_argument("--no_vignette", action="store_true",
                        help="Keep everything the chroma key kept, however far out it is")
    parser.add_argument("--fps", type=int, default=VIDEO_FPS,
                        help="Playback rate of the mp4 written beside the cutouts")
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    runs = [r for r in read_manifest(args.dir) if r["kind"] == "objectplates"]
    if args.run_id:
        runs = [r for r in runs if r["run_id"] == args.run_id]
    if not runs:
        parser.error(f"no objectplates runs in {args.dir}")
    output_dir = Path(args.output_dir or Path(args.dir) / "objects")

    for run in runs:
        extract_run(args.dir, run["run_id"], output_dir, label=args.label,
                    diameter_m=args.vignette_m, vignette=not args.no_vignette,
                    green_low=args.green_low, green_high=args.green_high)
    if not args.no_preview:
        objects = read_objects(output_dir)
        write_preview(output_dir, objects)
        write_video_preview(output_dir, objects, args.fps)


if __name__ == "__main__":
    main()
