#!/usr/bin/env python

"""Render an episode's gripper video with the mined grasp point and its trail drawn on every frame.

    python -m nf_robot.ml.visual_servoing.label_video \\
        --root datasets/chuck-aug28 \\
        --output_dir datasets/labelling_test/label_video
"""

import argparse
import json
import logging
import math
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.lerobot.trim_to_grasp import (
    MIN_GRASP_SECONDS, PRESSURE_THRESHOLD, RISE_M, find_grasp,
)
from nf_robot.ml.visual_servoing.mine_teleop import frame_bgr, read_columns
from nf_robot.ml.visual_servoing.uv_methods import (
    DEFAULT_UV_METHOD, PIXEL_METHODS, add_uv_arguments, anchor_is_usable,
    gripper_camera_calibration, target_track,
)

# Trail length in frames (two seconds at 30fps).
TRAIL_FRAMES = 60
GREEN = (80, 230, 120)
AMBER = (60, 170, 235)
GREY = (150, 150, 150)


def draw_mark(bgr, u, v, colour, radius=9):
    """A crosshair at normalized (u, v), drawn on the border when off-frame."""
    h, w = bgr.shape[:2]
    x, y = u * w, v * h
    inside = 0 <= x < w and 0 <= y < h
    cx = int(np.clip(x, 2, w - 3))
    cy = int(np.clip(y, 2, h - 3))
    if inside:
        cv2.circle(bgr, (cx, cy), radius, colour, 2, cv2.LINE_AA)
        cv2.line(bgr, (cx - radius - 6, cy), (cx + radius + 6, cy), colour, 1, cv2.LINE_AA)
        cv2.line(bgr, (cx, cy - radius - 6), (cx, cy + radius + 6), colour, 1, cv2.LINE_AA)
    else:
        # A hollow triangle on the edge, pointing the way the target lies.
        cv2.drawMarker(bgr, (cx, cy), colour, cv2.MARKER_TRIANGLE_UP, 18, 2, cv2.LINE_AA)
    return inside


def draw_trail(bgr, trail):
    """The mark's recent track, oldest faintest, so a drift reads as a curve."""
    h, w = bgr.shape[:2]
    for age, (u, v) in enumerate(trail):
        weight = (age + 1) / len(trail)
        x, y = int(np.clip(u * w, 0, w - 1)), int(np.clip(v * h, 0, h - 1))
        shade = tuple(int(c * weight) for c in GREEN)
        cv2.circle(bgr, (x, y), 2, shade, -1, cv2.LINE_AA)


def caption(bgr, lines, colour=(235, 235, 235)):
    """Text on a solid banner so it stays readable over the video."""
    step, pad = 17, 6
    banner = pad * 2 + step * len(lines)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], banner), (24, 24, 24), -1)
    for i, text in enumerate(lines):
        cv2.putText(bgr, text, (8, pad + step * i + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    colour, 1, cv2.LINE_AA)


def episode_video(dataset, rows, episode, start, grasp, calibration, fps, path,
                  approach_seconds, carry_seconds, vcodec, crf,
                  uv_method=DEFAULT_UV_METHOD, jaw_uv=None):
    """One episode's gripper frames with the grasp point drawn on each of them."""
    import av

    image_key = "observation.images.gripper_camera"
    track = target_track(
        rows, grasp, calibration, uv_method, jaw_uv,
        frames=(lambda i: frame_bgr(dataset, start + i, image_key))
        if uv_method in PIXEL_METHODS else None)
    window = (max(0, grasp - int(round(approach_seconds * fps))),
              min(len(rows) - 1, grasp + int(round(carry_seconds * fps))))

    first = dataset[start][image_key]
    frame0 = (first.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    height, width = frame0.shape[:2]

    container = av.open(str(path), mode="w")
    stream = container.add_stream(vcodec, rate=Fraction(int(round(fps)), 1))
    stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "8"} if vcodec == "libsvtav1" else {"crf": str(crf)}

    trail, on_object = [], 0
    labelled = 0
    for i, row in enumerate(rows):
        rgb = dataset[start + i][image_key].numpy().transpose(1, 2, 0)
        bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        projected = track[i]
        seconds = rows[grasp]["timestamp"] - row["timestamp"]
        in_window = window[0] <= i <= window[1]

        if projected is not None:
            u, v, distance = projected
            draw_trail(bgr, trail[-TRAIL_FRAMES:])
            inside = draw_mark(bgr, u, v, GREEN if in_window else GREY)
            trail.append((u, v))
            labelled += 1
            on_object += inside
            detail = f"uv {u:+.3f},{v:+.3f}  range {distance:.3f}m"
        else:
            detail = "behind the lens"

        pos = row["gripper_pos"]
        caption(bgr, [
            f"ep{episode:04d} f{i:05d}  t{seconds:+.2f}s  {uv_method}"
            + ("  [mined]" if in_window else "")
            + ("  <- GRASP" if i == grasp else ""),
            detail,
            f"gripper {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}  spin {math.degrees(row['spin']):+.1f}deg",
            f"laser {row['laser_rangefinder']:.3f}m  pressure {row['pressure']:.3f}",
        ], GREEN if in_window else GREY)

        packet = av.VideoFrame.from_ndarray(bgr, format="bgr24").reformat(format="yuv420p")
        for out in stream.encode(packet):
            container.mux(out)
    for out in stream.encode():
        container.mux(out)
    container.close()
    return len(rows), labelled, on_object


def render(root: Path, output_dir: Path, repo_id=None, limit=None, episodes_wanted=None,
           approach_seconds=5.0, carry_seconds=1.0, vcodec="libx264", crf=23,
           uv_method=DEFAULT_UV_METHOD, jaw_uv=None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = gripper_camera_calibration()
    episodes, fps = read_columns(root)

    repo_id = repo_id or json.loads((root / "meta" / "info.json").read_text()).get(
        "repo_id") or root.name
    dataset = LeRobotDataset(repo_id, root=root)
    starts = {
        int(r["episode_index"]): int(r["dataset_from_index"])
        for r in dataset.meta.episodes.select_columns(
            ["episode_index", "dataset_from_index"]).to_list()
    }

    written = 0
    for episode in sorted(episodes):
        if episodes_wanted is not None and episode not in episodes_wanted:
            continue
        if limit and written >= limit:
            break
        rows = episodes[episode]
        # The same three tests mine_episode applies.
        pressure = np.array([r["pressure"] for r in rows], dtype=np.float64)
        grasp = find_grasp(pressure, fps, PRESSURE_THRESHOLD, MIN_GRASP_SECONDS)
        if grasp is None:
            logging.info(f"ep{episode:04d}: no grasp, skipped (the miner skips it too)")
            continue
        heights = np.array([r["gripper_pos"][2] for r in rows])
        if not np.any(heights[grasp:] >= heights[grasp] + RISE_M):
            logging.info(f"ep{episode:04d}: no rise after the grasp, skipped (so does the miner)")
            continue
        if not anchor_is_usable(rows[grasp], calibration):
            logging.info(f"ep{episode:04d}: rangefinder read "
                         f"{rows[grasp]['laser_rangefinder']:.3f}m at the grasp, so there is "
                         f"no target to draw, skipped (so does the miner)")
            continue
        # Named for the method too, so renders under different methods don't overwrite each
        # other.
        suffix = "" if uv_method == DEFAULT_UV_METHOD else f"_{uv_method}"
        path = output_dir / f"ep{episode:04d}{suffix}.mp4"
        total, labelled, inside = episode_video(
            dataset, rows, episode, starts[episode], grasp, calibration, fps, path,
            approach_seconds, carry_seconds, vcodec, crf, uv_method, jaw_uv)
        logging.info(f"ep{episode:04d}: {total} frames, {labelled} projected, "
                     f"{inside} with the mark inside the frame -> {path.name}")
        written += 1
    logging.info(f"{written} episode video(s) in {output_dir}")
    return written


def main():
    # force=True because importing lerobot installs its own root handler.
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="A recorded LeRobot dataset directory")
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Render at most this many episodes")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Render only these episode indices")
    parser.add_argument("--approach_seconds", type=float, default=5.0,
                        help="Only marks the window; the whole episode is rendered either way")
    parser.add_argument("--carry_seconds", type=float, default=1.0)
    add_uv_arguments(parser)
    parser.add_argument("--vcodec", default="libx264",
                        help="h264 by default, which every browser and player will scrub")
    parser.add_argument("--crf", type=int, default=23)
    args = parser.parse_args()

    render(Path(args.root), Path(args.output_dir), args.repo_id, args.limit,
           set(args.episodes) if args.episodes else None,
           args.approach_seconds, args.carry_seconds, args.vcodec, args.crf,
           args.uv_method, tuple(args.jaw_uv) if args.jaw_uv else None)


if __name__ == "__main__":
    main()
