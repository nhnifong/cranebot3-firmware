#!/usr/bin/env python

"""Render an episode's gripper video with the mined grasp point drawn on every frame.

The still previews mine_teleop writes show one frame each, which says whether a label
landed on the object but not why it missed. A miss has a shape over time: a mark that
tracks the object and then slides off is a pose estimate drifting, one that sits at a
fixed offset all the way down is a mount constant, and one that jumps frame to frame is
noise in whatever the projection is being fed. Those look identical in stills.

Everything here comes from mine_teleop, deliberately: the same grasp detection, the same
grasp_point_room, the same projection and the same calibration. If this video is wrong
about where the label goes then the labels are wrong in the same way, which is the only
property that makes it worth watching.

Two things it draws that the stills do not:

  - The whole episode, not the mined window, so the frames either side of what training
    sees are visible too. The mined window is marked in the corner.
  - The mark's own history, as a fading trail. A steady drift is much easier to see as a
    curve across the frame than as a dot that moved.

    python -m nf_robot.ml.visual_servoing.label_video \\
        --root datasets/chuck-aug28 \\
        --output_dir datasets/labelling_test/label_video

One mp4 per episode, named for it. Episodes with no detectable grasp are skipped, the
same ones the miner skips.
"""

import argparse
import json
import logging
import math
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.lerobot_trim_to_grasp import (
    MIN_GRASP_SECONDS, PRESSURE_THRESHOLD, RISE_M, find_grasp,
)
from nf_robot.ml.visual_servoing.mine_teleop import (
    gripper_camera_calibration, grasp_point_room, project, read_columns,
)

# The trail is this many frames long. Two seconds at 30fps: enough to show the direction
# a drift is heading without the curve wrapping over itself.
TRAIL_FRAMES = 60
GREEN = (80, 230, 120)
AMBER = (60, 170, 235)
GREY = (150, 150, 150)


def draw_mark(bgr, u, v, colour, radius=9):
    """A crosshair at normalized (u, v), clipped to the frame but drawn even when outside.

    Off-frame labels are drawn on the border rather than dropped: the mined window keeps
    frames whose target is out of shot, and where it went off the edge is the useful part.
    """
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
    """Numbers on a solid banner rather than over the frame.

    These are read while the video plays, over whatever the camera happens to be looking
    at - and a pale outline on a pale carpet is not readable at 30fps.
    """
    step, pad = 17, 6
    banner = pad * 2 + step * len(lines)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], banner), (24, 24, 24), -1)
    for i, text in enumerate(lines):
        cv2.putText(bgr, text, (8, pad + step * i + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    colour, 1, cv2.LINE_AA)


def episode_video(dataset, rows, episode, start, grasp, calibration, fps, path,
                  approach_seconds, carry_seconds, vcodec, crf):
    """One episode's gripper frames with the grasp point drawn on each of them."""
    import av

    image_key = "observation.images.gripper_camera"
    target_room = grasp_point_room(rows[grasp])
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

        projected = project(target_room, row["gripper_pos"], row["spin"], calibration)
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
            f"ep{episode:04d} f{i:05d}  t{seconds:+.2f}s"
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
           approach_seconds=5.0, carry_seconds=1.0, vcodec="libx264", crf=23):
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
        # The same two tests mine_episode applies, so this renders the episodes that are
        # actually mined and nothing else.
        pressure = np.array([r["pressure"] for r in rows], dtype=np.float64)
        grasp = find_grasp(pressure, fps, PRESSURE_THRESHOLD, MIN_GRASP_SECONDS)
        if grasp is None:
            logging.info(f"ep{episode:04d}: no grasp, skipped (the miner skips it too)")
            continue
        heights = np.array([r["gripper_pos"][2] for r in rows])
        if not np.any(heights[grasp:] >= heights[grasp] + RISE_M):
            logging.info(f"ep{episode:04d}: no rise after the grasp, skipped (so does the miner)")
            continue
        path = output_dir / f"ep{episode:04d}.mp4"
        total, labelled, inside = episode_video(
            dataset, rows, episode, starts[episode], grasp, calibration, fps, path,
            approach_seconds, carry_seconds, vcodec, crf)
        logging.info(f"ep{episode:04d}: {total} frames, {labelled} projected, "
                     f"{inside} with the mark inside the frame -> {path.name}")
        written += 1
    logging.info(f"{written} episode video(s) in {output_dir}")
    return written


def main():
    # force=True: importing lerobot installs its own root handler, which makes a plain
    # basicConfig a no-op and silently drops every progress and skip line this prints.
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
    parser.add_argument("--vcodec", default="libx264",
                        help="h264 by default, which every browser and player will scrub")
    parser.add_argument("--crf", type=int, default=23)
    args = parser.parse_args()

    render(Path(args.root), Path(args.output_dir), args.repo_id, args.limit,
           set(args.episodes) if args.episodes else None,
           args.approach_seconds, args.carry_seconds, args.vcodec, args.crf)


if __name__ == "__main__":
    main()
