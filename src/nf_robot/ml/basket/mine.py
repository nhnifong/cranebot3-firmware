#!/usr/bin/env python

"""Mine basket centering labels from recordings made by the basketdata maneuver.

Every episode starts with the gripper centred over a basket at the height it should drop
from, and then flies away from it. The point under the jaws in that first frame is
followed forward by optical flow until it is lost, and every frame it was followed through
gets labelled with where that point is (u, v, distance along the ray) and the gripper move,
in the body frame, that would put it back under the jaws at the starting height.

The sideways part of the move comes from the flow, scaled by the point's depth: the first
frame's rangefinder reading plus the gantry's climb since. The vertical part is the gantry's
height change alone. The rangefinder over a basket of clothes reads into gaps and onto piles
rather than the level the clothes move with in the image, so nothing vertical is taken from
it, and after the first frame it reads the floor beside the basket anyway.

Writes the pool and then deals it into train and eval a whole episode at a time:

    python -m nf_robot.ml.basket.mine \\
        --repo_id naavox/basketdata-2 \\
        --output_root datasets/basket_centering \\
        --preview_dir datasets/basket_centering/preview
"""

import argparse
import logging
import random
from pathlib import Path

import cv2
import numpy as np

from nf_robot.ml.basket.geometry import return_body
from nf_robot.ml.visual_servoing.geometry import rotate_about_vertical
from nf_robot.ml.visual_servoing.mine_teleop import (
    CANVAS_SCALE, IMAGE_KEY, IMAGE_SIZE, POOL_SPLIT, ReservoirSampler, ShardWriter,
    encode_frame, frame_bgr, hub_root, read_columns, write_dataset_card,
)
from nf_robot.ml.visual_servoing.split_pool import split_pool
from nf_robot.ml.visual_servoing.uv_methods import (
    flow_track, gripper_camera_calibration, jaw_uv, range_to_depth, unproject,
)

SHARD_PREFIX = "basket"
SPLIT_SOURCE = "basket"
# Keep one frame in this many of what was tracked. Neighbouring frames at 30fps are near
# duplicates, but the flow has to step through every one of them regardless.
DEFAULT_STRIDE = 2
# An episode whose track is lost sooner than this is not kept: the flight barely began.
MIN_TRACKED_FRAMES = 10
# (seconds) how many first frames to look at for the steadiest rangefinder reading, since
# the centred frame is what every label in the episode hangs off.
CENTER_WINDOW_S = 0.3
DEFAULT_EVAL_FRACTION = 0.15
# Frames kept aside for the preview, sampled across every episode.
PREVIEW_POOL = 400


def row_schema():
    """Parquet schema for one labelled basket frame."""
    import pyarrow as pa

    return pa.schema([
        ("image", pa.binary()),
        ("split_source", pa.string()),
        ("source_repo_id", pa.string()),
        ("episode_index", pa.int32()),
        ("frame_index", pa.int32()),
        ("seconds_from_center", pa.float32()),
        # where the basket's drop point is in the frame, and how far along that ray
        ("target_uv", pa.list_(pa.float32())),
        ("target_range_m", pa.float32()),
        # the rangefinder reading when centred: how high over the basket to drop from
        ("drop_range_m", pa.float32()),
        # the gripper move back to centred, metres in the body frame: x, y from the flow,
        # z from the gantry's height change
        ("return_body", pa.list_(pa.float32())),
        # the same move from the recorded gantry positions, kept to check the flow against
        ("return_body_telemetry", pa.list_(pa.float32())),
        ("state", pa.struct([
            ("laser_rangefinder", pa.float32()),
            ("finger_angle", pa.float32()),
            ("target_force", pa.float32()),
        ])),
    ])


def center_range(rows, fps, window_s=CENTER_WINDOW_S):
    """The rangefinder reading in the centred frames, the median over the first window_s, or
    None if it read nothing."""
    first = rows[:max(1, int(round(window_s * fps)))]
    readings = [float(r["laser_rangefinder"]) for r in first if r["laser_rangefinder"] > 0]
    return float(np.median(readings)) if readings else None


def label_frame(uv, row, center, drop_range, calibration):
    """(target_uv, target_range_m, return_body) for one tracked frame, or None if the point
    has left the canvas or the ray cannot reach it."""
    half = (CANVAS_SCALE - 1.0) / 2.0
    if not (-half <= uv[0] <= 1 + half and -half <= uv[1] <= 1 + half):
        return None
    climbed = float(row["gantry_pos"][2] - center["gantry_pos"][2])
    distance = range_to_depth(uv[0], uv[1], drop_range + climbed, calibration)
    if distance is None:
        return None
    point_cam = unproject(uv[0], uv[1], distance, calibration)
    move = return_body(point_cam, drop_range, spin=row["spin"], centered_spin=center["spin"])
    # Already -climbed by construction, since the point's depth was set from it; said
    # outright so the vertical label plainly owes nothing to the rangefinder.
    move[2] = -climbed
    return [float(uv[0]), float(uv[1])], float(distance), move


def mine_episode(rows, fps, calibration, frames, stride=DEFAULT_STRIDE, image_size=IMAGE_SIZE):
    """Labelled rows for one episode as (rows, reason), with rows None and a reason when the
    episode is unusable. frames(i) is row i's BGR frame, and each kept one is stored as a
    JPEG at image_size."""
    center = rows[0]
    if center.get("gantry_pos") is None:
        raise ValueError("this recording has no gantry_position_x/y/z in observation.state, "
                         "which the vertical labels are taken from")
    drop_range = center_range(rows, fps)
    if drop_range is None:
        return None, "no_range"
    anchor = jaw_uv(drop_range, calibration)
    if anchor is None:
        return None, "no_range"

    # The flow decodes every frame anyway, so the kept ones are encoded on the way past
    # rather than decoded a second time.
    gray, encoded = {}, {}

    def at(i):
        if i not in gray:
            image = frames(i)
            gray[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if i % stride == 0:
                encoded[i] = encode_frame(image, image_size)
            for stale in [k for k in gray if abs(k - i) > 2]:
                del gray[stale]
        return gray[i]

    at(0)
    track = {0: anchor, **flow_track(at, 0, anchor, len(rows))}
    if len(track) < MIN_TRACKED_FRAMES:
        return None, "lost_early"

    out = []
    for i in sorted(track):
        if i % stride:
            continue
        r = rows[i]
        labelled = label_frame(track[i], r, center, drop_range, calibration)
        if labelled is None:
            break
        uv, distance, move = labelled
        # The move the gantry positions say, in the same body frame, for comparison.
        telemetry = rotate_about_vertical(
            np.asarray(center["gantry_pos"]) - np.asarray(r["gantry_pos"]), float(r["spin"]))
        out.append({
            "image": encoded[i],
            "split_source": SPLIT_SOURCE,
            "frame_index": r["frame_index"],
            "seconds_from_center": round(float(r["timestamp"] - center["timestamp"]), 3),
            "target_uv": [round(u, 5) for u in uv],
            "target_range_m": round(distance, 4),
            "drop_range_m": round(drop_range, 4),
            "return_body": [round(float(m), 4) for m in move],
            "return_body_telemetry": [round(float(m), 4) for m in telemetry],
            "state": {
                "laser_rangefinder": round(float(r["laser_rangefinder"]), 4),
                "finger_angle": round(float(r["finger_angle"]), 3),
                "target_force": round(float(r["target_force"]), 4),
            },
        })
    return out, None


def mine(sources, output_root: Path, stride=DEFAULT_STRIDE, limit=None,
         image_size=IMAGE_SIZE):
    """Replace the basket shards in the pool with rows mined from (repo_id, root) sources.
    Returns the rows written, and how far the flow disagreed with the gripper positions."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    split_dir = output_root / POOL_SPLIT
    split_dir.mkdir(parents=True, exist_ok=True)
    for stale in split_dir.glob(f"{SHARD_PREFIX}-*.parquet"):
        stale.unlink()
    writer = ShardWriter(split_dir, prefix=SHARD_PREFIX, schema=row_schema())
    calibration = gripper_camera_calibration()
    disagreement = []
    preview = ReservoirSampler(PREVIEW_POOL, seed=0)

    for repo_id, root in sources:
        episodes, fps = read_columns(root, need_finger_speed=False)
        dataset = LeRobotDataset(repo_id, root=root)
        if IMAGE_KEY not in dataset.meta.video_keys:
            raise ValueError(f"{repo_id} has no {IMAGE_KEY}; present: {dataset.meta.video_keys}")
        starts = {
            int(r["episode_index"]): int(r["dataset_from_index"])
            for r in dataset.meta.episodes.select_columns(
                ["episode_index", "dataset_from_index"]).to_list()
        }
        skipped = {}
        mined = 0
        chosen = sorted(episodes)[:limit] if limit else sorted(episodes)
        for ep in tqdm(chosen, desc=repo_id.split("/")[-1], unit="ep", dynamic_ncols=True):
            rows = episodes[ep]
            base = starts[ep]
            samples, reason = mine_episode(
                rows, fps, calibration,
                frames=lambda i, base=base, rows=rows: frame_bgr(dataset, base + rows[i]["frame_index"]),
                stride=stride, image_size=image_size)
            if samples is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            for sample in samples:
                sample["episode_index"] = ep
                sample["source_repo_id"] = repo_id
                flow = np.asarray(sample["return_body"])
                telemetry = np.asarray(sample["return_body_telemetry"])
                disagreement.append(float(np.linalg.norm((flow - telemetry)[:2])))
                preview.add(sample)
                writer.add(sample)
                mined += 1
        logging.info(f"{repo_id}: {mined} frames from {len(chosen) - sum(skipped.values())}"
                     f"/{len(chosen)} episodes, skipped {skipped or 'none'}")
    writer.flush()
    write_dataset_card(output_root)

    if disagreement:
        d = np.array(disagreement) * 100
        logging.info(f"lateral flow vs gantry position: median {np.median(d):.1f}cm, "
                     f"90th percentile {np.percentile(d, 90):.1f}cm. Large values mean the flow "
                     f"slid off the basket or the position estimate drifted; look at the "
                     f"preview before training on it.")
    return writer.total, preview.rows


def render_preview(samples, preview_dir: Path, count=60, seed=0, columns=4, group=20):
    """Annotated frames: the tracked point, and the move back to centre as the flow says and
    as the gantry positions say, which should agree to a few centimetres sideways."""
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old in list(preview_dir.glob("*.jpg")) + list(preview_dir.glob("*.png")):
        old.unlink()
    chosen = random.Random(seed).sample(samples, min(count, len(samples)))
    chosen.sort(key=lambda s: (s["episode_index"], s["frame_index"]))

    annotated = []
    for s in chosen:
        img = cv2.imdecode(np.frombuffer(s["image"], np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        h, w = img.shape[:2]
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        canvas = cv2.copyMakeBorder(img, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT,
                                    value=(40, 40, 40))
        u, v = s["target_uv"]
        point = (int(u * w + pad_x), int(v * h + pad_y))
        cv2.drawMarker(canvas, point, (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.circle(canvas, point, 16, (0, 255, 0), 2)
        lines = [
            f"ep{s['episode_index']} f{s['frame_index']}  t+{s['seconds_from_center']:.2f}s",
            "flow  " + "  ".join(f"{c * 100:+.0f}" for c in s["return_body"]) + " cm",
            "pos   " + "  ".join(f"{c * 100:+.0f}" for c in s["return_body_telemetry"]) + " cm",
            f"drop range {s['drop_range_m']:.2f}m  ray {s['target_range_m']:.2f}m",
        ]
        for i, line in enumerate(lines):
            y = 26 + i * 26
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        name = f"ep{s['episode_index']:04d}_f{s['frame_index']:05d}.jpg"
        cv2.imwrite(str(preview_dir / name), canvas)
        annotated.append(canvas)

    for start in range(0, len(annotated), group):
        cells = annotated[start:start + group]
        ch, cw = cells[0].shape[:2]
        cells = cells + [np.full((ch, cw, 3), 25, np.uint8)] * (-len(cells) % columns)
        sheet = np.vstack([np.hstack(cells[r:r + columns]) for r in range(0, len(cells), columns)])
        cv2.imwrite(str(preview_dir / f"_sheet_{start // group + 1:02d}.png"), sheet)
    logging.info(f"wrote {len(chosen)} preview frames to {preview_dir}")


def main():
    # force=True because importing lerobot/transformers installs a root handler.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, nargs="+",
                        help="Recordings made with the basketdata maneuver")
    parser.add_argument("--root", default=None, nargs="+",
                        help="Their roots on disk, in the same order (defaults to the HF cache)")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--preview_dir", default=None, help="Write annotated sample frames here")
    parser.add_argument("--preview_count", type=int, default=60)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="Keep one tracked frame in this many")
    parser.add_argument("--limit", type=int, default=None, help="Only mine this many episodes")
    parser.add_argument("--eval_fraction", type=float, default=DEFAULT_EVAL_FRACTION,
                        help="Share of episodes dealt to eval once the pool is written")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, nargs=2, default=list(IMAGE_SIZE),
                        metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()

    roots = args.root or []
    if roots and len(roots) != len(args.repo_id):
        parser.error(f"got {len(args.repo_id)} --repo_id but {len(roots)} --root")
    sources = [(repo_id, Path(roots[i]) if roots else Path(hub_root(repo_id)))
               for i, repo_id in enumerate(args.repo_id)]

    output_root = Path(args.output_root)
    total, kept = mine(sources, output_root, stride=args.stride, limit=args.limit,
                       image_size=tuple(args.image_size))
    if not total:
        raise SystemExit("nothing was mined")
    split_pool(output_root, args.eval_fraction, args.seed, schema=row_schema(),
               by_episode=True)
    if args.preview_dir:
        render_preview(kept, Path(args.preview_dir), args.preview_count, args.seed)


if __name__ == "__main__":
    main()
