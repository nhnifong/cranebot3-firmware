#!/usr/bin/env python

"""Tile many episodes of a LeRobot dataset into one mosaic video.

Built for media/PR shots of a dataset: a grid of episodes all playing at once
reads as "look how much data this is" in a way a single clip never does.

Each grid cell gets its own episode, scaled to the tile size. Episodes shorter
than the requested duration loop; longer ones are cut off. Episodes are sampled
longest-first by default, so the cells that do loop loop as few times as possible.

Decoding is one linear pass per underlying video file rather than a seek per
episode: a v3 dataset packs many episodes into a handful of mp4s, so reading each
mp4 once and slicing out the episode segments beats thousands of random seeks.

The whole mosaic is held in memory as uint8 tiles, which is
    rows * cols * duration * fps * tile * tile * 3 bytes
- about 2.6 GB at the 16x9x14s default. Drop --tile if that is too much.

Usage:
    python experiments/dataset_mosaic.py \
        --repo_id naavox/grasp_only_224 \
        --camera gripper_camera \
        --duration 14 \
        --output mosaic.mp4
"""

import argparse
import json
import logging
import random
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def resolve_camera_key(video_keys: list[str], camera: str) -> str:
    """Accept either a full feature key or the bare camera name."""
    if camera in video_keys:
        return camera
    matches = [k for k in video_keys if k.split(".")[-1] == camera]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No camera '{camera}' in this dataset. Available: {video_keys}")
    raise ValueError(f"Camera '{camera}' is ambiguous: {matches}")


def episode_segments(root: Path, camera_key: str) -> list[dict]:
    """Every episode's video file and time range for one camera, longest first."""
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for path in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        wanted = [
            "episode_index",
            f"videos/{camera_key}/chunk_index",
            f"videos/{camera_key}/file_index",
            f"videos/{camera_key}/from_timestamp",
            f"videos/{camera_key}/to_timestamp",
        ]
        missing = [c for c in wanted if c not in table.schema.names]
        if missing:
            raise ValueError(f"episode metadata has no {missing}; is '{camera_key}' really a video feature?")
        for row in table.select(wanted).to_pylist():
            rows.append(
                {
                    "episode": row["episode_index"],
                    "chunk": row[f"videos/{camera_key}/chunk_index"],
                    "file": row[f"videos/{camera_key}/file_index"],
                    "start": row[f"videos/{camera_key}/from_timestamp"],
                    "end": row[f"videos/{camera_key}/to_timestamp"],
                }
            )
    rows.sort(key=lambda r: r["end"] - r["start"], reverse=True)
    return rows


def video_path(root: Path, camera_key: str, chunk: int, file_index: int) -> Path:
    return root / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def _fit(frame: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to square then nearest-neighbour scale to size x size.

    Nearest-neighbour keeps this dependency-light; at mosaic tile sizes the
    difference from a smooth resize is invisible.
    """
    h, w = frame.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    square = frame[top:top + side, left:left + side]
    idx = (np.arange(size) * (side / size)).astype(np.int32)
    return square[idx][:, idx]


def load_tiles(root: Path, camera_key: str, segments: list[dict], tile: int, want_frames: int) -> list[np.ndarray]:
    """Decode each episode segment into a (T, tile, tile, 3) uint8 clip.

    One pass per mp4: segments are grouped by file and read in timestamp order,
    so the decoder only ever moves forward.
    """
    by_file: dict[tuple[int, int], list[dict]] = {}
    for seg in segments:
        by_file.setdefault((seg["chunk"], seg["file"]), []).append(seg)

    clips: dict[int, np.ndarray] = {}
    for (chunk, file_index), segs in sorted(by_file.items()):
        path = video_path(root, camera_key, chunk, file_index)
        segs = sorted(segs, key=lambda s: s["start"])
        logging.info(f"decoding {len(segs)} segment(s) from {path.name}")

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            pending = list(segs)
            buffers: dict[int, list[np.ndarray]] = {s["episode"]: [] for s in segs}
            for frame in container.decode(stream):
                if not pending:
                    break
                t = float(frame.pts * stream.time_base)
                # segments are disjoint and in order, so drop any we have passed
                while pending and t >= pending[0]["end"]:
                    pending.pop(0)
                if not pending or t < pending[0]["start"]:
                    continue
                seg = pending[0]
                if len(buffers[seg["episode"]]) < want_frames:
                    buffers[seg["episode"]].append(_fit(frame.to_ndarray(format="rgb24"), tile))

        for seg in segs:
            frames = buffers[seg["episode"]]
            if frames:
                clips[seg["episode"]] = np.stack(frames)

    ordered = [clips[s["episode"]] for s in segments if s["episode"] in clips]
    dropped = len(segments) - len(ordered)
    if dropped:
        logging.warning(f"{dropped} episode(s) decoded to nothing and were skipped")
    return ordered


def build_mosaic(clips: list[np.ndarray], rows: int, cols: int, tile: int, total_frames: int):
    """Yield each composed mosaic frame, looping clips shorter than the video."""
    canvas = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for frame_index in range(total_frames):
        for cell, clip in enumerate(clips):
            r, c = divmod(cell, cols)
            canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = clip[frame_index % len(clip)]
        yield canvas


def write_video(frames, path: Path, fps: float, crf: int, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "slow"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="Dataset to tile")
    parser.add_argument("--camera", required=True, help="Video feature key, or the bare camera name")
    parser.add_argument("--duration", type=float, default=14.0, help="Length of the mosaic, seconds")
    parser.add_argument("--root", default=None, help="Dataset root (defaults to the HF cache, downloading if needed)")
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--tile", type=int, default=120, help="Tile edge in pixels (16x9x120 -> 1920x1080)")
    parser.add_argument("--fps", type=float, default=None, help="Output fps (defaults to the dataset's)")
    parser.add_argument("--crf", type=int, default=18, help="x264 quality, lower is better")
    parser.add_argument("--shuffle", action="store_true",
                        help="Sample episodes at random instead of taking the longest")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --shuffle")
    parser.add_argument("--output", default="mosaic.mp4")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    root = Path(dataset.root)
    camera_key = resolve_camera_key(list(dataset.meta.video_keys), args.camera)
    fps = args.fps or dataset.meta.fps
    cells = args.rows * args.cols
    total_frames = max(1, int(round(args.duration * fps)))

    segments = episode_segments(root, camera_key)
    if len(segments) < cells:
        raise ValueError(
            f"{args.rows}x{args.cols} needs {cells} episodes but the dataset has {len(segments)}. "
            f"Use a smaller grid."
        )
    if args.shuffle:
        random.Random(args.seed).shuffle(segments)
    segments = segments[:cells]

    shortest = min(s["end"] - s["start"] for s in segments)
    logging.info(
        f"{cells} episodes of '{camera_key}', shortest {shortest:.1f}s vs {args.duration:.0f}s of output "
        f"-> up to {int(np.ceil(args.duration / max(shortest, 1e-6)))} loops in the worst cell"
    )
    est_gb = cells * total_frames * args.tile * args.tile * 3 / 1e9
    logging.info(f"holding tiles in memory: about {est_gb:.1f} GB")

    clips = load_tiles(root, camera_key, segments, args.tile, total_frames)
    if len(clips) < cells:
        raise ValueError(f"only {len(clips)} of {cells} episodes decoded; cannot fill the grid")

    width, height = args.cols * args.tile, args.rows * args.tile
    logging.info(f"writing {width}x{height} at {fps:g} fps, {total_frames} frames -> {args.output}")
    write_video(
        build_mosaic(clips, args.rows, args.cols, args.tile, total_frames),
        Path(args.output), fps, args.crf, width, height,
    )
    logging.info(f"done: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
