#!/usr/bin/env python

"""Resize a video feature in a LeRobot dataset.

Re-encodes all video files for a given feature key at a new resolution and
writes a new dataset (or modifies in-place with a backup) following the same
conventions as lerobot-edit-dataset.

Usage examples:

Resize observation.images.top to 320x240 in a new dataset:
    python experiments/lerobot_resize_video_feature.py \
        --repo_id my/dataset \
        --root /path/to/dataset \
        --new_root /path/to/dataset_resized \
        --feature_key observation.images.top \
        --width 320 \
        --height 240

Resize in-place (creates backup at dataset_old/):
    python experiments/lerobot_resize_video_feature.py \
        --repo_id my/dataset \
        --root /path/to/dataset \
        --feature_key observation.images.top \
        --width 320 \
        --height 240
"""

import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import av
from tqdm import tqdm

from lerobot.datasets.io_utils import write_info
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_video_info
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def resize_video(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: float,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    crf: int = 30,
    g: int = 2,
) -> None:
    """Re-encode a video file at a new resolution using PyAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_container = av.open(str(input_path))
    if not in_container.streams.video:
        raise ValueError(f"No video stream found in {input_path}")

    v_in = in_container.streams.video[0]
    fps_fraction = Fraction(fps).limit_denominator(1000)

    out = av.open(str(output_path), mode="w")
    v_out = out.add_stream(vcodec, rate=fps_fraction)
    v_out.width = width
    v_out.height = height
    v_out.pix_fmt = pix_fmt
    v_out.time_base = Fraction(1, int(fps))

    codec_options = {"crf": str(crf), "g": str(g)}
    if vcodec == "libsvtav1":
        codec_options = {"crf": str(crf), "preset": "8"}
    v_out.options = codec_options

    out.start_encoding()

    frame_count = 0
    for packet in in_container.demux(v_in):
        for frame in packet.decode():
            if frame is None:
                continue
            resized = frame.reformat(width=width, height=height, format=pix_fmt)
            resized.pts = frame_count
            resized.time_base = Fraction(1, int(fps))
            for pkt in v_out.encode(resized):
                out.mux(pkt)
            frame_count += 1

    for pkt in v_out.encode():
        out.mux(pkt)

    out.close()
    in_container.close()


def resize_video_feature(
    dataset: LeRobotDataset,
    feature_key: str,
    width: int,
    height: int,
    output_dir: Path,
    repo_id: str,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    crf: int = 30,
    g: int = 2,
    num_workers: int | None = None,
) -> LeRobotDataset:
    if feature_key not in dataset.meta.video_keys:
        raise ValueError(
            f"Feature '{feature_key}' is not a video feature. "
            f"Available video keys: {dataset.meta.video_keys}"
        )

    # Collect unique video files for this feature key
    video_files: set[Path] = set()
    for ep_idx in range(dataset.meta.total_episodes):
        try:
            rel_path = dataset.meta.get_video_file_path(ep_idx, feature_key)
            video_files.add(rel_path)
        except KeyError:
            continue

    fps = dataset.meta.fps
    workers = num_workers if num_workers is not None else os.cpu_count()
    sorted_files = sorted(video_files)

    # Re-encode each video file at the new resolution, one job per file
    logging.info(f"Resizing {len(sorted_files)} video file(s) with {workers} workers")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                resize_video,
                dataset.root / rel_path,
                output_dir / rel_path,
                width, height, fps, vcodec, pix_fmt, crf, g,
            ): rel_path
            for rel_path in sorted_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Resizing {feature_key}"):
            future.result()  # re-raises any encoding exception

    # Copy everything else (data, meta, other video keys)
    logging.info("Copying non-video data and metadata...")
    for item in dataset.root.iterdir():
        if item.name == "videos":
            continue
        dst_item = output_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)

    # Copy videos for other feature keys unchanged
    for other_key in dataset.meta.video_keys:
        if other_key == feature_key:
            continue
        for ep_idx in range(dataset.meta.total_episodes):
            try:
                rel_path = dataset.meta.get_video_file_path(ep_idx, other_key)
            except KeyError:
                continue
            src = dataset.root / rel_path
            dst = output_dir / rel_path
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # Update info.json with new video dimensions
    info = dict(dataset.meta.info)
    if feature_key in info.get("features", {}):
        # Refresh from the newly encoded file
        sample_video = output_dir / sorted(video_files)[0]
        new_video_info = get_video_info(sample_video)
        info["features"][feature_key]["info"] = new_video_info
        # Update shape if present
        feature_entry = info["features"][feature_key]
        if "shape" in feature_entry:
            channels = new_video_info.get("video.channels", feature_entry["shape"][0])
            feature_entry["shape"] = [channels, height, width]

    info["repo_id"] = repo_id
    write_info(info, output_dir)

    logging.info(f"Done. Resized dataset written to {output_dir}")
    return LeRobotDataset(repo_id=repo_id, root=output_dir)


def main() -> None:
    init_logging()

    parser = argparse.ArgumentParser(description="Resize a video feature in a LeRobot dataset.")
    parser.add_argument("--repo_id", required=True, help="Input dataset repo id")
    parser.add_argument("--root", default=None, help="Input dataset root path")
    parser.add_argument("--new_repo_id", default=None, help="Output dataset repo id (defaults to repo_id)")
    parser.add_argument("--new_root", default=None, help="Output dataset root path")
    parser.add_argument("--feature_key", required=True, help="Video feature key to resize")
    parser.add_argument("--width", type=int, required=True, help="Target width in pixels")
    parser.add_argument("--height", type=int, required=True, help="Target height in pixels")
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec (default: libsvtav1)")
    parser.add_argument("--pix_fmt", default="yuv420p", help="Pixel format (default: yuv420p)")
    parser.add_argument("--crf", type=int, default=30, help="Constant rate factor (default: 30)")
    parser.add_argument("--g", type=int, default=2, help="GOP size (default: 2)")
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Parallel encoding workers (default: all CPU cores)",
    )
    args = parser.parse_args()

    input_root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    output_repo_id = args.new_repo_id or args.repo_id
    output_root = Path(args.new_root) if args.new_root else HF_LEROBOT_HOME / output_repo_id

    # Handle in-place modification: move original to backup
    if output_root == input_root:
        backup = input_root.with_name(input_root.name + "_old")
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(input_root), str(backup))
        input_root = backup
        logging.info(f"In-place mode: original dataset backed up to {backup}")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=input_root)

    logging.info(
        f"Resizing '{args.feature_key}' in '{args.repo_id}' to {args.width}x{args.height}"
    )

    resize_video_feature(
        dataset=dataset,
        feature_key=args.feature_key,
        width=args.width,
        height=args.height,
        output_dir=output_root,
        repo_id=output_repo_id,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        crf=args.crf,
        g=args.g,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
