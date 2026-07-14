#!/usr/bin/env python

"""Derive a new LeRobot dataset from an existing one by changing camera_mode.

Drops video features that aren't part of the target camera_mode and/or
shrinks the resolution of the ones that are kept. The target camera_mode's
camera set must be a subset of the source dataset's, and each kept camera's
target resolution must be the same size or smaller than the source.

Camera modes are defined in nf_robot.ml.stringman_lerobot._CAMERA_MODES.

Usage example:

Derive a gripper_224 dataset from one recorded with camera_mode="all":
    python src/nf_robot/ml/lerobot_derive_dataset.py \
        --repo_id naavox/simple_grasp \
        --new_repo_id naavox/simple_grasp_224 \
        --new_root datasets/simple_grasp_224 \
        --camera_mode gripper_224

Derive a gripper_anchors_384 dataset (gripper + both overhead/anchor cams at
384 square, no floor cam) from one recorded with camera_mode="all":
    python src/nf_robot/ml/lerobot_derive_dataset.py \
        --repo_id naavox/simple_grasp \
        --new_repo_id naavox/simple_grasp_ga384 \
        --new_root datasets/simple_grasp_ga384 \
        --camera_mode gripper_anchors_384
"""

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import get_video_info
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

from nf_robot.ml.lerobot_resize_video_feature import resize_video
from nf_robot.ml.stringman_lerobot import _CAMERA_MODES, _FEED_NAMES


def _write_info_json(info: dict, root: Path) -> None:
    """Write meta/info.json directly from a plain dict.

    Bypasses lerobot's write_info, whose signature changed across versions
    (0.5.1 took a dict; >=0.6 takes a DatasetInfo and calls .to_dict()).
    """
    (Path(root) / "meta" / "info.json").write_text(json.dumps(info, indent=4))


def derive_dataset(
    dataset: LeRobotDataset,
    camera_mode: str,
    output_dir: Path,
    repo_id: str,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    crf: int = 30,
    g: int = 2,
    headroom: int = 0,
    center_crop: bool = False,
    pad_clamp: bool = False,
) -> LeRobotDataset:
    if camera_mode not in _CAMERA_MODES:
        raise ValueError(f"Unknown camera_mode '{camera_mode}'. Valid: {list(_CAMERA_MODES)}")

    target_specs = _CAMERA_MODES[camera_mode]
    target_keys = {
        f"observation.images.{_FEED_NAMES[feed]}": (width, height)
        for feed, (width, height) in target_specs.items()
    }

    missing = [key for key in target_keys if key not in dataset.meta.video_keys]
    if missing:
        raise ValueError(
            f"Source dataset is missing camera feature(s) required for camera_mode "
            f"'{camera_mode}': {missing}. Available: {dataset.meta.video_keys}"
        )

    for key, (target_w, target_h) in target_keys.items():
        src_h, src_w = dataset.meta.features[key]["shape"][:2]
        if not pad_clamp and (target_w > src_w or target_h > src_h):
            raise ValueError(
                f"Cannot derive camera_mode '{camera_mode}': target resolution "
                f"{target_w}x{target_h} for '{key}' exceeds source resolution {src_w}x{src_h}. "
                f"Pass pad_clamp=True to center and pad the smaller axis instead of failing."
            )

    features_to_remove = [key for key in dataset.meta.video_keys if key not in target_keys]

    if features_to_remove:
        logging.info(f"Removing features: {features_to_remove}")
        dataset = modify_features(
            dataset,
            remove_features=features_to_remove,
            output_dir=output_dir,
            repo_id=repo_id,
        )
    else:
        logging.info("Target camera set matches source; copying dataset as-is")
        shutil.copytree(dataset.root, output_dir, dirs_exist_ok=True)
        info_path = output_dir / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["repo_id"] = repo_id
        _write_info_json(info, output_dir)
        dataset = LeRobotDataset(repo_id=repo_id, root=output_dir)

    # lerobot >=0.6 wraps meta.info in a DatasetInfo object (use .to_dict());
    # 0.5.1 exposes a plain dict. Support both so this script survives version switches.
    meta_info = dataset.meta.info
    info = meta_info.to_dict() if hasattr(meta_info, "to_dict") else dict(meta_info)
    fps = dataset.meta.fps
    # All-file parallelism: one single-threaded encode per worker, one worker per
    # available core. For many independent files this beats fewer multi-threaded
    # encodes, because encoder-internal threading (esp. SVT-AV1) scales poorly
    # while independent files scale nearly linearly. `headroom` cores are left free.
    workers = max(1, (os.cpu_count() or 1) - headroom)
    threads_per_worker = 1

    for key, (target_w, target_h) in target_keys.items():
        src_h, src_w = dataset.meta.features[key]["shape"][:2]
        if (target_w, target_h) == (src_w, src_h):
            continue

        video_files: set[Path] = set()
        for ep_idx in range(dataset.meta.total_episodes):
            try:
                video_files.add(dataset.meta.get_video_file_path(ep_idx, key))
            except KeyError:
                continue
        sorted_files = sorted(video_files)

        logging.info(
            f"Resizing '{key}' from {src_w}x{src_h} to {target_w}x{target_h} "
            f"({len(sorted_files)} file(s)) with {workers} workers x "
            f"{threads_per_worker} encoder thread(s)"
        )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for rel_path in sorted_files:
                path = dataset.root / rel_path
                tmp_path = path.with_suffix(".tmp.mp4")
                futures[
                    pool.submit(
                        resize_video, path, tmp_path, target_w, target_h, fps, vcodec, pix_fmt, crf, g, center_crop,
                        threads_per_worker, pad_clamp,
                    )
                ] = (path, tmp_path)
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Resizing {key}"):
                future.result()  # re-raises any encoding exception
            for path, tmp_path in futures.values():
                tmp_path.replace(path)

        new_video_info = get_video_info(dataset.root / sorted_files[0])
        channels = info["features"][key]["shape"][2]
        info["features"][key]["info"] = new_video_info
        info["features"][key]["shape"] = [target_h, target_w, channels]

    _write_info_json(info, dataset.root)
    logging.info(f"Done. Derived '{camera_mode}' dataset written to {dataset.root}")
    return LeRobotDataset(repo_id=repo_id, root=dataset.root)


def main() -> None:
    init_logging()

    parser = argparse.ArgumentParser(description="Derive a new LeRobot dataset with a different camera_mode.")
    parser.add_argument("--repo_id", required=True, help="Source dataset repo id")
    parser.add_argument("--root", default=None, help="Source dataset root path")
    parser.add_argument("--new_repo_id", default=None, help="Output dataset repo id (defaults to repo_id)")
    parser.add_argument("--new_root", required=True, help="Output dataset root path")
    parser.add_argument(
        "--camera_mode", required=True, choices=list(_CAMERA_MODES.keys()), help="Target camera_mode"
    )
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec (default: libsvtav1)")
    parser.add_argument("--pix_fmt", default="yuv420p", help="Pixel format (default: yuv420p)")
    parser.add_argument("--crf", type=int, default=30, help="Constant rate factor (default: 30)")
    parser.add_argument("--g", type=int, default=2, help="GOP size (default: 2)")
    parser.add_argument(
        "--headroom", type=int, default=0,
        help="CPU cores to leave free; the rest run one single-threaded encode each (default: 0)",
    )
    parser.add_argument(
        "--center_crop", action="store_true",
        help="Center-crop to the target aspect ratio before resizing instead of stretching (default: stretch)",
    )
    parser.add_argument(
        "--pad_clamp", action="store_true",
        help="Allow target resolutions larger than the source: center the frame on the "
             "target canvas with no scaling and pad the smaller axis by replicating edge "
             "pixels, instead of raising (mutually exclusive with --center_crop)",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Upload the derived dataset to the Hugging Face Hub"
    )
    args = parser.parse_args()

    input_root = Path(args.root) if args.root else HF_LEROBOT_HOME / args.repo_id
    output_repo_id = args.new_repo_id or args.repo_id
    output_root = Path(args.new_root)

    if output_root == input_root:
        raise ValueError("--new_root must differ from the source dataset root")

    dataset = LeRobotDataset(repo_id=args.repo_id, root=input_root)

    logging.info(f"Deriving camera_mode '{args.camera_mode}' from '{args.repo_id}'")

    derived = derive_dataset(
        dataset=dataset,
        camera_mode=args.camera_mode,
        output_dir=output_root,
        repo_id=output_repo_id,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        crf=args.crf,
        g=args.g,
        headroom=args.headroom,
        center_crop=args.center_crop,
        pad_clamp=args.pad_clamp,
    )

    if args.push_to_hub:
        logging.info(f"Pushing '{output_repo_id}' to the Hugging Face Hub")
        derived.push_to_hub()


if __name__ == "__main__":
    main()
