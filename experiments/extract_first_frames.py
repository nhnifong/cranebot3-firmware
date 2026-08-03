#!/usr/bin/env python

"""Write the first frame of every episode of one camera to a folder of images.

For eyeballing a dataset episode by episode - which robot recorded it, where it was
recorded, whether the scene changed partway through a long recording session.

Only the requested camera's videos and the metadata are downloaded, not the whole
dataset, which for a four-camera recording is a fraction of the total.

Usage:
    python experiments/extract_first_frames.py \
        --repo_id naavox/move_clutter \
        --camera observation.images.overhead_camera \
        --output_dir move_clutter_first_frames
"""

import argparse
import collections
import os

import av
import cv2
import numpy as np
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--camera", default="observation.images.overhead_camera")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--root", default=None, help="local copy of the dataset (skips the download)")
    parser.add_argument("--ext", default="jpg", choices=["jpg", "png"])
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    root = args.root
    if root is None:
        print(f"Downloading meta and {args.camera} videos of {args.repo_id}...")
        root = snapshot_download(
            repo_id=args.repo_id, repo_type="dataset",
            allow_patterns=["meta/**", f"videos/{args.camera}/**"],
        )

    meta = LeRobotDatasetMetadata(repo_id=args.repo_id, root=root)
    if args.camera not in meta.video_keys:
        raise SystemExit(f"'{args.camera}' is not a video feature. Available: {meta.video_keys}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Episodes are packed several to a video file, so group by file and walk each one
    # in timestamp order rather than reopening it per episode.
    by_file = collections.defaultdict(list)
    for ep_idx in range(meta.total_episodes):
        episode = meta.episodes[ep_idx]
        rel = meta.get_video_file_path(ep_idx, args.camera)
        start = float(episode[f"videos/{args.camera}/from_timestamp"])
        by_file[str(rel)].append((start, ep_idx))

    written = 0
    for rel, wanted in sorted(by_file.items()):
        path = os.path.join(root, rel)
        wanted.sort()
        with av.open(path) as container:
            stream = container.streams.video[0]
            for start, ep_idx in wanted:
                # seek lands on the keyframe at or before the episode's first frame,
                # then decode forward to the first frame that belongs to the episode
                container.seek(int(start / stream.time_base), stream=stream, backward=True)
                frame = next(
                    (f for f in container.decode(stream) if f.time is not None and f.time >= start - 1e-3),
                    None,
                )
                if frame is None:
                    print(f"  episode {ep_idx}: no frame at t={start:.2f}s in {rel}")
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                out = os.path.join(args.output_dir, f"episode_{ep_idx:04d}.{args.ext}")
                cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                written += 1
        print(f"  {rel}: {len(wanted)} episode(s)")

    print(f"\nWrote {written}/{meta.total_episodes} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
