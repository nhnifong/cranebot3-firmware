#!/usr/bin/env python

"""Dump ortho (overhead_camera) frames from a LeRobot v3 dataset for diffusion fine-tuning.

Takes --per_episode evenly spaced frames from every episode by seeking, rather than decoding the
whole dataset; grip_o is 832k frames. Captions carry the episode's task, which hints at what's
in the room (hamper, toybox, ball pit...).

Writes <output_dir>/ep<episode>_<k>.png and metadata.jsonl in the layout train_lora_sdxl.py reads.

Usage:
    python experiments/anchor_diffusion/extract_ortho.py \
        --root /media/nhn/nfdrive/huggingface/hub/datasets--naavox--grip_o/snapshots/<rev> \
        --output_dir experiments/anchor_diffusion/data/grip_o_ortho
"""

import argparse
import glob
import json
import os

import av
import numpy as np
import pyarrow.parquet as pq

KEY = "observation.images.overhead_camera"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="local LeRobot v3 dataset directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--per_episode", type=int, default=4)
    parser.add_argument("--token", default="strngmn")
    args = parser.parse_args()

    episodes = pq.read_table(glob.glob(os.path.join(args.root, "meta", "episodes", "chunk-*", "*.parquet"))).to_pandas()
    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    with open(os.path.join(args.output_dir, "metadata.jsonl"), "w") as meta:
        # group by video file so each mp4 is opened once
        for (chunk, file), group in episodes.groupby([f"videos/{KEY}/chunk_index", f"videos/{KEY}/file_index"]):
            path = os.path.join(args.root, "videos", KEY, f"chunk-{chunk:03d}", f"file-{file:03d}.mp4")
            with av.open(path) as container:
                stream = container.streams.video[0]
                for ep in group.to_dict("records"):
                    start, end = ep[f"videos/{KEY}/from_timestamp"], ep[f"videos/{KEY}/to_timestamp"]
                    task = str(np.atleast_1d(ep["tasks"])[0]).strip().lower().replace("hamp[er", "hamper")
                    caption = f"{args.token} ortho top-down view of a room, {task}"
                    # avoid the very start and end of each episode
                    for k, ts in enumerate(np.linspace(start, end, args.per_episode + 2)[1:-1]):
                        container.seek(int(ts / stream.time_base), stream=stream, backward=True, any_frame=False)
                        frame = None
                        for frame in container.decode(stream):
                            if frame.time >= ts - 1e-3:
                                break
                        if frame is None:
                            continue
                        name = f"ep{ep['episode_index']:05d}_{k}.png"
                        frame.to_image().save(os.path.join(args.output_dir, name))
                        meta.write(json.dumps({"file_name": name, "text": caption, "episode": int(ep["episode_index"])}) + "\n")
                        count += 1
            print(f"{path}: {count} frames so far")
    print(f"wrote {count} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
