#!/usr/bin/env python

"""Audit recorded datasets for frozen camera feeds.

A stalled camera keeps delivering frames at full rate, but every frame is the
same image, so the episodes recorded during the stall carry no visual
information at all. This finds them after the fact and prints a paste-ready
`exclude_episodes:` block for a lerobot_build_dataset.py recipe.

The signal is runs of pixel-identical consecutive frames. Live video is never
identical frame to frame - sensor noise moves pixels even when camera and scene
are motionless. The catch is that these cameras run below the dataset's fps
(e.g. a 7.5 fps overhead feed in a 30 fps dataset repeats every image 4 times),
so identical frames are normal and their *fraction* is meaningless: what
separates a stall is the length of the longest unbroken run. Healthy episodes
top out around 1.8 s; real stalls run for whole episodes.

Calibrated against the episodes previously excluded by hand: on naavox/move_clutter
a 3 s threshold reproduces exactly episodes 289-325 and 448-481 (71 of 71, no
false positives), where the frozen gripper camera sat unchanged for a median of
17 s per episode against a healthy maximum of 1.8 s.

Videos are read from the dataset's local snapshot, so each dataset is downloaded
in full on first use (tens of GB for a large one).

Usage:
    python src/nf_robot/ml/lerobot_find_frozen_video.py \
        --repo_id justink04/laundry-in-hamper2-8-1-26 justink04/trash-in-trashcan-7-23-26 \
        [--min_frozen_seconds 3.0] \
        [--json_out frozen_report.json]
"""

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nf_robot.ml.frozen_camera_monitor import FROZEN_SECONDS, fingerprint

# Runs shorter than this are the normal sub-fps duplication, not a stall. Kept
# low so short runs still land in the report for context; --min_frozen_seconds
# decides what actually gets flagged.
MIN_RUN_FRAMES = 4


def scan_video(path: str):
    """Find runs of identical frames in one video file.

    Returns (path, [(start_seconds, end_seconds, n_frames), ...], total_frames).
    """
    runs = []
    prev = None
    start_t = last_t = None
    run_len = 1
    n_frames = 0

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base)
        for frame in container.decode(stream):
            n_frames += 1
            t = (frame.pts * time_base) if frame.pts is not None else float("nan")
            fp = fingerprint(frame.to_ndarray(format="rgb24"))
            if prev is not None and np.array_equal(fp, prev):
                run_len += 1
            else:
                if run_len >= MIN_RUN_FRAMES:
                    runs.append((start_t, last_t, run_len))
                run_len = 1
                start_t = t
            last_t = t
            prev = fp

    if run_len >= MIN_RUN_FRAMES:
        runs.append((start_t, last_t, run_len))
    return path, runs, n_frames


def dataset_root(repo_id: str) -> Path:
    """Local snapshot of a dataset, downloading it (videos included) if needed."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return Path(LeRobotDataset(repo_id=repo_id, root=None).root)


def episode_metadata(root: Path) -> tuple[dict, list[str], pd.DataFrame]:
    info = json.loads((root / "meta" / "info.json").read_text())
    cameras = [k for k in info["features"] if k.startswith("observation.images")]
    episodes = pd.concat([pq.read_table(p).to_pandas()
                          for p in sorted((root / "meta" / "episodes").glob("**/*.parquet"))])
    return info, cameras, episodes


def video_path(root: Path, camera: str, row) -> Path:
    chunk = int(row[f"videos/{camera}/chunk_index"])
    file_index = int(row[f"videos/{camera}/file_index"])
    return root / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"


def attribute_to_episodes(root: Path, runs_by_path: dict) -> dict:
    """Turn per-file frozen runs into per-episode, per-camera frozen time.

    One video file spans many episodes, so each run is intersected with each
    episode's [from_timestamp, to_timestamp) window for that camera.
    """
    info, cameras, episodes = episode_metadata(root)
    per_episode = defaultdict(dict)
    lengths = {}

    for _, row in episodes.iterrows():
        episode = int(row["episode_index"])
        lengths[episode] = int(row["length"])
        for camera in cameras:
            runs = runs_by_path.get(str(video_path(root, camera, row)), [])
            t0 = float(row[f"videos/{camera}/from_timestamp"])
            t1 = float(row[f"videos/{camera}/to_timestamp"])
            frozen = longest = 0.0
            for run_start, run_end, _n in runs:
                overlap = min(run_end, t1) - max(run_start, t0)
                if overlap > 0:
                    frozen += overlap
                    longest = max(longest, overlap)
            if frozen > 0:
                per_episode[episode][camera.split(".")[-1]] = {
                    "frozen_s": round(frozen, 2),
                    "longest_s": round(longest, 2),
                    "episode_s": round(t1 - t0, 2),
                    "frac": round(frozen / max(t1 - t0, 1e-9), 3),
                }

    return {
        "root": str(root),
        "total_episodes": info["total_episodes"],
        "episode_lengths": lengths,
        "frozen": {str(k): v for k, v in sorted(per_episode.items())},
    }


def scan_datasets(repo_ids: list[str], workers: int = 8) -> dict:
    roots = {repo_id: dataset_root(repo_id) for repo_id in repo_ids}

    jobs = []
    for root in roots.values():
        jobs.extend(str(p) for p in sorted((root / "videos").glob("**/*.mp4")))
    logging.info(f"Scanning {len(jobs)} video files from {len(roots)} dataset(s)")

    runs_by_path = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, runs, n_frames in pool.map(scan_video, jobs, chunksize=1):
            runs_by_path[path] = runs
            logging.info(f"  {Path(path).parts[-3]}/{Path(path).name}: "
                         f"{n_frames} frames, {len(runs)} repeated-frame run(s)")

    return {repo_id: attribute_to_episodes(root, runs_by_path) for repo_id, root in roots.items()}


def flagged_episodes(report: dict, min_frozen_seconds: float) -> list[tuple[int, dict]]:
    """Episodes with a single unbroken stall of at least min_frozen_seconds."""
    flagged = []
    for episode, cameras in report["frozen"].items():
        bad = {c: v for c, v in cameras.items() if v["longest_s"] >= min_frozen_seconds}
        if bad:
            flagged.append((int(episode), bad))
    return sorted(flagged)


def as_ranges(indices: list[int]) -> list:
    """Collapse [1,2,3,7] into ["1-3", 7] for a recipe's exclude_episodes."""
    if not indices:
        return []
    ranges = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    ranges.append((start, prev))
    return [f"{a}-{b}" if a != b else a for a, b in ranges]


def print_report(report: dict, min_frozen_seconds: float) -> None:
    for repo_id, result in report.items():
        flagged = flagged_episodes(result, min_frozen_seconds)
        print(f"\n=== {repo_id}: {len(flagged)}/{result['total_episodes']} episodes with "
              f"video frozen for >= {min_frozen_seconds}s")
        for episode, cameras in flagged:
            for camera, stats in sorted(cameras.items()):
                print(f"  ep {episode:4d}  {camera:16s} longest={stats['longest_s']:6.2f}s  "
                      f"total={stats['frozen_s']:6.2f}s of {stats['episode_s']:6.2f}s  "
                      f"({int(stats['frac'] * 100)}% of episode)")
        if flagged:
            print(f"\n  - repo_id: {repo_id}")
            print(f"    exclude_episodes: {as_ranges([e for e, _ in flagged])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", nargs="+", required=True,
                        help="one or more dataset repo ids to scan")
    parser.add_argument("--min_frozen_seconds", type=float, default=FROZEN_SECONDS,
                        help="length of an unbroken frozen run that flags an episode")
    parser.add_argument("--workers", type=int, default=8, help="parallel video decoders")
    parser.add_argument("--json_out", type=Path, default=None,
                        help="write the full per-episode, per-camera report here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = scan_datasets(args.repo_id, workers=args.workers)

    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        logging.info(f"Wrote {args.json_out}")

    print_report(report, args.min_frozen_seconds)


if __name__ == "__main__":
    main()
