#!/usr/bin/env python

"""Trim every episode down to the segment where the robot actually grasps something.

A recorded episode typically contains three phases: traversing the room toward the
object, the grasp itself, and carrying the object to its destination. For policies
that only need to learn the grasp, the traverse and carry phases are mostly wasted
frames. This cuts each episode down to just the grasp.

The segment is defined as:

  start  The moment the robot arrived near where the grasp ends up happening, i.e.
         the beginning of the unbroken run of frames leading into the grasp during
         which the gripper stayed within `near_radius_m` (horizontally) of the grasp
         position. An episode that never traverses - the gripper was already in the
         neighbourhood the whole time - simply starts at frame 0, so "if there is any
         traversal" needs no special case.

  end    The first frame after the grasp at which the gripper has risen
         `rise_m` above the height it grasped at. That is the point the object is
         unambiguously picked up and off whatever it was resting on.

The grasp itself is the first frame where finger_pressure holds at or above
`pressure_threshold` for `min_grasp_seconds` - the hold requirement rejects the
brief pressure spikes that come from bumping the object before actually closing on it.

Episodes with no detectable grasp, no subsequent rise, or a segment shorter than
`min_segment_seconds` are dropped entirely, on the grounds that a dataset of grasps
should not contain non-grasps. The counts are logged so a recipe that silently
discards most of its input is obvious.

Detection reads only the parquet state columns, so it is fast and can be previewed
with --dry_run before paying for the rewrite. The rewrite itself goes through
LeRobotDataset's public writer API, which means every video is decoded and
re-encoded - the slow step, and the reason this is best run once on a merged dataset
rather than per source.

Requires observation.state to carry gripper_pos_x/y/z and finger_pressure. Run this
before any keep_state_features trim that would drop them.

Usage:
    python src/nf_robot/ml/lerobot_trim_to_grasp.py \
        --repo_id naavox/some_dataset \
        --root /path/to/dataset \
        --output_dir /path/to/trimmed \
        [--dry_run]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

# (metres) how close to the eventual grasp point the gripper must be, horizontally,
# to count as "arrived". Chosen from the sensitivity of real move_clutter episodes:
# below ~0.2 m the start creeps forward into the grasp itself, and by ~1.0 m most
# episodes start at frame 0 because the whole traversal falls inside the radius.
NEAR_RADIUS_M = 0.3
# (metres) how far the gripper must climb above the grasp height to call it lifted.
RISE_M = 0.10
# (normalized 0-1) finger_pressure at or above this counts as holding something.
PRESSURE_THRESHOLD = 0.1
# (seconds) how long pressure must hold before it counts as a grasp rather than a bump.
MIN_GRASP_SECONDS = 0.3
# (seconds) segments shorter than this are degenerate - typically a pressure blip
# while already at the object - and are dropped rather than kept as near-empty episodes.
MIN_SEGMENT_SECONDS = 1.0

REQUIRED_STATE = ("gripper_pos_x", "gripper_pos_y", "gripper_pos_z", "finger_pressure")


def grasp_window(
    pressure: np.ndarray,
    pos: np.ndarray,
    fps: float,
    pressure_threshold: float = PRESSURE_THRESHOLD,
    min_grasp_seconds: float = MIN_GRASP_SECONDS,
    rise_m: float = RISE_M,
    near_radius_m: float = NEAR_RADIUS_M,
    min_segment_seconds: float = MIN_SEGMENT_SECONDS,
) -> tuple[tuple[int, int] | None, str]:
    """Find one episode's grasp segment.

    pressure is (N,) finger_pressure, pos is (N, 3) gripper position in metres.
    Returns ((start, end) inclusive, "ok") or (None, reason).
    """
    n = len(pressure)
    hold = max(1, int(round(min_grasp_seconds * fps)))
    if n < hold:
        return None, "too_short"

    # A grasp is `hold` consecutive frames at or above the threshold; the convolution
    # counts how many of each window's frames qualify, so == hold means all of them.
    over = (pressure >= pressure_threshold).astype(np.int32)
    held = np.convolve(over, np.ones(hold, dtype=np.int32), mode="valid") == hold
    grasps = np.flatnonzero(held)
    if grasps.size == 0:
        return None, "no_grasp"
    grasp = int(grasps[0])

    grasp_pos = pos[grasp]
    risen = np.flatnonzero(pos[grasp:, 2] >= grasp_pos[2] + rise_m)
    if risen.size == 0:
        return None, "no_rise"
    end = grasp + int(risen[0])

    # Walk back from the grasp for as long as the gripper stayed in the neighbourhood.
    # Hitting frame 0 means the episode had no traversal to trim.
    near = np.linalg.norm(pos[:, :2] - grasp_pos[:2], axis=1) <= near_radius_m
    start = grasp
    while start > 0 and near[start - 1]:
        start -= 1

    if (end - start + 1) < max(1, int(round(min_segment_seconds * fps))):
        return None, "too_short"
    return (start, end), "ok"


def _read_state(root: Path) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """Read observation.state and episode_index straight from the parquets.

    Avoids constructing a LeRobotDataset, which would decode video we don't need.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    names = list(info["features"]["observation.state"]["names"])
    missing = [n for n in REQUIRED_STATE if n not in names]
    if missing:
        raise ValueError(
            f"observation.state is missing {missing}, which grasp detection needs. "
            f"Trim to the grasp before dropping state components, not after. Available: {names}"
        )

    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise ValueError(f"No data parquets under {root}")
    table = pa.concat_tables([pq.read_table(f, columns=["observation.state", "episode_index"]) for f in files])
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    episodes = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    index = {n: i for i, n in enumerate(names)}
    return state, episodes, index, float(info["fps"])


def compute_windows(root: Path, **kwargs) -> tuple[dict[int, tuple[int, int]], dict[str, int]]:
    """Grasp segment per episode, as episode_index -> (start, end) frame offsets.

    Offsets are relative to the start of the episode. Episodes with no usable
    segment are absent from the result. Also returns a count of rejection reasons.
    """
    state, episodes, index, fps = _read_state(root)
    pressure_all = state[:, index["finger_pressure"]]
    pos_all = state[:, [index["gripper_pos_x"], index["gripper_pos_y"], index["gripper_pos_z"]]]

    windows: dict[int, tuple[int, int]] = {}
    reasons: dict[str, int] = {}
    for ep in sorted(set(episodes.tolist())):
        mask = episodes == ep
        window, reason = grasp_window(pressure_all[mask], pos_all[mask], fps, **kwargs)
        reasons[reason] = reasons.get(reason, 0) + 1
        if window is not None:
            windows[int(ep)] = window
    return windows, reasons


def log_window_summary(root: Path, windows: dict[int, tuple[int, int]], reasons: dict[str, int]) -> None:
    _, episodes, _, fps = _read_state(root)
    total = len(set(episodes.tolist()))
    kept_frames = sum(e - s + 1 for s, e in windows.values())
    source_frames = sum(int((episodes == ep).sum()) for ep in windows)
    logging.info(
        f"Grasp segments found in {len(windows)}/{total} episodes "
        f"(rejected: { {k: v for k, v in reasons.items() if k != 'ok'} })"
    )
    if windows:
        lengths = np.array([e - s + 1 for s, e in windows.values()])
        logging.info(
            f"Segment length: median {np.median(lengths) / fps:.1f}s, "
            f"range {lengths.min() / fps:.1f}-{lengths.max() / fps:.1f}s; "
            f"keeping {kept_frames} of {source_frames} frames "
            f"({kept_frames / max(1, source_frames) * 100:.0f}%) in those episodes"
        )


def _to_frame_value(value):
    """Convert one __getitem__ value into what add_frame/validate_frame expect.

    LeRobotDataset hands back video frames as float CHW in [0, 1]; the writer wants
    the uint8 HWC that the feature's (H, W, C) shape describes.
    """
    import torch

    if isinstance(value, torch.Tensor):
        value = value.numpy()
    return value


def _to_image(value):
    array = _to_frame_value(value)
    if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[0] < array.shape[-1]:
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.dtype != np.uint8:
        array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return array


def trim_dataset(dataset, windows: dict[int, tuple[int, int]], output_dir: Path, repo_id: str):
    """Write a new dataset containing only each episode's grasp segment.

    Episodes absent from `windows` are dropped. Uses the public writer API, so the
    output is built by lerobot itself rather than by editing parquets and video in
    place - slower, but it cannot leave the metadata and the media disagreeing.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not windows:
        raise ValueError("No episode had a detectable grasp segment; refusing to write an empty dataset")

    feature_keys = [k for k in dataset.meta.features if k not in
                    ("timestamp", "frame_index", "episode_index", "index", "task_index")]
    video_keys = set(dataset.meta.video_keys)

    new_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        root=output_dir,
        robot_type=dataset.meta.robot_type,
        use_videos=len(video_keys) > 0,
    )

    episode_starts = {
        int(row["episode_index"]): int(row["dataset_from_index"])
        for row in dataset.meta.episodes.select_columns(["episode_index", "dataset_from_index"]).to_list()
    }

    for ep in sorted(windows):
        start, end = windows[ep]
        base = episode_starts[ep]
        for i in range(base + start, base + end + 1):
            item = dataset[i]
            frame = {}
            for key in feature_keys:
                frame[key] = _to_image(item[key]) if key in video_keys else _to_frame_value(item[key])
            frame["task"] = item["task"]
            new_dataset.add_frame(frame)
        new_dataset.save_episode()
        logging.info(f"episode {ep}: kept frames {start}-{end} ({end - start + 1})")

    # Without this the parquet footers are never written and the result is unreadable.
    new_dataset.finalize()
    return new_dataset


def trim_dataset_to_grasp(dataset, output_dir: Path, repo_id: str, dry_run: bool = False, **kwargs):
    """Detect grasp segments in `dataset` and write the trimmed copy to output_dir."""
    windows, reasons = compute_windows(Path(dataset.root), **kwargs)
    log_window_summary(Path(dataset.root), windows, reasons)
    if dry_run:
        return None
    return trim_dataset(dataset, windows, Path(output_dir), repo_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="Source dataset repo id")
    parser.add_argument("--root", default=None, help="Source dataset root (defaults to the HF cache)")
    parser.add_argument("--new_repo_id", default=None, help="Output repo id (defaults to <repo_id>_grasp)")
    parser.add_argument("--output_dir", default=None, help="Where to write the trimmed dataset")
    parser.add_argument("--dry_run", action="store_true", help="Report the segments without writing anything")
    parser.add_argument("--pressure_threshold", type=float, default=PRESSURE_THRESHOLD)
    parser.add_argument("--min_grasp_seconds", type=float, default=MIN_GRASP_SECONDS)
    parser.add_argument("--rise_m", type=float, default=RISE_M)
    parser.add_argument("--near_radius_m", type=float, default=NEAR_RADIUS_M)
    parser.add_argument("--min_segment_seconds", type=float, default=MIN_SEGMENT_SECONDS)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    new_repo_id = args.new_repo_id or f"{args.repo_id}_grasp"
    output_dir = Path(args.output_dir) if args.output_dir else Path(dataset.root).parent / new_repo_id.split("/")[-1]

    trim_dataset_to_grasp(
        dataset, output_dir, new_repo_id, dry_run=args.dry_run,
        pressure_threshold=args.pressure_threshold,
        min_grasp_seconds=args.min_grasp_seconds,
        rise_m=args.rise_m,
        near_radius_m=args.near_radius_m,
        min_segment_seconds=args.min_segment_seconds,
    )


if __name__ == "__main__":
    main()
