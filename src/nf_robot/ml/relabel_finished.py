#!/usr/bin/env python
"""
Batch relabeling tools for LeRobot datasets.

Modes:

  finished  (default)
    Re-label finished=1.0 for the last N frames of each episode.
    For datasets recorded before live finish-labeling was added.

  prompts
    Replace the single "Pick up the item" prompt with three phase-based prompts
    derived from finger_speed and finger_pressure signals:

      phase 0 – "Choose an item …"    (start of episode)
      phase 1 – "grasp the item"      (60 frames before first finger_speed 0→+ transition)
      phase 2 – "lift the item …"     (first frame where finger_pressure > 0.1)

Usage:
    python relabel_finished.py --repo_id myuser/mydataset
    python relabel_finished.py --repo_id myuser/mydataset --root /path/to/dataset
    python relabel_finished.py --repo_id myuser/mydataset --num_frames 20
    python relabel_finished.py --repo_id myuser/mydataset --mode prompts
    python relabel_finished.py --repo_id myuser/mydataset --mode prompts --root /path/to/dataset
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── prompt-mode task texts ────────────────────────────────────────────────────
TASK_CHOOSE = (
    "Choose an item on the floor that could be picked up with one hand, "
    "is lighter than 2 kg, and isn't flat like a book, and center the gripper over that item"
)
TASK_GRASP = "grasp the item"
TASK_LIFT = (
    "lift the item and bring it either to a bin or pile with similar items in it "
    "such as a laundry hamper, trash can, or bucket, and drop it there."
)
PROMPT_TASKS = [TASK_CHOOSE, TASK_GRASP, TASK_LIFT]
TASK_IDX_CHOOSE, TASK_IDX_GRASP, TASK_IDX_LIFT = 0, 1, 2

# ── shared helpers ────────────────────────────────────────────────────────────

def find_finished_index(feature: dict) -> int | None:
    """Return the index of 'finished' in a sequence feature's names list, or None."""
    names = feature.get("names")
    if isinstance(names, list):
        try:
            return names.index("finished")
        except ValueError:
            return None
    if isinstance(names, dict):
        offset = 0
        for group in names.values():
            try:
                return offset + group.index("finished")
            except ValueError:
                offset += len(group)
    return None


def fix_meta_episodes_indices(dataset_root: Path) -> None:
    """Fix stale meta/episodes/chunk_index and meta/episodes/file_index values.

    A prior merge may append episodes from multiple source files into one destination
    file while leaving the self-referential chunk/file index columns pointing at
    non-existent files.  This re-stamps each row with the indices of the file it
    actually lives in.
    """
    meta_episodes_dir = dataset_root / "meta" / "episodes"
    if not meta_episodes_dir.exists():
        return

    fixed_count = 0
    for chunk_dir in sorted(meta_episodes_dir.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        chunk_idx = int(chunk_dir.name.split("-")[1])

        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            if parquet_file.suffix != ".parquet":
                continue
            file_idx = int(parquet_file.stem.split("-")[1])

            df = pd.read_parquet(parquet_file)
            if (
                "meta/episodes/chunk_index" not in df.columns
                or "meta/episodes/file_index" not in df.columns
            ):
                continue

            if (
                (df["meta/episodes/chunk_index"] == chunk_idx).all()
                and (df["meta/episodes/file_index"] == file_idx).all()
            ):
                continue

            shutil.copy2(parquet_file, parquet_file.with_suffix(".parquet.bak"))
            df["meta/episodes/chunk_index"] = chunk_idx
            df["meta/episodes/file_index"] = file_idx
            df.to_parquet(parquet_file, index=False)
            logger.info(
                f"  Fixed meta/episodes/{chunk_dir.name}/{parquet_file.name}: "
                f"reset self-referential indices to chunk={chunk_idx}, file={file_idx}"
            )
            fixed_count += 1

    if fixed_count:
        logger.info(f"Fixed {fixed_count} meta/episodes parquet file(s) with stale indices.")
    else:
        logger.info("meta/episodes indices are consistent, no fix needed.")


# ── finished-mode ─────────────────────────────────────────────────────────────

def patch_list_column(
    table: pa.Table, col_name: str, finished_idx: int, episode_col: str, num_frames: int
) -> tuple[pa.Table, bool]:
    """Set action[finished_idx]=1.0 for the last num_frames of each episode. Returns (table, changed)."""
    field_idx = table.schema.get_field_index(col_name)
    if field_idx < 0:
        return table, False

    col = table.column(col_name)
    rows = np.array(col.to_pylist(), dtype=np.float32)

    ep_col_idx = table.schema.get_field_index(episode_col)
    if ep_col_idx >= 0:
        episode_ids = table.column(episode_col).to_pylist()
        episode_series = pd.Series(episode_ids)
        changed = False
        for _, group in episode_series.groupby(episode_series):
            idx = group.index.tolist()
            start = idx[max(0, len(idx) - num_frames)]
            end = idx[-1] + 1
            if not np.all(rows[start:end, finished_idx] == 1.0):
                rows[start:end, finished_idx] = 1.0
                changed = True
    else:
        # Fallback: label last num_frames of the whole file
        start = max(0, len(rows) - num_frames)
        if np.all(rows[start:, finished_idx] == 1.0):
            return table, False
        rows[start:, finished_idx] = 1.0
        changed = True

    if not changed:
        return table, False
    new_col = pa.array(rows.tolist(), type=col.type)
    return table.set_column(field_idx, table.schema.field(col_name), new_col), True


def relabel_finished(repo_id: str, root: str | None, num_frames: int, do_recompute_stats: bool = False) -> None:
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id
    dataset = LeRobotDataset(repo_id, root=dataset_root)

    features = dataset.meta.features
    action_idx = find_finished_index(features.get("action", {}))
    obs_idx = find_finished_index(features.get("observation.state", {}))

    if action_idx is None and obs_idx is None:
        logger.error("'finished' not found in action or observation.state features. Nothing to do.")
        return

    logger.info(f"action 'finished' index:            {action_idx}")
    logger.info(f"observation.state 'finished' index: {obs_idx}")

    parquet_files = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found under {dataset_root}/data/")
        return

    logger.info(f"Found {len(parquet_files)} parquet files, labeling last {num_frames} frames per episode")
    modified_count = 0

    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path)
        changed = False

        if action_idx is not None:
            table, c = patch_list_column(table, "action", action_idx, "episode_index", num_frames)
            changed = changed or c

        if obs_idx is not None:
            table, c = patch_list_column(table, "observation.state", obs_idx, "episode_index", num_frames)
            changed = changed or c

        if changed:
            shutil.copy2(parquet_path, parquet_path.with_suffix(".parquet.bak"))
            pq.write_table(table, parquet_path)
            logger.info(f"  patched {parquet_path.stem}")
            modified_count += 1

    logger.info(f"Done. Modified {modified_count}/{len(parquet_files)} parquet files.")

    fix_meta_episodes_indices(dataset_root)

    if do_recompute_stats:
        logger.info("Recomputing dataset stats...")
        recompute_stats(dataset)
        logger.info("Stats recomputed.")


# ── prompts-mode ──────────────────────────────────────────────────────────────

def _compute_prompt_boundaries(
    obs: np.ndarray,
    finger_speed_idx: int,
    finger_pressure_idx: int,
    grasp_lookahead: int = 60,
) -> tuple[int, int | None]:
    """Return (grasp_start, lift_start) as local frame offsets within an episode.

    grasp_start: episode-local row index where phase 1 ("grasp") begins.
    lift_start:  episode-local row index where phase 2 ("lift") begins, or None.
    """
    n = len(obs)
    speeds = obs[:, finger_speed_idx]

    # First frame where finger_speed transitions 0 → positive
    grasp_trigger = None
    for i in range(1, n):
        if speeds[i - 1] == 0.0 and speeds[i] > 0.0:
            grasp_trigger = i
            break

    grasp_start = max(0, grasp_trigger - grasp_lookahead) if grasp_trigger is not None else n

    # First frame where finger_pressure exceeds 0.1
    pressures = obs[:, finger_pressure_idx]
    above = np.where(pressures > 0.1)[0]
    lift_start = int(above[0]) if len(above) > 0 else None

    if lift_start is not None and lift_start < grasp_start:
        logger.warning(
            f"  lift_start ({lift_start}) precedes grasp_start ({grasp_start}) — check signals"
        )

    return grasp_start, lift_start


def _assign_task_indices(n_frames: int, grasp_start: int, lift_start: int | None) -> np.ndarray:
    task_ids = np.full(n_frames, TASK_IDX_CHOOSE, dtype=np.int64)
    if grasp_start < n_frames:
        task_ids[grasp_start:] = TASK_IDX_GRASP
    if lift_start is not None and lift_start < n_frames:
        task_ids[lift_start:] = TASK_IDX_LIFT
    return task_ids


def relabel_prompts(repo_id: str, root: str | None) -> None:
    dataset_root = Path(root) if root else HF_LEROBOT_HOME / repo_id

    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    state_names = info["features"]["observation.state"]["names"]
    try:
        finger_speed_idx = state_names.index("finger_speed")
        finger_pressure_idx = state_names.index("finger_pressure")
    except ValueError as exc:
        logger.error(f"Required observation.state feature not found: {exc}")
        return

    logger.info(f"finger_speed at obs index {finger_speed_idx}, finger_pressure at {finger_pressure_idx}")

    # ── 1. Rewrite meta/tasks.parquet ────────────────────────────────────────
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    shutil.copy2(tasks_path, tasks_path.with_suffix(".parquet.bak"))
    tasks_df = pd.DataFrame(
        {"task_index": list(range(len(PROMPT_TASKS)))},
        index=pd.Index(PROMPT_TASKS, name="task"),
    )
    tasks_df.to_parquet(tasks_path)
    logger.info(f"Wrote {len(PROMPT_TASKS)} tasks to meta/tasks.parquet")

    # ── 2. Rewrite task_index in every data parquet ───────────────────────────
    data_files = sorted(dataset_root.glob("data/**/*.parquet"))
    if not data_files:
        logger.error(f"No parquet files found under {dataset_root}/data/")
        return

    # Collect per-episode task sets for updating episode metadata later
    ep_task_ids: dict[int, set[int]] = {}

    for parquet_path in data_files:
        table = pq.read_table(parquet_path)
        task_idx_field = table.schema.get_field_index("task_index")
        obs_field = table.schema.get_field_index("observation.state")
        ep_field = table.schema.get_field_index("episode_index")

        if task_idx_field < 0 or obs_field < 0 or ep_field < 0:
            logger.warning(f"  {parquet_path.name}: missing required columns, skipping")
            continue

        episode_ids = table.column("episode_index").to_pylist()
        obs_list = table.column("observation.state").to_pylist()
        new_task_indices = list(table.column("task_index").to_pylist())

        episode_series = pd.Series(episode_ids)
        for ep_id, group in episode_series.groupby(episode_series):
            row_indices = group.index.tolist()
            obs = np.array([obs_list[i] for i in row_indices], dtype=np.float32)

            grasp_start, lift_start = _compute_prompt_boundaries(
                obs, finger_speed_idx, finger_pressure_idx
            )
            task_ids = _assign_task_indices(len(row_indices), grasp_start, lift_start)

            for local_i, global_i in enumerate(row_indices):
                new_task_indices[global_i] = int(task_ids[local_i])

            unique_ids = set(int(t) for t in np.unique(task_ids))
            ep_task_ids[int(ep_id)] = unique_ids
            logger.info(
                f"  ep {ep_id}: grasp_start={grasp_start}, lift_start={lift_start}, "
                f"phases={sorted(unique_ids)}"
            )

        shutil.copy2(parquet_path, parquet_path.with_suffix(".parquet.bak"))
        new_col = pa.array(new_task_indices, type=table.schema.field("task_index").type)
        table = table.set_column(task_idx_field, table.schema.field("task_index"), new_col)
        pq.write_table(table, parquet_path)
        logger.info(f"  wrote {parquet_path.name}")

    # ── 3. Rewrite tasks list in every meta/episodes parquet ─────────────────
    for ep_parquet in sorted(dataset_root.glob("meta/episodes/**/*.parquet")):
        if ep_parquet.suffix != ".parquet":
            continue
        ep_df = pd.read_parquet(ep_parquet)
        new_tasks_col = []
        for ep_id in ep_df["episode_index"]:
            used_ids = ep_task_ids.get(int(ep_id), {TASK_IDX_CHOOSE})
            # Preserve canonical phase order
            ordered = [PROMPT_TASKS[i] for i in sorted(used_ids)]
            new_tasks_col.append(ordered)
        shutil.copy2(ep_parquet, ep_parquet.with_suffix(".parquet.bak"))
        ep_df["tasks"] = new_tasks_col
        ep_df.to_parquet(ep_parquet, index=False)

    fix_meta_episodes_indices(dataset_root)

    # ── 4. Update info.json ───────────────────────────────────────────────────
    shutil.copy2(info_path, str(info_path) + ".bak")
    info["total_tasks"] = len(PROMPT_TASKS)
    info_path.write_text(json.dumps(info, indent=2))

    logger.info("Prompt relabeling complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch relabeling tools for LeRobot datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo_id", required=True, help="Dataset repo_id (e.g. myuser/mydata)")
    parser.add_argument("--root", default=None, help="Local dataset root (contains meta/, data/)")
    parser.add_argument(
        "--mode",
        choices=["finished", "prompts"],
        default="finished",
        help="finished: label last N frames as finished=1.0  |  prompts: rewrite phase-based language prompts",
    )
    # finished-mode options
    parser.add_argument("--num_frames", type=int, default=15, help="(finished mode) trailing frames to label finished=1.0")
    parser.add_argument("--recompute_stats", action="store_true", help="(finished mode) recompute dataset stats after relabeling")
    args = parser.parse_args()

    if args.mode == "finished":
        relabel_finished(args.repo_id, args.root, args.num_frames, args.recompute_stats)
    else:
        relabel_prompts(args.repo_id, args.root)


if __name__ == "__main__":
    main()
