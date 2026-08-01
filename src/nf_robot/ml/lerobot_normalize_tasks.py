#!/usr/bin/env python

"""Collapse a LeRobot dataset's task strings onto a canonical set.

Datasets recorded with speech-to-text accumulate many spellings of the same
instruction ("put laundry in the hamburger"), and every distinct string is a
distinct conditioning signal to a language-conditioned policy. This rewrites the
task table so only the canonical strings remain.

The remap is done per task_index rather than per episode, so episodes whose
frames carry several different tasks keep that structure - each frame follows
its own task to whichever canonical string it maps to.

Only meta/ and data/ are touched; the videos (the bulk of a dataset) are never
downloaded or re-uploaded.

Mapping file (YAML or JSON):

    tasks:                      # canonical strings; order sets task_index 0..N-1
      - "Put laundry in the hamper"
      - "Put trash in the trash can"
    map:                        # every raw string in the dataset, incl. canonical ones
      "put laundry in the hamper": "Put laundry in the hamper"
      "put laundry in the hamburger": "Put laundry in the hamper"
      "put trash in trash can": "Put trash in the trash can"

Usage:
    python src/nf_robot/ml/lerobot_normalize_tasks.py \
        --repo_id naavox/move_clutter_rect \
        --mapping src/nf_robot/ml/recipes/move_clutter_rect_tasks.yaml \
        --root /tmp/move_clutter_rect_tasks \
        [--push]
"""

import argparse
import collections
import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


def load_mapping(path: Path) -> tuple[list[str], dict[str, str]]:
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        spec = yaml.safe_load(text)
    else:
        spec = json.loads(text)

    tasks = spec.get("tasks")
    mapping = spec.get("map")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Mapping file needs a non-empty 'tasks' list")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Mapping file needs a non-empty 'map' mapping")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"Duplicate canonical tasks: {tasks}")

    unknown = sorted(set(mapping.values()) - set(tasks))
    if unknown:
        raise ValueError(f"'map' targets strings that are not in 'tasks': {unknown}")
    return tasks, mapping


def normalize_tasks(root: Path, tasks: list[str], mapping: dict[str, str]) -> None:
    old_tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    # tasks.parquet is indexed by the task string, with task_index as a column.
    old_by_index = {int(row.task_index): str(name) for name, row in old_tasks.iterrows()}

    missing = sorted(set(old_by_index.values()) - set(mapping))
    if missing:
        raise ValueError(
            f"{len(missing)} task string(s) in the dataset have no entry in the mapping file:\n  "
            + "\n  ".join(repr(m) for m in missing)
        )

    new_index_of = {task: i for i, task in enumerate(tasks)}
    remap = {old_i: new_index_of[mapping[name]] for old_i, name in old_by_index.items()}

    data_files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files under {root}/data")

    before = collections.Counter()
    after = collections.Counter()

    for f in data_files:
        table = pq.read_table(f)
        col_idx = table.schema.get_field_index("task_index")
        field = table.schema.field(col_idx)
        old_col = table.column("task_index").to_pylist()
        for i in old_col:
            before[old_by_index[int(i)]] += 1
        new_col = [remap[int(i)] for i in old_col]
        for i in new_col:
            after[tasks[i]] += 1
        table = table.set_column(col_idx, field, pa.array(new_col, type=field.type))
        pq.write_table(table, f)

    # meta/episodes carries a per-episode list of the task strings its frames use.
    for f in sorted(root.glob("meta/episodes/chunk-*/file-*.parquet")):
        table = pq.read_table(f)
        col_idx = table.schema.get_field_index("tasks")
        field = table.schema.field(col_idx)
        new_lists = []
        for entry in table.column("tasks").to_pylist():
            names = [entry] if isinstance(entry, str) else list(entry)
            # dict.fromkeys dedupes while keeping first-seen order
            new_lists.append(list(dict.fromkeys(mapping[n] for n in names)))
        if pa.types.is_string(field.type):
            # Single-task-per-episode schema: only collapsible if it stays single.
            if any(len(v) != 1 for v in new_lists):
                raise ValueError("Episode 'tasks' column is a plain string but some episodes map to several tasks")
            new_col = pa.array([v[0] for v in new_lists], type=field.type)
        else:
            new_col = pa.array(new_lists, type=field.type)
        table = table.set_column(col_idx, field, new_col)
        pq.write_table(table, f)

    new_tasks = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task"))
    new_tasks.to_parquet(root / "meta" / "tasks.parquet")

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(tasks)
    info_path.write_text(json.dumps(info, indent=4))

    total = sum(before.values())
    print(f"\nBefore: {len(before)} task strings over {total} frames")
    for name, n in before.most_common():
        print(f"  {n:8d} frames ({100 * n / total:5.1f}%)  {name!r}  ->  {mapping[name]!r}")
    print(f"\nAfter: {len(after)} task strings")
    for name, n in after.most_common():
        print(f"  {n:8d} frames ({100 * n / total:5.1f}%)  {name!r}")


def verify(repo_id: str, root: Path, tasks: list[str]) -> None:
    """Confirm frames resolve to canonical strings through lerobot's own lookup.

    Uses LeRobotDatasetMetadata plus the parquet columns rather than
    LeRobotDataset, whose constructor would download the videos. The lookup
    mirrors DatasetReader.get_item: tasks.iloc[task_index].name, which is
    positional - so a task table whose row order disagrees with task_index would
    silently mislabel every frame, and this catches that.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    if sorted(meta.tasks.index) != sorted(tasks):
        raise ValueError(f"Rewritten task table is {list(meta.tasks.index)}, expected {tasks}")

    seen = set()
    for f in sorted(root.glob("data/chunk-*/file-*.parquet")):
        for task_idx in set(pq.read_table(f, columns=["task_index"]).column("task_index").to_pylist()):
            seen.add(meta.tasks.iloc[int(task_idx)].name)
    unexpected = seen - set(tasks)
    if unexpected:
        raise ValueError(f"Frames still resolve to non-canonical tasks: {sorted(unexpected)}")
    print(f"\nVerified: every frame resolves to one of {sorted(seen)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="dataset repo id to normalize")
    parser.add_argument("--mapping", required=True, help="YAML/JSON mapping file")
    parser.add_argument("--root", required=True, help="local working copy (meta + data only, no videos)")
    parser.add_argument("--push", action="store_true", help="upload the rewritten meta/ and data/ to the Hub")
    args = parser.parse_args()

    tasks, mapping = load_mapping(Path(args.mapping))
    root = Path(args.root)

    logging.info(f"Downloading meta/ and data/ of '{args.repo_id}' to {root} (videos skipped)")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=["meta/**", "data/**"],
    )

    normalize_tasks(root, tasks, mapping)
    verify(args.repo_id, root, tasks)

    if args.push:
        logging.info(f"Uploading rewritten meta/ and data/ to '{args.repo_id}'")
        HfApi().upload_folder(
            folder_path=root,
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=["meta/**", "data/**"],
            commit_message=f"Normalize task strings onto {len(tasks)} canonical prompts",
        )
        print(f"Pushed. https://huggingface.co/datasets/{args.repo_id}")
    else:
        print(f"\nNot pushed. Re-run with --push to upload {root}/meta and {root}/data.")


if __name__ == "__main__":
    main()
