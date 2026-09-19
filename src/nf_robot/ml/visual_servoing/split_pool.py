#!/usr/bin/env python

"""Deal the mined pool into train and eval, one row at a time and at random.

    python -m nf_robot.ml.visual_servoing.split_pool \
        --data_root datasets/visual_servoing_pool_252
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from nf_robot.ml.visual_servoing.mine_teleop import (
    POOL_SPLIT, ROW_GROUP_SIZE, SHARD_TARGET_BYTES, row_schema, write_dataset_card,
)

DEFAULT_EVAL_FRACTION = 0.1


def pool_shards(root: Path):
    """The pool's parquet shards, in a fixed order so a seed means one deal."""
    pool = Path(root) / POOL_SPLIT
    shards = sorted(pool.glob("*.parquet"))
    if not shards:
        raise SystemExit(
            f"No parquet shards in {pool}. Mine and composite into the pool first:\n"
            f"  python -m nf_robot.ml.visual_servoing.mine_teleop --output_root {root} ...\n"
            f"  python -m nf_robot.ml.visual_servoing.synth_frames --output_root {root} ...")
    return shards


def normalized(table):
    """One shard under the canonical schema, with missing label columns filled as nulls."""
    import pyarrow as pa

    schema = row_schema()
    columns = [
        table.column(field.name).cast(field.type) if field.name in table.schema.names
        else pa.nulls(table.num_rows, field.type)
        for field in schema
    ]
    return pa.Table.from_arrays(columns, schema=schema)


def sample_group(row):
    """What a row is a near-duplicate of: one teleop episode, or one synthetic plate."""
    return (row.get("split_source"), row.get("source_repo_id"), row.get("episode_index"))


class ShardBuffer:
    """Accumulates arrow tables and writes them out at roughly SHARD_TARGET_BYTES."""

    def __init__(self, split_dir: Path, prefix="shard", target_bytes=SHARD_TARGET_BYTES):
        self.split_dir = split_dir
        self.prefix = prefix
        self.target_bytes = target_bytes
        self.tables = []
        self.pending = 0
        self.shards = 0
        self.total = 0

    def add(self, table):
        if not table.num_rows:
            return
        self.tables.append(table)
        self.pending += table.nbytes
        self.total += table.num_rows
        if self.pending >= self.target_bytes:
            self.flush()

    def flush(self):
        if not self.tables:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.split_dir / f"{self.prefix}-{self.shards:04d}.parquet"
        pq.write_table(pa.concat_tables(self.tables), path,
                       compression="snappy", row_group_size=ROW_GROUP_SIZE)
        logging.info(f"wrote {path.name}: {sum(t.num_rows for t in self.tables)} rows, "
                     f"{self.pending / 1e6:.0f} MB")
        self.shards += 1
        self.tables = []
        self.pending = 0


def split_pool(root, eval_fraction=DEFAULT_EVAL_FRACTION, seed=0):
    """Deal every row of the pool into train and eval a shard at a time, returning (train
    rows, eval rows)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    shards = pool_shards(root)
    counts = [pq.read_metadata(shard).num_rows for shard in shards]
    total = sum(counts)
    if not total:
        raise SystemExit(f"{root / POOL_SPLIT} holds {len(shards)} shard(s) and no rows")

    cut = int(round(total * eval_fraction))
    chosen = np.random.default_rng(seed).permutation(total)[:cut]
    is_eval = np.zeros(total, dtype=bool)
    is_eval[chosen] = True
    logging.info(f"{total} row(s) in {len(shards)} pool shard(s): "
                 f"{total - cut} train, {cut} eval (fraction {eval_fraction:g}, seed {seed})")

    writers = {}
    for split in ("train", "eval"):
        split_dir = root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        writers[split] = ShardBuffer(split_dir)

    groups = {"train": set(), "eval": []}
    at = 0
    for shard, count in zip(shards, counts):
        table = normalized(pq.read_table(shard))
        mask = is_eval[at:at + count]
        at += count
        provenance = table.select(["split_source", "source_repo_id", "episode_index"]).to_pylist()
        for split, wanted in (("train", ~mask), ("eval", mask)):
            writers[split].add(table.filter(pa.array(wanted)))
            keys = (sample_group(r) for r, w in zip(provenance, wanted) if w)
            if split == "train":
                groups["train"].update(keys)
            else:
                groups["eval"].extend(keys)

    for split, writer in writers.items():
        writer.flush()
        logging.info(f"{split}: {writer.total} row(s) in {writer.shards} shard(s)")
    write_dataset_card(root)

    shared = sum(1 for key in groups["eval"] if key in groups["train"])
    if groups["eval"]:
        logging.info(
            f"{shared} of {len(groups['eval'])} eval row(s) come from an episode or plate that "
            f"also appears in train, so they are near-duplicates of something trained on; "
            f"see split_pool for what that makes the eval numbers mean")
    return writers["train"].total, writers["eval"].total


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True,
                        help=f"Dataset root holding the {POOL_SPLIT}/ pool")
    parser.add_argument("--eval_fraction", type=float, default=DEFAULT_EVAL_FRACTION)
    parser.add_argument("--seed", type=int, default=0,
                        help="The same seed over the same pool deals the same split. Adding "
                             "to the pool re-deals all of it, so metrics either side of a "
                             "rebuild compare two different eval sets")
    args = parser.parse_args()
    split_pool(Path(args.data_root), args.eval_fraction, args.seed)


if __name__ == "__main__":
    main()
