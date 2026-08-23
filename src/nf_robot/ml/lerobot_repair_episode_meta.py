#!/usr/bin/env python

"""Point a merged dataset's episode metadata rows at the files they are actually in.

Every row in meta/episodes/* carries `meta/episodes/chunk_index` and
`meta/episodes/file_index`, naming the file that row lives in. lerobot reads an episode's
stats by looking those up (dataset_tools._load_episode_with_stats), so they have to be
true.

A merge breaks them. aggregate.py offsets them by a running counter:

    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"]  = df["meta/episodes/file_index"]  + meta_idx["file"]

while the write that follows concatenates into whichever destination file is still under
100MB. Episode metadata is a fraction of that - a tenth of a megabyte per source file -
so the counter never advances and everything lands in file-000, but rows from a source's
second metadata file keep saying file-001. That file is never created, and the first read
of such an episode fails with FileNotFoundError. lerobot does this correctly for the data
files in the same function, using an explicit src_to_dst mapping.

Nothing else in the row is touched, and the fix is idempotent: each file's rows are
stamped with that file's own indices, which is what they should have said.

Usage:
    python src/nf_robot/ml/lerobot_repair_episode_meta.py \
        --root /home/nhn/data_scratch/combined_targets
"""

import argparse
import logging
import re
from pathlib import Path

CHUNK_FILE = re.compile(r"chunk-(\d+)/file-(\d+)\.parquet$")


def episode_meta_files(root):
    """Every meta/episodes parquet under root, with the (chunk, file) its name declares."""
    for path in sorted(Path(root).glob("meta/episodes/chunk-*/file-*.parquet")):
        match = CHUNK_FILE.search(path.as_posix())
        if match:
            yield path, int(match.group(1)), int(match.group(2))


def repair(root, dry_run=False):
    """Rewrite the location columns of every episode metadata file. Returns rows changed."""
    import pandas as pd

    changed = 0
    for path, chunk, file_index in episode_meta_files(root):
        df = pd.read_parquet(path)
        wrong = ((df["meta/episodes/chunk_index"] != chunk)
                 | (df["meta/episodes/file_index"] != file_index))
        if not wrong.any():
            logging.info(f"{path.relative_to(root)}: {len(df)} rows already correct")
            continue
        episodes = df.loc[wrong, "episode_index"]
        logging.info(f"{path.relative_to(root)}: {int(wrong.sum())} of {len(df)} rows "
                     f"(episodes {episodes.min()}-{episodes.max()}) point elsewhere; "
                     f"{'would set' if dry_run else 'setting'} them to "
                     f"chunk {chunk} file {file_index}")
        changed += int(wrong.sum())
        if dry_run:
            continue
        df.loc[wrong, "meta/episodes/chunk_index"] = chunk
        df.loc[wrong, "meta/episodes/file_index"] = file_index
        df.to_parquet(path)
    return changed


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="Dataset folder holding meta/, data/ and videos/")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what is wrong and change nothing")
    args = parser.parse_args()

    changed = repair(Path(args.root), args.dry_run)
    if changed:
        logging.info(f"{changed} row(s) {'would be' if args.dry_run else ''} repaired")
    else:
        logging.info("nothing to repair")


if __name__ == "__main__":
    main()
