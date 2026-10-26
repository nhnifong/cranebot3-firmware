#!/usr/bin/env python

"""Deal an ortho_target pool into train and a held-out eval of one unseen room (James's).

`python -m nf_robot.ml.ortho_target split` deals rows at random, so eval frames are near
duplicates of train frames from the same room and say nothing about a room the model has never
seen, which is what synthetic data is meant to help with. This holds out every frame of James's
room instead and puts everything else in train. Use it in place of `split`.

It then swaps in the corrected labels of those frames from ortho_target_user_labels_james_eval/.
The originals live in naavox/ and justink04/ortho-target-user-labels on the hub, so every
merge_labels brings the old ones back, and this re-applies the fixes each time.

James's room is:
- grip_o episodes of naavox/james-sep22 and naavox/james-oct15 (ranges below, by merge order in
  recipes/grip_o.yaml; appending new sources to the end of that recipe leaves them unchanged)
- hand labels with "james" in their name
- JAMES_HAND: UI and untagged labeler frames identified by picture, since their names say
  nothing about the room

Usage (from the repo root):
    python experiments/anchor_diffusion/james_holdout.py --data_root datasets/ortho_target_complete
"""

import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from nf_robot.ml.ortho_target.dataset import (POOL_SPLIT, is_complete, read_rows, write_dataset_readme,
                                              write_row_shards)

JAMES_EPISODES = [(1027, 1081), (1228, 1240)]  # grip_o: james-sep22, james-oct15
JAMES_HAND = {
    "user-000000-000250.jpg", "user-000030-000244.jpg", "user-000054-000519.jpg",
    "user-20260922-141400-d2e75f.jpg", "user-20260922-183846-94ea6d.jpg",
    "user-20260922-190143-4f24f9.jpg", "user-20261015-115938-7fc351.jpg",
    "user-20261015-133257-7f4181.jpg", "user-20261015-134312-9fd2c0.jpg",
}
CORRECTED = Path("ortho_target_user_labels_james_eval")
CORRECTED_PREFIX = "user-james_eval-"


def is_james(row):
    name = row["file_name"]
    if name.startswith("ep"):
        return any(a <= row["episode_index"] <= b for a, b in JAMES_EPISODES)
    return "james" in name or name in JAMES_HAND


def corrected_rows():
    """Relabeled frames, keyed by the file_name of the row they replace."""
    fixed = {}
    for path in sorted(CORRECTED.glob(f"{CORRECTED_PREFIX}*.parquet")):
        row = pq.read_table(path).to_pylist()[0]
        original = "user-" + path.stem[len(CORRECTED_PREFIX):] + ".jpg"
        fixed[original] = dict(row, file_name=original)
    return fixed


def rewrite(split_dir, rows):
    tmp = split_dir.with_name(split_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    write_row_shards(tmp, rows)
    shutil.rmtree(split_dir, ignore_errors=True)
    tmp.rename(split_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()

    root = Path(args.data_root)
    fixed = corrected_rows()
    rows = read_rows(root / POOL_SPLIT)
    replaced = sum(r["file_name"] in fixed for r in rows)
    rows = [fixed.get(r["file_name"], r) for r in rows]
    rewrite(root / POOL_SPLIT, rows)
    print(f"{root / POOL_SPLIT}: swapped in {replaced} of {len(fixed)} corrected James label(s)")

    picked = {"eval": [r for r in rows if is_james(r)], "train": [r for r in rows if not is_james(r)]}
    for split, chosen in picked.items():
        rewrite(root / split, chosen)
        complete = sum(is_complete(r) for r in chosen)
        print(f"{root / split}: {len(chosen)} frame(s), {complete} complete, "
              f"{sum(len(r['points']) for r in chosen)} target(s)")
    write_dataset_readme(root)


if __name__ == "__main__":
    main()
