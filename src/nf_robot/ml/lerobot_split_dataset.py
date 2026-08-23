#!/usr/bin/env python

"""Split a LeRobot dataset into random train and eval halves.

A thin wrapper over `lerobot-edit-dataset --operation.type split`, which this exists
because of: handed fractions, that command does not sample. It turns 0.1 into a
contiguous range of episode indices - the last tenth, in order - and a merged dataset
lays its sources down in recipe order, so the tail is whichever sources sit at the bottom
of the recipe rather than a spread across the collection. Handed explicit indices it does
exactly as it is told, so this shuffles them first.

It also reads total_episodes out of the dataset's own metadata, so splitting does not
start with looking that up.

The split is seeded and the seed is printed, so the same command twice is the same split
- which matters when the eval half is mined hours after the train half and the two must
not overlap.

Outputs are named after the splits, `<repo_id>_train` and `<repo_id>_eval`, in the
LeRobot home unless --new_root says otherwise. Downstream tools then find them by repo id
with no path juggling; see visual_servoing/readme.md step 3. --upload pushes both to the
hub under those same names, replacing whatever is already there.

Usage:
    python src/nf_robot/ml/lerobot_split_dataset.py \
        --repo_id naavox/combined_targets \
        --root /home/nhn/data_scratch/combined_targets --upload
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

# The AV1 encoder prints a twenty line configuration banner per video file, straight to
# stderr from libSvtAv1Enc rather than through ffmpeg's log system, so the split's own
# progress is unreadable without this. 1 is the library's "errors only" level; export
# SVT_LOG to keep the banner.
os.environ.setdefault("SVT_LOG", "1")

def dataset_root(repo_id, root=None):
    """The dataset's directory, from --root or the LeRobot home.

    The home comes from lerobot rather than a copy of its default, so that
    HF_LEROBOT_HOME points this at the same place it points everything else. Imported
    here rather than at module scope: --root is the common case and does not need it.
    """
    if root:
        return Path(root)
    from lerobot.utils.constants import HF_LEROBOT_HOME

    return HF_LEROBOT_HOME / repo_id


def split_outputs(repo_id, new_root=None):
    """Where each half lands, as [(split repo id, directory), ...].

    Mirrors what lerobot's split operation does with the names: `<repo_id>_<split>` in the
    LeRobot home, or `<new_root>/<split>` when one is given.
    """
    outputs = []
    for name in ("train", "eval"):
        split_repo = f"{repo_id}_{name}"
        if new_root:
            outputs.append((split_repo, Path(new_root) / name))
        else:
            from lerobot.utils.constants import HF_LEROBOT_HOME

            outputs.append((split_repo, HF_LEROBOT_HOME / split_repo))
    return outputs


def total_episodes(root):
    """How many episodes a built dataset holds, from its own metadata."""
    info = Path(root) / "meta" / "info.json"
    if not info.exists():
        raise SystemExit(f"no {info}; --root should be the folder holding meta/, data/ and videos/")
    return int(json.loads(info.read_text())["total_episodes"])


def random_splits(count, eval_fraction, seed):
    """{split name: episode indices}, sampled rather than sliced.

    Sorted within each split only for legibility - membership is what the split is, and
    the order episodes are named in does not reach the output.
    """
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    cut = count - max(1, round(count * eval_fraction))
    return {"train": sorted(indices[:cut]), "eval": sorted(indices[cut:])}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="Dataset to split")
    parser.add_argument("--root", default=None,
                        help="Its folder on disk (defaults to the LeRobot home)")
    parser.add_argument("--new_root", default=None,
                        help="Where the two halves go (defaults to the LeRobot home, "
                             "named <repo_id>_train and <repo_id>_eval)")
    parser.add_argument("--eval_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0,
                        help="Reruns with the same seed give the same split, which is "
                             "what lets the two halves be mined at different times")
    parser.add_argument("--upload", action="store_true",
                        help="Push both halves to the hub as <repo_id>_train and "
                             "<repo_id>_eval, replacing whatever is at those ids")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the split and the command, and stop")
    args = parser.parse_args()

    root = dataset_root(args.repo_id, args.root)
    count = total_episodes(root)
    splits = random_splits(count, args.eval_fraction, args.seed)
    logging.info(f"{count} episodes in {root}: {len(splits['train'])} train, "
                 f"{len(splits['eval'])} eval (seed {args.seed})")
    logging.info(f"eval episodes: {splits['eval']}")

    command = [
        "lerobot-edit-dataset",
        "--repo_id", args.repo_id,
        "--root", str(root),
        "--operation.type", "split",
        "--operation.splits", json.dumps(splits),
    ]
    if args.new_root:
        command += ["--new_root", args.new_root]

    outputs = split_outputs(args.repo_id, args.new_root)
    if args.dry_run:
        logging.info(" ".join(command))
        for split_repo, path in outputs:
            logging.info(f"would write {path}" + (f", upload to {split_repo}" if args.upload else ""))
        return

    returncode = subprocess.call(command)
    if returncode or not args.upload:
        sys.exit(returncode)

    # Reused rather than reimplemented: a plain upload_folder publishes a dataset with no
    # frames in it (os.walk does not follow the symlinked video directory) and no version
    # tag (which is how lerobot resolves a repo at all). Both are silent.
    from nf_robot.ml.visual_servoing.recover_spin import upload_dataset

    for split_repo, path in outputs:
        logging.info(f"uploading {path} to {split_repo}")
        upload_dataset(path, split_repo, what=f"a {args.eval_fraction:g} random split "
                                              f"of {args.repo_id}, seed {args.seed}")


if __name__ == "__main__":
    main()
