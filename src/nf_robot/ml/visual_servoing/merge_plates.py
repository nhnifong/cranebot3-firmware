#!/usr/bin/env python

"""Merge plate captures from several machines into one collection.

Every capture is already a self-contained set of files named after a run id that carries
the kind, the moment and six random hex digits, so merging is copying files and
concatenating manifests. Runs are never combined, rewritten or resampled: whatever the
camera produced stays exactly as it was, which is what lets the matte and compositing
steps be redone later without going back to the robots.

Runs already present in the destination are skipped by run id, so merging the same
source twice is harmless and an interrupted merge can be rerun.

Usage:
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all --from plates
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all \
        --from /mnt/contractor/plates --dry_run
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all --list
"""

import argparse
import collections
import json
import logging
import shutil
from pathlib import Path

from nf_robot.ml.visual_servoing.plates import MANIFEST_NAME, read_manifest, run_files


def merge(sources, into: Path, move=False, dry_run=False):
    """Copy every run from each source into one directory, skipping ones already there."""
    into = Path(into)
    if not dry_run:
        into.mkdir(parents=True, exist_ok=True)

    known = {entry["run_id"] for entry in read_manifest(into)}
    added, skipped, broken = [], 0, 0

    for source in sources:
        source = Path(source)
        entries = read_manifest(source)
        if not entries:
            logging.warning(f"{source}: no {MANIFEST_NAME}, nothing to merge")
            continue
        for entry in entries:
            files = run_files(entry)
            missing = [f for f in files if not (source / f).exists()]
            if missing:
                logging.warning(f"{entry['run_id']}: missing {missing} in {source}; skipping")
                broken += 1
                continue
            if entry["run_id"] in known:
                skipped += 1
                continue

            size = sum((source / f).stat().st_size for f in files)
            logging.info(f"{'would add' if dry_run else 'adding'} {entry['run_id']} "
                         f"({entry['kind']}, {size / 1e6:.0f} MB) from {source}")
            if not dry_run:
                for name in files:
                    if move:
                        shutil.move(str(source / name), str(into / name))
                    else:
                        shutil.copy2(source / name, into / name)
            known.add(entry["run_id"])
            added.append({**entry, "merged_from": str(source)})

    if added and not dry_run:
        with open(into / MANIFEST_NAME, "a") as f:
            for entry in added:
                f.write(json.dumps(entry) + "\n")

    logging.info(f"{'would merge' if dry_run else 'merged'} {len(added)} run(s) into {into}; "
                 f"{skipped} already present, {broken} incomplete")
    return added


def summarize(directory: Path):
    """What a collection holds, by kind and by who captured it."""
    entries = read_manifest(directory)
    if not entries:
        logging.info(f"{directory}: empty")
        return

    by_kind = collections.Counter(e["kind"] for e in entries)
    by_source = collections.Counter(
        f"{e.get('robot_id') or '?'}@{e.get('host') or '?'}" for e in entries)
    total = 0
    for entry in entries:
        total += sum((Path(directory) / f).stat().st_size
                     for f in run_files(entry) if (Path(directory) / f).exists())

    print(f"{directory}: {len(entries)} runs, {total / 1e9:.2f} GB")
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind:14s} {count:4d} runs")
    print("  by capturer:")
    for source, count in sorted(by_source.items()):
        print(f"    {source:32s} {count:4d} runs")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--into", required=True, help="Destination collection")
    parser.add_argument("--from", dest="sources", nargs="*", default=[],
                        help="Collections to merge in")
    parser.add_argument("--move", action="store_true",
                        help="Move the run files instead of copying them")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="Summarize the destination and exit")
    args = parser.parse_args()

    if args.list or not args.sources:
        summarize(Path(args.into))
        return
    merge(args.sources, Path(args.into), args.move, args.dry_run)
    summarize(Path(args.into))


if __name__ == "__main__":
    main()
