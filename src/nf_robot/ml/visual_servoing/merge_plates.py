#!/usr/bin/env python

"""Collect plate captures from several machines, and keep them somewhere everyone can see.

Every capture is already a self-contained set of files named after a run id that carries
the kind, the moment and six random hex digits, so merging is copying files and
concatenating manifests. Runs are never combined, rewritten or resampled: whatever the
camera produced stays exactly as it was, which is what lets the matte and compositing
steps be redone later without going back to the robots.

Runs already present in the destination are skipped by run id, so merging the same
source twice is harmless, an interrupted merge can be rerun, and a hub dataset can be
both a source and the upload target of the same collection.

Only raw captures move: the manifest and the files it names. Mattes and previews sit in
subdirectories of the same collection and are deliberately left behind everywhere here -
they are derived by keying thresholds still being tuned, so a copy of one is a stale
answer to a question that will be asked again. Rerun finger_matte and object_matte on
whatever collection you end up with.

Usage:
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all --from plates
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all \
        --from /mnt/contractor/plates nathanielnifong/stringman-plates --dry_run
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all --list
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all --prune
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all \
        --upload nathanielnifong/stringman-plates
"""

import argparse
import collections
import json
import logging
import shutil
from pathlib import Path

from nf_robot.ml.visual_servoing.plates import MANIFEST_NAME, read_manifest, run_files

# Sources given as "hf://owner/name" are always the hub; a bare "owner/name" is the hub
# only when there is no directory by that name, so a local path always wins.
HF_PREFIX = "hf://"

# Where a merge is reading from. `present` is False for a manifest fetched without its
# runs, `remote` marks a huggingface cache directory, whose files are shared symlinks that
# must be copied out of rather than moved, and `sizes` is {filename: bytes} read off the
# hub when the files themselves are not here to measure.
Source = collections.namedtuple("Source", "dir present remote sizes", defaults=(None,))


def plate_count(entry):
    """How many plates a run holds.

    Video runs record telemetry samples rather than a frame count, because counting frames
    means decoding the stream. The two are captured together over the same sweep, so
    samples are the honest stand-in for a number that would otherwise cost a decode.
    """
    return int(entry.get("frames") or entry.get("samples") or 0)


def raw_files(entries):
    """The manifest and every file it names - the raw captures of a collection, and nothing
    else. Anything not in this list is derived and gets rebuilt rather than moved."""
    return [MANIFEST_NAME] + [f for entry in entries for f in run_files(entry)]


def _hub_manifest(repo_id):
    """The manifest already on the hub, or [] for a repo that has none yet."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=MANIFEST_NAME)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    return read_manifest(Path(path).parent)


def _hub_sizes(repo_id):
    """{filename: bytes} for a hub dataset, or {} if the listing cannot be had.

    One metadata call, no download - which is what lets a dry run report what a merge
    would actually cost. Without it the only honest answer is that the size is unknown,
    since a manifest-only pull has nothing on disk to measure.
    """
    from huggingface_hub import HfApi

    try:
        return {item.path: item.size for item in
                HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
                if getattr(item, "size", None) is not None}
    except Exception as error:
        logging.debug(f"{repo_id}: could not list file sizes ({error})")
        return {}


def _pull(repo_id, manifest_only=False):
    """Fetch a hub collection into the usual cache.

    The manifest comes down first and names the files worth pulling, which is what keeps a
    matted collection's cutouts on the hub from being dragged along with the captures.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    entries = _hub_manifest(repo_id)
    if not entries:
        logging.warning(f"{repo_id}: no {MANIFEST_NAME} on the hub")
        return Source(None, False, True)
    if manifest_only:
        path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=MANIFEST_NAME)
        return Source(Path(path).parent, False, True, _hub_sizes(repo_id))

    logging.info(f"pulling {len(entries)} run(s) of {repo_id} from the hub")
    root = snapshot_download(repo_id=repo_id, repo_type="dataset",
                             allow_patterns=raw_files(entries))
    return Source(Path(root), True, True)


def resolve_source(source, manifest_only=False):
    """Where to read a source named as a path or as a hub dataset repo id."""
    text = str(source)
    if text.startswith(HF_PREFIX):
        return _pull(text[len(HF_PREFIX):], manifest_only)
    if Path(text).is_dir():
        return Source(Path(text), True, False)
    if text.count("/") == 1 and not text.startswith("."):
        return _pull(text, manifest_only)
    raise FileNotFoundError(f"{source} is neither a directory nor an owner/name repo id")


def merge(sources, into: Path, move=False, dry_run=False):
    """Copy every run from each source into one directory, skipping ones already there.

    A source is a directory or a hub dataset repo id, which is pulled first. A dry run
    against the hub fetches only the manifest, so asking what a merge would cost does not
    cost the download itself.
    """
    into = Path(into)
    if not dry_run:
        into.mkdir(parents=True, exist_ok=True)

    known = {entry["run_id"] for entry in read_manifest(into)}
    added, skipped, broken = [], 0, 0
    total_bytes = 0

    for name in sources:
        source = resolve_source(name, manifest_only=dry_run)
        if source.dir is None:
            continue
        entries = read_manifest(source.dir)
        if not entries:
            logging.warning(f"{name}: no {MANIFEST_NAME}, nothing to merge")
            continue
        # the cache copy is shared with every other reader of that dataset
        taking = move and not source.remote
        for entry in entries:
            files = run_files(entry)
            missing = [f for f in files if source.present and not (source.dir / f).exists()]
            if missing:
                logging.warning(f"{entry['run_id']}: missing {missing} in {name}; skipping")
                broken += 1
                continue
            if entry["run_id"] in known:
                skipped += 1
                continue

            if source.present:
                size = sum((source.dir / f).stat().st_size for f in files)
            elif source.sizes:
                size = sum(source.sizes.get(f, 0) for f in files)
            else:
                # a manifest-only pull from a hub that would not list its files
                size = None
            total_bytes += size or 0
            logging.info(f"{'would add' if dry_run else 'adding'} {entry['run_id']} "
                         f"({entry['kind']}, "
                         f"{f'{size / 1e6:.0f} MB' if size is not None else 'size unknown'})"
                         f" from {name}")
            if not dry_run:
                for file in files:
                    if taking:
                        shutil.move(str(source.dir / file), str(into / file))
                    else:
                        shutil.copy2(source.dir / file, into / file)
            known.add(entry["run_id"])
            added.append({**entry, "merged_from": str(name)})

    if added and not dry_run:
        with open(into / MANIFEST_NAME, "a") as f:
            for entry in added:
                f.write(json.dumps(entry) + "\n")

    logging.info(f"{'would merge' if dry_run else 'merged'} {len(added)} run(s), "
                 f"{total_bytes / 1e9:.2f} GB, into {into}; "
                 f"{skipped} already present, {broken} incomplete")
    if dry_run and added:
        logging.info("dry run: hub sources were listed but not downloaded, and nothing "
                     "was written; rerun without --dry_run to fetch and merge")
    return added


def summarize(directory: Path):
    """What a collection holds, by kind and by who captured it."""
    directory = Path(directory)
    entries = read_manifest(directory)
    if not entries:
        logging.info(f"{directory}: empty")
        return

    runs = collections.Counter()
    plates = collections.Counter()
    for entry in entries:
        runs[entry["kind"]] += 1
        plates[entry["kind"]] += plate_count(entry)
    by_source = collections.Counter(
        f"{e.get('robot_id') or '?'}@{e.get('host') or '?'}" for e in entries)

    total = 0
    incomplete = []
    for entry in entries:
        for name in run_files(entry):
            if (directory / name).exists():
                total += (directory / name).stat().st_size
            else:
                incomplete.append(entry["run_id"])

    print(f"{directory}: {len(entries)} runs, {sum(plates.values())} plates, "
          f"{total / 1e9:.2f} GB")
    for kind in sorted(runs):
        print(f"  {kind:14s} {runs[kind]:4d} runs  {plates[kind]:7d} plates")
    print("  by capturer:")
    for source, count in sorted(by_source.items()):
        print(f"    {source:32s} {count:4d} runs")

    if incomplete:
        print(f"  {len(set(incomplete))} run(s) with missing files: --prune drops them")
    strays = _strays(directory, entries)
    if strays:
        print(f"  {len(strays)} capture file(s) no manifest mentions: {sorted(strays)[:4]}")


def _strays(directory, entries):
    """Capture files in a collection that no manifest entry claims.

    An aborted run leaves its video behind without ever writing a manifest line, and
    nothing can read those bytes without one - but they are still the only copy of
    something a robot did, so they are reported rather than deleted.
    """
    directory = Path(directory)
    claimed = set(raw_files(entries))
    found = {p.name for pattern in ("*.parquet", "*.ts", "*.jsonl")
             for p in directory.glob(pattern)}
    return found - claimed


def prune(directory: Path, dry_run=False):
    """Drop manifest entries whose files are gone.

    Deleting a bad capture is a normal thing to do between the capture and the matte, and
    it leaves the manifest describing a run nothing can read: every reader here skips
    those, but --list would still count them and the numbers would be a lie.
    """
    directory = Path(directory)
    entries = read_manifest(directory)
    kept, dropped = [], []
    for entry in entries:
        missing = [f for f in run_files(entry) if not (directory / f).exists()]
        if not missing:
            kept.append(entry)
            continue
        dropped.append(entry)
        logging.info(f"{'would drop' if dry_run else 'dropping'} {entry['run_id']} "
                     f"({entry['kind']}), missing {missing}")
    if dropped and not dry_run:
        path = directory / MANIFEST_NAME
        shutil.copy2(path, path.with_suffix(".jsonl.bak"))
        with open(path, "w") as f:
            for entry in kept:
                f.write(json.dumps(entry) + "\n")
        logging.info(f"pruned {len(dropped)} entr(ies); previous {MANIFEST_NAME} kept as .bak")
    elif not dropped:
        logging.info(f"{directory}: every manifest entry has its files")
    return dropped


def upload(directory: Path, repo_id: str, dry_run=False):
    """Push a collection's raw captures to a hub dataset, adding to what is already there.

    The hub copy is the shared one, so this only ever adds: runs it already has are left
    alone and its manifest lines are carried into the new manifest verbatim, which is what
    lets two machines upload their own captures to one dataset without either erasing the
    other. Mattes stay local - see the module docstring.
    """
    from huggingface_hub import HfApi, create_repo

    directory = Path(directory)
    local = read_manifest(directory)
    if not local:
        raise ValueError(f"no {MANIFEST_NAME} in {directory}; nothing to upload")

    missing = {e["run_id"] for e in local
               for f in run_files(e) if not (directory / f).exists()}
    if missing:
        raise ValueError(f"{directory} is missing the files of {sorted(missing)}. Uploading "
                         f"would publish a manifest naming runs nobody can read; --prune first.")

    hub = _hub_manifest(repo_id)
    known = {e["run_id"] for e in hub}
    new = [e for e in local if e["run_id"] not in known]
    files = [f for e in new for f in run_files(e)]
    size = sum((directory / f).stat().st_size for f in files)

    logging.info(f"{'would upload' if dry_run else 'uploading'} {len(new)} run(s) "
                 f"({size / 1e9:.2f} GB) to {repo_id}; {len(known)} already there")
    if dry_run or not new:
        return new

    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api = HfApi()
    # Runs first and the manifest last. An interrupted upload then leaves files the
    # manifest does not mention yet, which the next run of this re-uploads; the other
    # order would publish a manifest naming files that never arrived.
    api.upload_folder(folder_path=str(directory), repo_id=repo_id, repo_type="dataset",
                      allow_patterns=files,
                      commit_message=f"add {len(new)} plate run(s)")
    merged = hub + new
    api.upload_file(path_or_fileobj="".join(json.dumps(e) + "\n" for e in merged).encode(),
                    path_in_repo=MANIFEST_NAME, repo_id=repo_id, repo_type="dataset",
                    commit_message=f"manifest: {len(merged)} run(s)")
    api.upload_file(path_or_fileobj=_readme(repo_id, merged).encode(),
                    path_in_repo="README.md", repo_id=repo_id, repo_type="dataset",
                    commit_message="update dataset card")
    logging.info(f"uploaded to https://huggingface.co/datasets/{repo_id}")
    return new


def _readme(repo_id, entries):
    """A dataset card saying what is in there and what to do with it."""
    runs = collections.Counter(e["kind"] for e in entries)
    plates = collections.Counter()
    for entry in entries:
        plates[entry["kind"]] += plate_count(entry)
    rows = "\n".join(f"| {kind} | {runs[kind]} | {plates[kind]} |" for kind in sorted(runs))
    return f"""---
tags:
- robotics
- stringman
---

# {repo_id.split('/')[-1]}

Raw plate captures for the stringman visual servoing dataset: what the gripper camera saw,
stored exactly as it came off the robot. Nothing here is matted, keyed or composited.

| kind | runs | plates |
| --- | --- | --- |
{rows}

Each run is a parquet file of encoded frames, or a `.ts` video plus a `.jsonl` telemetry
track, described by one line of `{MANIFEST_NAME}`. Read it with
`nf_robot.ml.visual_servoing.plates`, and pull it into a local collection with:

    python -m nf_robot.ml.visual_servoing.merge_plates --into plates --from {repo_id}

Mattes and synthetic frames are derived from these by `finger_matte.py`, `object_matte.py`
and `synth_frames.py`, whose thresholds are still being tuned - so they are rebuilt from
this, never stored alongside it.
"""


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--into", required=True, help="The collection to work on")
    parser.add_argument("--from", dest="sources", nargs="*", default=[],
                        help="Collections to merge in: directories or hub dataset repo ids")
    parser.add_argument("--move", action="store_true",
                        help="Move the run files instead of copying them (local sources only)")
    parser.add_argument("--upload", metavar="REPO_ID", default=None,
                        help="Add this collection's raw captures to a hub dataset")
    parser.add_argument("--prune", action="store_true",
                        help="Drop manifest entries whose files have been deleted")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="Summarize the collection and exit")
    args = parser.parse_args()

    into = Path(args.into)
    if args.list or not (args.sources or args.upload or args.prune):
        summarize(into)
        return

    if args.prune:
        prune(into, args.dry_run)
    if args.sources:
        merge(args.sources, into, args.move, args.dry_run)
    if args.upload:
        upload(into, args.upload, args.dry_run)
    summarize(into)


if __name__ == "__main__":
    main()
