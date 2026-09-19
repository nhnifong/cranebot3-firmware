#!/usr/bin/env python

"""Recover a teleop dataset's missing `spin` from target bearings and the wrist angle.

Usage:
    python -m nf_robot.ml.visual_servoing.recover_spin --repo_id naavox/simple_grasp \\
        --into datasets/simple_grasp_spin
    python -m nf_robot.ml.visual_servoing.recover_spin --repo_id naavox/simple_grasp \\
        --report_only
"""

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np

# (metres) minimum gripper distance from the origin for the bearing back to it to be
# reliable.
MIN_HORIZONTAL_M = 0.30
# (metres) tolerance for a target's distance to equal the gripper's distance from the
# origin.
ORIGIN_TOLERANCE_M = 1e-3
# Frames an episode needs before its calibration constant is trusted.
MIN_FRAMES_PER_EPISODE = 30
# (degrees) how far the calibration constant may wander within an episode before it is left
# unlabelled.
MAX_SPREAD_DEG = 1.0

# Named targets whose bearing and distance the recorder wrote.
TARGET_NAMES = ("hamper", "toybox", "trashcan", "gamepad", "parking_location")

STATE_FEATURE = "observation.state"
SPIN_FIELD = "spin"


def circular_mean(angles):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(angles)))))


def wrap_pi(angles):
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def spin_from_bearings(state, index):
    """Per-frame spin from every target column pointing at the room origin, as (spin, usable)."""
    x = state[:, index["gripper_pos_x"]]
    y = state[:, index["gripper_pos_y"]]
    horizontal = np.hypot(x, y)
    # bearing = room_angle - spin, with the room angle pointing from the gripper back to the
    # origin.
    room_angle = np.arctan2(-x, -y)

    spins, usable = [], []
    for name in TARGET_NAMES:
        bearing = state[:, index[f"{name}_bearing"]]
        distance = state[:, index[f"{name}_distance"]]
        spins.append(room_angle - bearing)
        usable.append((np.abs(distance - horizontal) < ORIGIN_TOLERANCE_M)
                      & (horizontal > MIN_HORIZONTAL_M))
    return np.stack(spins), np.stack(usable)


def episode_constant(state, index):
    """(constant, spread_deg, frames) relating spin to the wrist for one episode."""
    spins, usable = spin_from_bearings(state, index)
    wrist = np.radians(state[:, index["wrist_angle"]])

    offsets = []
    for spin, ok in zip(spins, usable):
        if ok.any():
            offsets.append(wrap_pi(spin[ok] - wrist[ok]))
    if not offsets:
        return None, None, 0

    pooled = np.concatenate(offsets)
    if len(pooled) < MIN_FRAMES_PER_EPISODE:
        return None, None, len(pooled)

    constant = circular_mean(pooled)
    spread = float(np.degrees(np.percentile(np.abs(wrap_pi(pooled - constant)), 95)))
    return constant, spread, len(pooled)


def episode_spin(state, index):
    """(spin per frame, diagnostics) for one episode, or (None, diagnostics) if unusable."""
    constant, spread, frames = episode_constant(state, index)
    if constant is None:
        return None, {"frames": frames, "reason": "too few frames with a usable bearing"}
    if spread > MAX_SPREAD_DEG:
        return None, {"frames": frames, "spread_deg": spread,
                      "reason": f"spin is not a fixed offset from the wrist here "
                                f"({spread:.1f} deg of drift)"}
    # Every frame, since the wrist is exact everywhere.
    spin = np.radians(state[:, index["wrist_angle"]]) + constant
    return spin, {"frames": frames, "spread_deg": spread,
                  "constant_deg": float(np.degrees(constant))}


def read_state(root: Path):
    """(episode_index, frame order, state matrix, index by name) for a whole dataset."""
    import pyarrow.parquet as pq

    info = json.loads((root / "meta" / "info.json").read_text())
    names = info["features"][STATE_FEATURE]["names"]
    if SPIN_FIELD in names:
        raise ValueError(f"{root} already has a {SPIN_FIELD} field; nothing to recover")
    missing = [n for n in ("gripper_pos_x", "gripper_pos_y", "wrist_angle")
               if n not in names]
    missing += [f"{TARGET_NAMES[0]}_bearing"] if f"{TARGET_NAMES[0]}_bearing" not in names else []
    if missing:
        raise ValueError(f"{root} cannot be recovered: state has no {missing}")

    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquets under {root}/data")

    episodes, states = [], []
    for path in files:
        table = pq.read_table(path, columns=["episode_index", STATE_FEATURE])
        episodes.append(np.array(table.column("episode_index").to_pylist()))
        states.append(np.array(table.column(STATE_FEATURE).to_pylist(), dtype=np.float64))
    return (np.concatenate(episodes), np.concatenate(states),
            {n: i for i, n in enumerate(names)}, names, info)


def recover(root: Path):
    """Spin for every frame of a dataset as (values, mask, diagnostics), 0.0 where unmeasured."""
    episode_index, state, index, names, info = read_state(root)
    spin = np.zeros(len(state))
    known = np.zeros(len(state), bool)
    report = {}

    for episode in sorted(set(episode_index.tolist())):
        rows = episode_index == episode
        values, diagnostics = episode_spin(state[rows], index)
        report[int(episode)] = diagnostics
        if values is not None:
            spin[rows] = values
            known[rows] = True
    return spin, known, report, names, info


def summarize(report, known):
    """Log what the recovery found, including the calibration sessions it implies."""
    good = {e: d for e, d in report.items() if "constant_deg" in d}
    logging.info(f"spin recovered for {known.sum()} of {len(known)} frames "
                 f"({known.mean() * 100:.1f}%), {len(good)} of {len(report)} episodes")
    for episode, diagnostics in sorted(report.items()):
        if "reason" in diagnostics:
            logging.info(f"   episode {episode}: {diagnostics['reason']}")

    if not good:
        return
    spreads = np.array([d["spread_deg"] for d in good.values()])
    logging.info(f"   drift of the constant inside an episode: median "
                 f"{np.median(spreads):.3f} deg, worst {spreads.max():.3f} deg")
    # Distinct constants correspond to calibration sessions.
    constants = np.array([d["constant_deg"] for d in good.values()])
    values, counts = np.unique(np.round(constants, 1), return_counts=True)
    order = np.argsort(-counts)
    logging.info("   calibration constants (spin - wrist), by episode count:")
    for i in order[:6]:
        logging.info(f"      {values[i]:+8.1f} deg   {counts[i]} episode(s)")
    if len(values) > 6:
        logging.info(f"      ... and {len(values) - 6} more")


def write_dataset(source: Path, into: Path, spin, names, info, copy_videos=False):
    """A copy of the dataset with spin appended to its state column."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    into.mkdir(parents=True, exist_ok=True)
    (into / "data").mkdir(exist_ok=True)

    written = 0
    for path in sorted(source.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(path)
        rows = table.num_rows
        widened = [list(row) + [float(value)] for row, value in
                   zip(table.column(STATE_FEATURE).to_pylist(), spin[written:written + rows])]
        column = pa.array(widened, type=pa.list_(pa.float32(), len(names) + 1))
        table = table.set_column(table.schema.get_field_index(STATE_FEATURE),
                                 STATE_FEATURE, column)
        out = into / path.relative_to(source)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)
        written += rows
    logging.info(f"rewrote {written} rows of {STATE_FEATURE} into {into}/data")

    _write_meta(source, into, spin, names, info)

    videos = source / "videos"
    target = into / "videos"
    if videos.exists() and not target.exists():
        if copy_videos:
            logging.info("copying videos (this is the bulk of the dataset)")
            shutil.copytree(videos, target)
        else:
            target.symlink_to(videos.resolve(), target_is_directory=True)
            logging.info(f"symlinked videos -> {videos.resolve()}")
    return into


def _write_meta(source: Path, into: Path, spin, names, info):
    """Copy the metadata across with every record of the state's width brought up to date."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    (into / "meta").mkdir(parents=True, exist_ok=True)
    for item in (source / "meta").iterdir():
        target = into / "meta" / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    info = json.loads(json.dumps(info))
    info["features"][STATE_FEATURE]["names"] = list(names) + [SPIN_FIELD]
    info["features"][STATE_FEATURE]["shape"] = [len(names) + 1]
    (into / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")

    # Whole-dataset stats: one more entry per per-dimension array.
    stats_path = into / "meta" / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        entry = stats.get(STATE_FEATURE, {})
        for key, value in list(entry.items()):
            if isinstance(value, list) and len(value) == len(names):
                entry[key] = value + [_stat(key, spin)]
        stats_path.write_text(json.dumps(stats) + "\n")

    # Per-episode stats live in the episodes table, in fixed width list columns.
    for path in sorted((into / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(path)
        for name in [n for n in table.schema.names if n.startswith(f"stats/{STATE_FEATURE}/")]:
            values = table.column(name).to_pylist()
            if not values or values[0] is None or len(values[0]) != len(names):
                continue
            key = name.rsplit("/", 1)[1]
            widened = [row + [_stat(key, spin)] for row in values]
            table = table.set_column(table.schema.get_field_index(name), name,
                                     pa.array(widened, type=pa.list_(pa.float64())))
        pq.write_table(table, path)


def _stat(key, spin):
    """One statistic of the recovered column, for the metadata's per-dimension arrays."""
    if key == "min":
        return float(np.min(spin))
    if key == "max":
        return float(np.max(spin))
    if key == "std":
        return float(np.std(spin))
    if key.startswith("q"):
        return float(np.percentile(spin, int(key[1:])))
    return float(np.mean(spin))


def upload_dataset(root: Path, repo_id: str, what="the recovered spin field"):
    """Publish a LeRobot dataset, following the video symlink and moving the version tag."""
    from huggingface_hub import HfApi, create_repo

    root = Path(root)
    version = json.loads((root / "meta" / "info.json").read_text())["codebase_version"]
    api = HfApi()
    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    api.upload_folder(folder_path=str(root), repo_id=repo_id, repo_type="dataset",
                      ignore_patterns=["videos*", ".DS_Store"],
                      commit_message=f"state and metadata with {what}")
    videos = root / "videos"
    if videos.exists():
        api.upload_folder(folder_path=str(videos.resolve()), path_in_repo="videos",
                          repo_id=repo_id, repo_type="dataset",
                          commit_message="videos")
    else:
        logging.warning(f"no videos under {root}; the upload will not be mineable. "
                        f"Recover again with --videos to fetch them.")

    # Move the tag, since LeRobotDataset reads a repo at the tag rather than main.
    tags = [t.name for t in api.list_repo_refs(repo_id, repo_type="dataset").tags]
    if version in tags:
        api.delete_tag(repo_id, tag=version, repo_type="dataset")
    api.create_tag(repo_id, tag=version, repo_type="dataset",
                   tag_message="lerobot codebase version, matching meta/info.json")
    logging.info(f"uploaded to https://huggingface.co/datasets/{repo_id} (tag {version})")


def resolve_root(repo_id, root=None, videos=False):
    """Where the dataset is on disk, downloading videos only if they will be mined."""
    if root:
        return Path(root)
    from huggingface_hub import snapshot_download

    patterns = None if videos else ["meta/*", "data/*"]
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset",
                                  allow_patterns=patterns))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo_id", required=True, help="Source teleop dataset")
    parser.add_argument("--root", default=None,
                        help="Its root on disk (defaults to the HF cache; the videos are "
                             "not needed to recover spin, only to mine the result)")
    parser.add_argument("--into", default=None,
                        help="Where the recovered copy is written. Required unless --report_only")
    parser.add_argument("--report_only", action="store_true",
                        help="Say what would be recovered and stop")
    parser.add_argument("--videos", action="store_true",
                        help="Also fetch the videos, so the result can be mined and not "
                             "just inspected. They are the bulk of the download.")
    parser.add_argument("--copy_videos", action="store_true",
                        help="Copy the videos instead of symlinking them")
    parser.add_argument("--upload", metavar="REPO_ID", default=None,
                        help="Publish the recovered dataset to this hub repo")
    args = parser.parse_args()

    if not args.into and not args.report_only:
        parser.error("--into is required unless --report_only")

    root = resolve_root(args.repo_id, args.root, args.videos)
    logging.info(f"reading {root}")
    spin, known, report, names, info = recover(root)
    summarize(report, known)

    if args.report_only:
        return
    if not known.any():
        logging.error("nothing recovered; not writing a dataset")
        return
    write_dataset(root, Path(args.into), spin, names, info, args.copy_videos)
    logging.info(f"done. Check it with:\n"
                 f"    python -m nf_robot.ml.visual_servoing.mine_teleop --check "
                 f"--repo_id {args.into}")
    if args.upload:
        upload_dataset(Path(args.into), args.upload)


if __name__ == "__main__":
    main()
