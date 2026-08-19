#!/usr/bin/env python

"""Put the `spin` state field back into a teleop dataset that was recorded without it.

`spin` is the gripper camera's heading in the room, and mine_teleop cannot label anything
without it: the whole method is projecting one room point into the camera, and a camera of
unknown heading projects nothing. Some older recordings never logged it.

It is recoverable because the recorder logged something else derived from it. Every named
target carries a bearing, built as `room_angle - spin` where room_angle points from the
gripper to that target (stringman_lerobot._build_state). In these recordings no target was
ever detected, so every one of them sits at the room origin - which is visible in the data
as `distance` being exactly the gripper's horizontal distance from the origin. The room
angle is then known from the gripper position alone, and

    spin = atan2(-x, -y) - bearing

is an identity rather than an estimate. It degenerates only near the origin, where the
direction back to it stops being well defined.

That gap is closed by the wrist. `spin` moves with `wrist_angle` and nothing else -
get_spin() is `radians(wrist) + (frame_room_spin - pi)` - so the difference between the two
is a calibration constant, per recording session. Measuring that constant on the frames
where the bearing is well conditioned lets every frame in the episode be filled exactly,
including the ones near the origin. The constant is also the check: it has to come out flat
across an episode, and an episode where it does not is left without spin rather than
guessed at.

The source dataset is never modified. A new root is written beside it with the state
column widened by one, its metadata patched to match, and its videos symlinked - they are
the bulk of the bytes and none of them change.

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

# (metres) how far the gripper has to be from the room origin for the direction back to it
# to be worth reading. Close in, a centimetre of position noise swings the bearing wildly.
MIN_HORIZONTAL_M = 0.30
# (metres) how exactly a target's distance must match the gripper's own distance from the
# origin before that column is believed to be pointing at the origin rather than at
# something real. These are the same float twice, so the tolerance is only for round-trips.
ORIGIN_TOLERANCE_M = 1e-3
# Frames an episode needs before its calibration constant is trusted.
MIN_FRAMES_PER_EPISODE = 30
# (degrees) how far the measured constant may wander inside one episode. spin is an exact
# function of the wrist, so anything past a rounding error means the assumption is wrong
# for that episode and it should go unlabelled.
MAX_SPREAD_DEG = 1.0

# The named targets the recorder wrote a bearing and distance for. Any of them will do; all
# of them together outvote a column that happened to hold a real detection.
TARGET_NAMES = ("hamper", "toybox", "trashcan", "gamepad", "parking_location")

STATE_FEATURE = "observation.state"
SPIN_FIELD = "spin"


def circular_mean(angles):
    """Mean of angles that wrap, which the arithmetic mean of radians is not."""
    return float(np.angle(np.mean(np.exp(1j * np.asarray(angles)))))


def wrap_pi(angles):
    return (np.asarray(angles) + np.pi) % (2 * np.pi) - np.pi


def spin_from_bearings(state, index):
    """Per-frame spin from every target column that points at the room origin.

    Returns (spin, usable), both (targets, frames): spin is only meaningful where usable.
    """
    x = state[:, index["gripper_pos_x"]]
    y = state[:, index["gripper_pos_y"]]
    horizontal = np.hypot(x, y)
    # bearing = room_angle - spin, and with the target at the origin the room angle is the
    # direction from the gripper back to it
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
    """(constant, spread_deg, frames) relating spin to the wrist for one episode.

    The constant is `spin - radians(wrist_angle)`, which get_spin makes a property of the
    calibration rather than of the moment. spread_deg is how much it moved across the
    episode and is the reason to believe or disbelieve the result.
    """
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
    # Every frame, not just the well conditioned ones: the wrist is exact everywhere and
    # the constant is what was missing.
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
    """Spin for every frame of a dataset, as (values, mask, per-episode diagnostics).

    Frames of an episode that could not be measured get 0.0 and a False mask; they are
    written as a value like any other, because a state column cannot have holes, and the
    report says how many there are.
    """
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
    # The constant is a calibration, so a dataset recorded across sessions shows a handful
    # of distinct values. Seeing them is the sanity check that this is a real quantity.
    constants = np.array([d["constant_deg"] for d in good.values()])
    values, counts = np.unique(np.round(constants, 1), return_counts=True)
    order = np.argsort(-counts)
    logging.info("   calibration constants (spin - wrist), by episode count:")
    for i in order[:6]:
        logging.info(f"      {values[i]:+8.1f} deg   {counts[i]} episode(s)")
    if len(values) > 6:
        logging.info(f"      ... and {len(values) - 6} more")


def write_dataset(source: Path, into: Path, spin, names, info, copy_videos=False):
    """A copy of the dataset with spin appended to its state column.

    Appended rather than inserted so that every existing field keeps its index, which
    anything reading the old dataset by position keeps working against.
    """
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
    """One statistic of the recovered column, for the metadata's per-dimension arrays.

    Whole-dataset values for a per-episode table are wrong in detail, and harmless: nothing
    trains on the metadata, and a loader that normalizes by it gets a sane scale either way.
    """
    if key == "min":
        return float(np.min(spin))
    if key == "max":
        return float(np.max(spin))
    if key == "std":
        return float(np.std(spin))
    if key.startswith("q"):
        return float(np.percentile(spin, int(key[1:])))
    return float(np.mean(spin))


def resolve_root(repo_id, root=None, videos=False):
    """Where the dataset is on disk.

    Only the metadata and the state columns are needed to recover spin, and they are a
    rounding error next to the videos - so the videos come down only when the result is
    meant to be mined, which is what needs the frames.
    """
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


if __name__ == "__main__":
    main()
