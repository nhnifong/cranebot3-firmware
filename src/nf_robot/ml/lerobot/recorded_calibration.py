"""Anchor calibration recorded alongside a dataset: the anchor camera poses and mount tilts."""

import json
import logging
import os
import pathlib

import numpy as np


def load_anchor_poses(config):
    """Anchor camera poses as [(rotvec, position), ...] from a robot config path or parsed
    config."""
    if isinstance(config, (str, os.PathLike)):
        config = json.loads(pathlib.Path(config).read_text())
    poses = []
    for anchor in config["anchors"]:
        r = anchor["pose"]["rotation"]
        p = anchor["pose"]["position"]
        poses.append((np.array([r["x"], r["y"], r["z"]], dtype=float),
                      np.array([p["x"], p["y"], p["z"]], dtype=float)))
    return poses


# Anchor poses recorded per frame, so a dataset can be converted without the robot's config.
ANCHOR_POSES_KEY = "anchor_poses"
N_RECORDED_ANCHORS = 2
ANCHOR_POSE_NAMES = [
    f"anchor_{i}_{c}" for i in range(N_RECORDED_ANCHORS) for c in ("rx", "ry", "rz", "x", "y", "z")
]


def anchor_poses_feature():
    """The dataset feature holding the anchor camera poses, for LeRobotDataset.create."""
    return {
        ANCHOR_POSES_KEY: {
            "dtype": "float32",
            "shape": (len(ANCHOR_POSE_NAMES),),
            "names": list(ANCHOR_POSE_NAMES),
        }
    }


def pack_anchor_poses(poses):
    """[(rotvec, position), ...] -> the flat per-frame vector, with unknown poses as zeros."""
    flat = np.zeros(len(ANCHOR_POSE_NAMES), dtype=np.float32)
    for i, (rotvec, position) in enumerate(list(poses)[:N_RECORDED_ANCHORS]):
        flat[i * 6:i * 6 + 3] = np.asarray(rotvec, dtype=np.float32)
        flat[i * 6 + 3:i * 6 + 6] = np.asarray(position, dtype=np.float32)
    return flat


def unpack_anchor_poses(flat):
    """Flat vector -> [(rotvec, position), ...], or None if it holds no calibration."""
    flat = np.asarray(flat, dtype=np.float64)
    if not np.any(flat):
        return None
    poses = []
    for i in range(len(flat) // 6):
        block = flat[i * 6:i * 6 + 6]
        if not np.any(block):
            continue
        poses.append((block[:3], block[3:]))
    return poses or None


# Anchor camera mount tilts, kept out of ANCHOR_POSES_KEY so its width stays mergeable with
# old datasets.
ANCHOR_CAM_TILT_KEY = "anchor_cam_tilt"
ANCHOR_CAM_TILT_NAMES = [f"anchor_{i}_cam_tilt" for i in range(N_RECORDED_ANCHORS)]


def anchor_cam_tilt_feature():
    """The dataset feature holding the anchor camera tilts, for LeRobotDataset.create."""
    return {
        ANCHOR_CAM_TILT_KEY: {
            "dtype": "float32",
            "shape": (len(ANCHOR_CAM_TILT_NAMES),),
            "names": list(ANCHOR_CAM_TILT_NAMES),
        }
    }


def recorded_calibration_features():
    return {**anchor_poses_feature(), **anchor_cam_tilt_feature()}


def pack_anchor_cam_tilt(tilts):
    """[degrees, ...] -> the flat per-frame vector, with 0 meaning unknown."""
    flat = np.zeros(len(ANCHOR_CAM_TILT_NAMES), dtype=np.float32)
    for i, tilt in enumerate(list(tilts)[:N_RECORDED_ANCHORS]):
        flat[i] = float(tilt)
    return flat


def unpack_anchor_cam_tilt(flat):
    """Flat vector -> [degrees, ...], or None if it holds no tilts."""
    flat = np.asarray(flat, dtype=np.float64)
    if not np.all(flat):
        return None
    return [float(t) for t in flat]


def add_anchor_poses_feature(root, anchor_poses):
    """Add the anchor-pose feature to an older dataset, filled with the poses it was
    converted with, so it merges with new ones."""
    if has_feature(root, ANCHOR_POSES_KEY):
        return False
    if len(anchor_poses) < N_RECORDED_ANCHORS:
        raise ValueError(
            f"cannot add the {ANCHOR_POSES_KEY} feature without {N_RECORDED_ANCHORS} anchor poses"
        )
    return add_recorded_calibration_feature(
        root, ANCHOR_POSES_KEY, anchor_poses_feature()[ANCHOR_POSES_KEY],
        pack_anchor_poses(anchor_poses))


def add_anchor_cam_tilt_feature(root, cam_tilts):
    """Add the camera-tilt feature to an older dataset, skipping episodes whose tilt is unknown."""
    if has_feature(root, ANCHOR_CAM_TILT_KEY):
        return False
    by_episode = isinstance(cam_tilts, dict)
    packed = {e: pack_anchor_cam_tilt(t)
              for e, t in (cam_tilts if by_episode else {None: cam_tilts}).items()}
    if not packed or not all(np.all(v) for v in packed.values()):
        logging.info(f"not adding {ANCHOR_CAM_TILT_KEY}: no tilt is known for every anchor")
        return False
    return add_recorded_calibration_feature(
        root, ANCHOR_CAM_TILT_KEY, anchor_cam_tilt_feature()[ANCHOR_CAM_TILT_KEY],
        packed if by_episode else packed[None])


def has_feature(root, key):
    """Whether a dataset on disk already declares one feature."""
    import pathlib

    info = json.loads((pathlib.Path(root) / "meta" / "info.json").read_text())
    return key in info["features"]


def add_recorded_calibration_feature(root, key, feature, values):
    """Add a per-episode-constant calibration feature, updating parquets, info.json and all
    stats together."""
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if key in info["features"]:
        return False

    width = len(feature["names"])
    def vector_for(episode):
        flat = values[episode] if isinstance(values, dict) else values
        return np.asarray(flat, dtype=np.float64)

    field = pa.field(key, pa.list_(pa.float32(), width))
    for f in sorted(root.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(f)
        rows = [vector_for(e).tolist() for e in table.column("episode_index").to_pylist()]
        pq.write_table(table.append_column(field, pa.array(rows, type=field.type)), f)

    info["features"][key] = {
        "dtype": feature["dtype"],
        "shape": list(feature["shape"]),
        "names": list(feature["names"]),
    }
    info_path.write_text(json.dumps(info, indent=4))

    zeros = [0.0] * width
    vector_type = pa.list_(pa.float64(), width)
    count_type = pa.list_(pa.int64(), 1)
    per_episode = []
    for f in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(f)
        if f"stats/{key}/min" in table.schema.names:
            continue
        episodes = table.column("episode_index").to_pylist()
        lengths = table.column("length").to_pylist()
        rows = [vector_for(e).tolist() for e in episodes]
        per_episode += list(zip(rows, lengths))
        for stat in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):
            table = table.append_column(pa.field(f"stats/{key}/{stat}", vector_type),
                                        pa.array(rows, type=vector_type))
        table = table.append_column(pa.field(f"stats/{key}/std", vector_type),
                                    pa.array([zeros] * table.num_rows, type=vector_type))
        table = table.append_column(pa.field(f"stats/{key}/count", count_type),
                                    pa.array([[int(n)] for n in lengths], type=count_type))
        pq.write_table(table, f)

    # Aggregate stats across episodes, weighted by episode length.
    rows = np.asarray([r for r, _ in per_episode], dtype=np.float64)
    weights = np.asarray([n for _, n in per_episode], dtype=np.float64)
    mean = np.average(rows, axis=0, weights=weights)
    std = np.sqrt(np.average((rows - mean) ** 2, axis=0, weights=weights))
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[key] = {
        "min": rows.min(axis=0).tolist(), "max": rows.max(axis=0).tolist(),
        "mean": mean.tolist(), "std": std.tolist(),
        "count": [int(info["total_frames"])],
        **{q: mean.tolist() for q in ("q01", "q10", "q50", "q90", "q99")},
    }
    stats_path.write_text(json.dumps(stats, indent=4))

    logging.info(f"Added the {key} feature, filled with the calibration used for conversion")
    return True
