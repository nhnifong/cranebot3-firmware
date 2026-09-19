#!/usr/bin/env python

"""The camera_goal action space: the gripper's goal position in each camera's frame, derived
from contact-labelled recordings and fused back into robot control."""

import collections
import json
import logging
import os
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

import nf_robot.common.definitions as model_constants
from nf_robot.common.pose_functions import compose_poses, invert_pose

ACTION_SPACE_NAME = "camera_goal"

# Order matters: it is the action vector layout.
ACTION_NAMES = [
    "goal_gripper_cam_x", "goal_gripper_cam_y", "goal_gripper_cam_z",
    "goal_anchor_0_x", "goal_anchor_0_y", "goal_anchor_0_z",
    "goal_anchor_1_x", "goal_anchor_1_y", "goal_anchor_1_z",
    "wrist_offset",
    "finger_speed",
    "episode_end",
]

# Which action components hold each camera's goal, and which camera they belong to.
GOAL_SLOTS = {
    "gripper_camera": ("goal_gripper_cam_x", "goal_gripper_cam_y", "goal_gripper_cam_z"),
    "anchor_camera_0": ("goal_anchor_0_x", "goal_anchor_0_y", "goal_anchor_0_z"),
    "anchor_camera_1": ("goal_anchor_1_x", "goal_anchor_1_y", "goal_anchor_1_z"),
}

# Fusion weights: the gripper camera is best at grasping range, the anchors from across the
# room.
FUSION_WEIGHTS = {"gripper_camera": 1.0, "anchor_camera_0": 1.0, "anchor_camera_1": 1.0}

# Goal approach speed in m/s, a controller constant so eval runs at demonstration pace.
APPROACH_SPEED = 0.25
# Wrist offsets are turned into a rate the same way: proportional, capped. deg/s.
WRIST_GAIN = 2.0
WRIST_MAX_SPEED = 120.0
WRIST_DEADBAND_RAD = 0.05
# A wrist run is movement faster than MIN_WRIST_SPEED_DPS, merged across gaps under
# WRIST_GAP_S and ignored under WRIST_MIN_TRAVEL_DEG.
MIN_WRIST_SPEED_DPS = 5.0
WRIST_GAP_S = 0.25
WRIST_MIN_TRAVEL_DEG = 5.0
# Stop commanding motion once this close; prevents dithering on top of the goal.
GOAL_DEADBAND_M = 0.03


def gripper_camera_pose(gripper_pos, gripper_rot_6d):
    """Room-frame pose of the gripper camera as (rotvec, position), from the 6D rotation in
    observation.state."""
    a = np.asarray(gripper_rot_6d[:3], dtype=float)
    b = np.asarray(gripper_rot_6d[3:6], dtype=float)
    c1 = a / (np.linalg.norm(a) + 1e-9)
    c2 = b - np.dot(c1, b) * c1
    c2 = c2 / (np.linalg.norm(c2) + 1e-9)
    c3 = np.cross(c1, c2)
    rot = Rotation.from_matrix(np.column_stack([c1, c2, c3]))
    gripper_pose = (rot.as_rotvec(), np.asarray(gripper_pos, dtype=float))
    return compose_poses([gripper_pose, model_constants.gripper_camera])


def goal_in_camera_frame(goal_room, camera_pose):
    inv = invert_pose(camera_pose)
    return compose_poses([inv, (np.zeros(3), np.asarray(goal_room, dtype=float))])[1]


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


# --------------------------------------------------------------------------
# Calibration recorded alongside the data
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Derivation: recorded action space -> camera_goal
# --------------------------------------------------------------------------

def convert_actions(states, actions, state_names, action_names, anchor_poses,
                    episode_index=None, timestamps=None, pressure_threshold=0.1, blend_seconds=0.5,
                    stored_anchor_poses=None):
    """Convert one dataset's recorded actions (n_frames, dim) into camera_goal actions."""
    required_state = ["gripper_pos_x", "gripper_pos_y", "gripper_pos_z"] + [f"gripper_rot_{i}" for i in range(6)]
    missing = [n for n in required_state if n not in state_names]
    if missing:
        raise ValueError(
            f"camera_goal needs {missing} in observation.state; derive it before trimming state features"
        )
    if "wrist_angle" not in state_names:
        raise ValueError("camera_goal needs 'wrist_angle' in observation.state")
    required_action = ["contact_vec_x", "contact_vec_y", "contact_vec_z", "finger_speed", "episode_end"]
    missing = [n for n in required_action if n not in action_names]
    if missing:
        raise ValueError(
            f"camera_goal needs {missing} in the recorded action space; run label_contact_actions first"
        )
    if stored_anchor_poses is None and len(anchor_poses) < 2:
        raise ValueError(f"camera_goal needs 2 anchor camera poses, got {len(anchor_poses)}")

    s_idx = {n: i for i, n in enumerate(state_names)}
    a_idx = {n: i for i, n in enumerate(action_names)}

    pos = states[:, [s_idx["gripper_pos_x"], s_idx["gripper_pos_y"], s_idx["gripper_pos_z"]]]
    rot6 = states[:, [s_idx[f"gripper_rot_{i}"] for i in range(6)]]
    contact = actions[:, [a_idx["contact_vec_x"], a_idx["contact_vec_y"], a_idx["contact_vec_z"]]]
    goal_room = pos + contact

    out = np.zeros((len(actions), len(ACTION_NAMES)), dtype=np.float32)
    o_idx = {n: i for i, n in enumerate(ACTION_NAMES)}

    for t in range(len(actions)):
        cam_pose = gripper_camera_pose(pos[t], rot6[t])
        g = goal_in_camera_frame(goal_room[t], cam_pose)
        out[t, [o_idx[n] for n in GOAL_SLOTS["gripper_camera"]]] = g

    # Poses recorded with the data win over the ones passed in, since they are what was
    # actually running.
    if stored_anchor_poses is not None:
        blocks = _group_rows_by_value(stored_anchor_poses)
    else:
        blocks = [(None, np.arange(len(actions)))]

    for value, rows in blocks:
        poses = unpack_anchor_poses(value) if value is not None else None
        if poses is None:
            poses = anchor_poses
        if len(poses) < 2:
            raise ValueError(
                "frames record no anchor poses and no anchor_config was supplied for them"
            )
        for anchor_num in (0, 1):
            key = f"anchor_camera_{anchor_num}"
            inv = invert_pose(poses[anchor_num])
            rot = Rotation.from_rotvec(inv[0])
            out[np.ix_(rows, [o_idx[n] for n in GOAL_SLOTS[key]])] = (
                Rotation.from_rotvec(inv[0]).apply(goal_room[rows]) + inv[1]
            )

    for name in ("finger_speed", "episode_end"):
        out[:, o_idx[name]] = actions[:, a_idx[name]]

    for ep in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == ep)
        rows = rows[np.argsort(timestamps[rows])]
        out[rows, o_idx["wrist_offset"]] = _wrist_offsets(
            states[rows, s_idx["wrist_angle"]], timestamps[rows]
        )
    return out



def _group_rows_by_value(rows):
    """[(value, row indices), ...] for each distinct row of a 2-D array."""
    values, inverse = np.unique(np.asarray(rows), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    return [(values[i], np.flatnonzero(inverse == i)) for i in range(len(values))]


def derive_dataset_actions(root, anchor_poses=(), pressure_threshold=0.1, blend_seconds=0.5):
    """Rewrite a dataset in place so its action feature, stats included, is the camera_goal
    space."""
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    state_names = list(info["features"]["observation.state"]["names"])
    action_names = list(info["features"]["action"]["names"])

    all_new, episode_columns = [], []
    used_stored = 0
    for f in sorted(root.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(f)
        states = np.array(table.column("observation.state").to_pylist(), dtype=np.float64)
        actions = np.array(table.column("action").to_pylist(), dtype=np.float64)
        stored = None
        if ANCHOR_POSES_KEY in table.schema.names:
            stored = np.array(table.column(ANCHOR_POSES_KEY).to_pylist(), dtype=np.float64)
            used_stored += int(np.any(stored, axis=1).sum())

        new = convert_actions(
            states, actions, state_names, action_names, anchor_poses,
            episode_index=np.array(table.column("episode_index").to_pylist()),
            timestamps=np.array(table.column("timestamp").to_pylist(), dtype=np.float64),
            pressure_threshold=pressure_threshold, blend_seconds=blend_seconds,
            stored_anchor_poses=stored,
        )
        all_new.append(new)
        episode_columns.append(np.array(table.column("episode_index").to_pylist()))

        field = pa.field("action", pa.list_(pa.float32(), len(ACTION_NAMES)))
        col = table.schema.get_field_index("action")
        table = table.set_column(col, field, pa.array(new.tolist(), type=field.type))
        pq.write_table(table, f)

    if used_stored:
        logging.info(f"Used anchor poses recorded with the data for {used_stored} frame(s)")
    else:
        logging.info("Dataset records no anchor poses; using the supplied calibration")

    stacked = np.concatenate(all_new)
    info["features"]["action"]["names"] = list(ACTION_NAMES)
    info["features"]["action"]["shape"] = [len(ACTION_NAMES)]
    info_path.write_text(json.dumps(info, indent=4))

    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = {
        "min": stacked.min(axis=0).tolist(),
        "max": stacked.max(axis=0).tolist(),
        "mean": stacked.mean(axis=0).tolist(),
        "std": stacked.std(axis=0).tolist(),
        "count": [int(stacked.shape[0])],
        **{q: np.quantile(stacked, float(q[1:]) / 100, axis=0).tolist()
           for q in ("q01", "q10", "q50", "q90", "q99")},
    }
    stats_path.write_text(json.dumps(stats, indent=4))

    _rewrite_episode_stats(root, episode_columns, all_new)
    # so a merge of new and old recordings sees the same feature set
    if not used_stored and anchor_poses:
        add_anchor_poses_feature(root, anchor_poses)
    logging.info(f"action space is now {ACTION_SPACE_NAME}: {ACTION_NAMES}")



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


def _rewrite_episode_stats(root, episode_columns, all_new):
    """Recompute the per-episode action stats so they match the new action width."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_episode = {}
    for episodes, new in zip(episode_columns, all_new):
        for ep in np.unique(episodes):
            rows = new[episodes == ep]
            ep = int(ep)
            by_episode[ep] = rows if ep not in by_episode else np.vstack([by_episode[ep], rows])

    stat_fn = {
        "min": lambda a: a.min(axis=0), "max": lambda a: a.max(axis=0),
        "mean": lambda a: a.mean(axis=0), "std": lambda a: a.std(axis=0),
        "q01": lambda a: np.quantile(a, 0.01, axis=0), "q10": lambda a: np.quantile(a, 0.10, axis=0),
        "q50": lambda a: np.quantile(a, 0.50, axis=0), "q90": lambda a: np.quantile(a, 0.90, axis=0),
        "q99": lambda a: np.quantile(a, 0.99, axis=0),
    }
    for f in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(f)
        eps = table.column("episode_index").to_pylist()
        changed = False
        for stat, fn in stat_fn.items():
            name = f"stats/action/{stat}"
            if name not in table.schema.names:
                continue
            values = [fn(by_episode[int(e)]).tolist() if int(e) in by_episode else None for e in eps]
            idx = table.schema.get_field_index(name)
            field = table.schema.field(idx)
            new_type = (pa.list_(field.type.value_type, len(ACTION_NAMES))
                        if pa.types.is_fixed_size_list(field.type) else field.type)
            table = table.set_column(idx, field.with_type(new_type), pa.array(values, type=new_type))
            changed = True
        if changed:
            pq.write_table(table, f)


def wrist_runs(wrist_angles_deg, timestamps, min_speed_dps=None, gap_s=None, min_travel_deg=None):
    """Index ranges [(start, stop), ...] over which the wrist was turning, with brief pauses
    merged and noise dropped."""
    min_speed_dps = MIN_WRIST_SPEED_DPS if min_speed_dps is None else min_speed_dps
    gap_s = WRIST_GAP_S if gap_s is None else gap_s
    min_travel_deg = WRIST_MIN_TRAVEL_DEG if min_travel_deg is None else min_travel_deg

    angles = np.asarray(wrist_angles_deg, dtype=float)
    times = np.asarray(timestamps, dtype=float)
    if len(angles) < 2:
        return []

    dt = np.diff(times)
    dt[dt <= 0] = 1e-6
    speed = np.abs(np.diff(angles)) / dt  # deg/s, indexed by the interval's first frame
    moving = speed > min_speed_dps

    runs = []
    i = 0
    while i < len(moving):
        if not moving[i]:
            i += 1
            continue
        start = i
        while i < len(moving) and moving[i]:
            i += 1
        runs.append([start, min(i, len(angles) - 1)])  # stop frame: where it came to rest

    merged = []
    for run in runs:
        if merged and times[run[0]] - times[merged[-1][1]] < gap_s:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    return [(a, b) for a, b in merged if abs(angles[b] - angles[a]) >= min_travel_deg]


def _wrist_offsets(wrist_angles_deg, timestamps, anticipate=True):
    """Radians from each frame's wrist angle to the angle the next wrist turn ends at."""
    angles = np.asarray(wrist_angles_deg, dtype=float)
    offsets = np.zeros(len(angles), dtype=np.float64)

    previous_stop = 0
    for start, stop in wrist_runs(angles, timestamps):
        first = previous_stop if anticipate else start
        offsets[first:stop + 1] = np.radians(angles[stop] - angles[first:stop + 1])
        previous_stop = stop + 1
    return offsets


# --------------------------------------------------------------------------
# Runtime: camera_goal -> robot control
# --------------------------------------------------------------------------

def fuse_goal_to_room(action, gripper_pos, gripper_rot_6d, anchor_poses, robust=False):
    """Fuse per-camera goal predictions into (goal_room, spread), skipping cameras with
    unknown pose."""
    estimates, weights = [], []

    if all(n in action for n in GOAL_SLOTS["gripper_camera"]):
        cam_pose = gripper_camera_pose(gripper_pos, gripper_rot_6d)
        local = np.array([action[n] for n in GOAL_SLOTS["gripper_camera"]], dtype=float)
        estimates.append(compose_poses([cam_pose, (np.zeros(3), local)])[1])
        weights.append(FUSION_WEIGHTS["gripper_camera"])

    for anchor_num in (0, 1):
        key = f"anchor_camera_{anchor_num}"
        if anchor_num >= len(anchor_poses) or not all(n in action for n in GOAL_SLOTS[key]):
            continue
        local = np.array([action[n] for n in GOAL_SLOTS[key]], dtype=float)
        estimates.append(compose_poses([anchor_poses[anchor_num], (np.zeros(3), local)])[1])
        weights.append(FUSION_WEIGHTS[key])

    if not estimates:
        return None, None

    estimates = np.array(estimates)
    weights = np.array(weights, dtype=float)
    if robust and len(estimates) > 2:
        # one head that has lost the target drags a mean but not a median
        goal = np.median(estimates, axis=0)
    else:
        goal = np.average(estimates, axis=0, weights=weights)
    spread = float(np.mean(np.linalg.norm(estimates - goal, axis=1)))
    return goal, spread



# --------------------------------------------------------------------------
# Turning a stream of predictions into a destination (opt-in at eval time)
# --------------------------------------------------------------------------

# A half-second rolling median cuts per-frame goal jitter from 0.10m to 0.01m, below the
# arrival radius.
MEDIAN_WINDOW_FRAMES = 15
ARRIVAL_RADIUS_M = 0.08
# How far and how long a new destination must disagree before it replaces the latched one.
CHALLENGE_DISTANCE_M = 0.25
CHALLENGE_SECONDS = 0.5
# Escapes, so a latch onto something unreachable cannot hold forever.
STALL_SECONDS = 4.0
STALL_PROGRESS_M = 0.05
# Cross-camera spread above which a prediction may not move the latch (~0.06m in-
# distribution, ~1.8m on unseen rooms).
SPREAD_GATE_M = 0.30
APPROACH_GAIN = 1.0
MIN_APPROACH_SPEED = 0.05


class GoalStabilizer:
    """Latch one destination and replace it only on arrival, sustained disagreement, a stall
    or a phase change."""

    def __init__(self, window=MEDIAN_WINDOW_FRAMES, arrival_radius=ARRIVAL_RADIUS_M,
                 challenge_distance=CHALLENGE_DISTANCE_M, challenge_seconds=CHALLENGE_SECONDS,
                 stall_seconds=STALL_SECONDS, spread_gate=SPREAD_GATE_M):
        self.window = window
        self.arrival_radius = arrival_radius
        self.challenge_distance = challenge_distance
        self.challenge_seconds = challenge_seconds
        self.stall_seconds = stall_seconds
        self.spread_gate = spread_gate

        self._recent = collections.deque(maxlen=window)
        self.destination = None
        self.reason = "no destination yet"
        self._challenger_since = None
        self._best_distance = None
        self._best_at = None

    def reset(self):
        self._recent.clear()
        self.destination = None
        self.reason = "reset"
        self._challenger_since = None
        self._best_distance = None
        self._best_at = None

    def _adopt(self, goal, now, reason):
        self.destination = np.asarray(goal, dtype=float)
        self.reason = reason
        self._challenger_since = None
        self._best_distance = None
        self._best_at = now

    def update(self, goal_room, spread, gripper_pos, now, hold=False):
        """Feed one prediction and return the destination to drive to, or None; hold freezes it."""
        if goal_room is not None:
            self._recent.append(np.asarray(goal_room, dtype=float))
        if not self._recent:
            return None

        candidate = np.median(np.array(self._recent), axis=0)
        trusted = spread is None or spread <= self.spread_gate

        if self.destination is None:
            if trusted and len(self._recent) >= max(2, self.window // 3):
                self._adopt(candidate, now, "first destination")
            return self.destination

        if hold:
            self.reason = "held (grasping)"
            return self.destination

        distance = float(np.linalg.norm(self.destination - np.asarray(gripper_pos, dtype=float)))
        if self._best_distance is None or distance < self._best_distance - STALL_PROGRESS_M:
            self._best_distance, self._best_at = distance, now

        if distance <= self.arrival_radius:
            self._adopt(candidate, now, "arrived, taking the next destination")
        elif now - self._best_at > self.stall_seconds:
            self._adopt(candidate, now, f"stalled {self.stall_seconds:.0f}s short of it")
        elif trusted and float(np.linalg.norm(candidate - self.destination)) > self.challenge_distance:
            if self._challenger_since is None:
                self._challenger_since = now
            elif now - self._challenger_since >= self.challenge_seconds:
                self._adopt(candidate, now, "predictions moved and stayed moved")
        else:
            self._challenger_since = None

        return self.destination

    def velocity(self, gripper_pos, speed=APPROACH_SPEED, gain=APPROACH_GAIN,
                 min_speed=MIN_APPROACH_SPEED):
        """Proportional approach to the latched destination; zero once inside it."""
        if self.destination is None:
            return np.zeros(3)
        delta = self.destination - np.asarray(gripper_pos, dtype=float)
        distance = float(np.linalg.norm(delta))
        if distance <= self.arrival_radius:
            return np.zeros(3)
        return delta / distance * float(np.clip(gain * distance, min_speed, speed))


def wrist_offset_to_speed(offset_rad, gain=WRIST_GAIN, max_speed=WRIST_MAX_SPEED,
                          deadband=WRIST_DEADBAND_RAD):
    """Wrist rate in deg/s that closes a wrist offset, or 0 inside the deadband."""
    if abs(offset_rad) < deadband:
        return 0.0
    return float(np.clip(np.degrees(offset_rad) * gain, -max_speed, max_speed))


def goal_to_velocity(goal_room, gripper_pos, speed=APPROACH_SPEED, deadband=GOAL_DEADBAND_M):
    """Room-frame velocity that heads toward the goal, or zeros once inside the deadband."""
    delta = np.asarray(goal_room, dtype=float) - np.asarray(gripper_pos, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance < deadband:
        return np.zeros(3)
    return delta / distance * speed
