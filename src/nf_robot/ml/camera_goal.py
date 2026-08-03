#!/usr/bin/env python

"""The "camera_goal" action space: where to go, expressed in each camera's frame.

Everything specific to this action space lives here - the component layout, the
derivation from recorded data, and the conversion back into robot control - so it
can be reasoned about in one place.

What the policy predicts
------------------------
The gripper's goal position, once per camera, in that camera's optical frame:

    goal_gripper_cam_{x,y,z}   goal seen from the gripper camera
    goal_anchor_0_{x,y,z}      goal seen from anchor camera 0
    goal_anchor_1_{x,y,z}      goal seen from anchor camera 1
    wrist_offset               radians to turn the wrist by to reach its angle at the
                               goal; 0 means "stay where you are"
    finger_speed               unchanged from the recorded action space
    episode_end                unchanged

Why camera frames rather than the room frame: the room origin is wherever an
operator happened to place the origin card during calibration, so room
coordinates mean something different in every installation and a policy trained
on them learns that installation. A camera frame is defined by the images the
policy is looking at, so it transfers.

The wrist follows the same principle as the position: an offset to the angle the
wrist ends up at, rather than a rate. It is an offset rather than an absolute angle
because the wrist turns freely through more than one revolution, so its absolute
angle has no fixed relationship to anything the cameras see.

Why a goal position rather than a velocity: for the two anchor cameras - which do
not move - the goal is one number that stays put for the whole approach, instead
of thirty small deltas whose meaning only appears after integration. The gripper
camera moves with the robot, so its "goal position" is really a displacement; it
earns its place by being the view that resolves the target at grasping range.

Three predictions of one point is redundancy on purpose: each is transformed back
to the room and fused, and how far they disagree is a usable confidence signal.

Deriving it from recordings
---------------------------
Recordings store gamepad velocities, which say nothing about where the operator
was heading. The goal comes from the contact labelling pass instead:
`contact_vec_*` is already the room-frame vector from the gripper to the position
it eventually reaches, so `goal_room = gripper_pos + contact_vec`. That requires
label_contact_actions to have run with rotate_contact_vec=false.

Each camera's pose in the room is then needed to express that goal in its frame:
  - the gripper camera rides the gripper, so its pose comes from the recorded
    gripper position and 6D rotation composed with the fixed camera mount
  - the anchor cameras are fixed, so their poses come from the calibration file
    of the robot that recorded the dataset, which must be supplied per source
"""

import json
import logging

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

# Fusion weights. The gripper camera resolves the target at grasping range; the
# fixed anchors are the ones that still see it from across the room.
FUSION_WEIGHTS = {"gripper_camera": 1.0, "anchor_camera_0": 1.0, "anchor_camera_1": 1.0}

# Speed the goal is approached at, m/s. Speed is a controller constant here rather
# than something the policy emits, which is what keeps eval at demonstration pace.
APPROACH_SPEED = 0.25
# Wrist offsets are turned into a rate the same way: proportional, capped. deg/s.
WRIST_GAIN = 2.0
WRIST_MAX_SPEED = 120.0
WRIST_DEADBAND_RAD = 0.05
# Wrist run detection. A run is movement faster than MIN_WRIST_SPEED_DPS; runs closer
# together than WRIST_GAP_S are one turn, and a run must travel WRIST_MIN_TRAVEL_DEG
# to count as intent rather than sensor noise.
MIN_WRIST_SPEED_DPS = 5.0
WRIST_GAP_S = 0.25
WRIST_MIN_TRAVEL_DEG = 5.0
# Stop commanding motion once this close; prevents dithering on top of the goal.
GOAL_DEADBAND_M = 0.03


def gripper_camera_pose(gripper_pos, gripper_rot_6d):
    """Room-frame pose of the gripper camera, as (rotvec, position).

    gripper_rot_6d is the 6D rotation stored in observation.state: the first two
    columns of the rotation matrix, from which the third is recovered by
    orthonormalizing.
    """
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
    """Express a room-frame point in a camera's frame, given that camera's room pose."""
    inv = invert_pose(camera_pose)
    return compose_poses([inv, (np.zeros(3), np.asarray(goal_room, dtype=float))])[1]


def load_anchor_poses(path):
    """Anchor camera poses as [(rotvec, position), ...] from a robot's config file."""
    config = json.loads(open(path).read()) if isinstance(path, str) else path
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

# A dataset that carries its own anchor poses can be converted without hunting for the
# config the robot was running. Recorded per frame because that is the only granularity
# lerobot datasets have; the value is constant within an episode and survives merging.
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
    """[(rotvec, position), ...] -> the flat vector stored in each frame.

    Unknown poses record as zeros, which unpack_anchor_poses reports as absent rather
    than as an anchor at the origin.
    """
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


# --------------------------------------------------------------------------
# Derivation: recorded action space -> camera_goal
# --------------------------------------------------------------------------

def convert_actions(states, actions, state_names, action_names, anchor_poses,
                    episode_index=None, timestamps=None, pressure_threshold=0.1, blend_seconds=0.5,
                    stored_anchor_poses=None):
    """Convert one dataset's recorded actions into camera_goal actions.

    states/actions are (n_frames, dim) arrays straight out of a data parquet.
    episode_index and timestamps are the matching columns, needed for the wrist
    offset: it is measured against the wrist angle at the same target the position
    goal points at, so both are found with the same contact rule.
    Returns an (n_frames, len(ACTION_NAMES)) array.
    """
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

    # Poses recorded with the data win over the ones passed in: they are the
    # calibration that was actually running, where a config file is only right if
    # nothing has been recalibrated since. Frames sharing a calibration go together.
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
    """Rewrite a dataset in place so its action feature is the camera_goal space.

    anchor_poses is the fallback for datasets recorded before they carried their own;
    a dataset with an anchor_poses feature ignores it. The data parquets, info.json,
    meta/stats.json and the per-episode stats are updated together, so nothing that
    later re-aggregates stats sees the old action width.
    """
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
    """Give a dataset recorded before the feature existed the poses it was converted with.

    Datasets recorded now carry their own calibration, older ones do not, and lerobot's
    merge requires every source to have exactly the same features. Rather than dropping
    the feature from the new ones, the old ones gain it, filled with the calibration the
    conversion just used - so the dataset ends up documenting the poses its labels were
    derived from. Returns False when the dataset already has the feature.
    """
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    if ANCHOR_POSES_KEY in info["features"]:
        return False
    if len(anchor_poses) < N_RECORDED_ANCHORS:
        raise ValueError(
            f"cannot add the {ANCHOR_POSES_KEY} feature without {N_RECORDED_ANCHORS} anchor poses"
        )

    flat = pack_anchor_poses(anchor_poses)
    field = pa.field(ANCHOR_POSES_KEY, pa.list_(pa.float32(), len(ANCHOR_POSE_NAMES)))

    for f in sorted(root.glob("data/chunk-*/file-*.parquet")):
        table = pq.read_table(f)
        column = pa.array([flat.tolist()] * table.num_rows, type=field.type)
        pq.write_table(table.append_column(field, column), f)

    feature = anchor_poses_feature()[ANCHOR_POSES_KEY]
    info["features"][ANCHOR_POSES_KEY] = {
        "dtype": feature["dtype"],
        "shape": list(feature["shape"]),
        "names": list(feature["names"]),
    }
    info_path.write_text(json.dumps(info, indent=4))

    value = flat.astype(np.float64).tolist()
    zeros = [0.0] * len(value)
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats[ANCHOR_POSES_KEY] = {
        "min": value, "max": value, "mean": value, "std": zeros,
        "count": [int(info["total_frames"])],
        **{q: list(value) for q in ("q01", "q10", "q50", "q90", "q99")},
    }
    stats_path.write_text(json.dumps(stats, indent=4))

    # Per-episode stats too: a feature present in info.json but missing from the episode
    # stats trips anything that later re-aggregates them (delete_episodes, merge).
    for f in sorted((root / "meta" / "episodes").glob("**/*.parquet")):
        table = pq.read_table(f)
        if f"stats/{ANCHOR_POSES_KEY}/min" in table.schema.names:
            continue
        lengths = table.column("length").to_pylist()
        vector_type = pa.list_(pa.float64(), len(value))
        for stat, row in [("min", value), ("max", value), ("mean", value), ("std", zeros),
                          ("q01", value), ("q10", value), ("q50", value),
                          ("q90", value), ("q99", value)]:
            table = table.append_column(
                pa.field(f"stats/{ANCHOR_POSES_KEY}/{stat}", vector_type),
                pa.array([row] * table.num_rows, type=vector_type),
            )
        count_type = pa.list_(pa.int64(), 1)
        table = table.append_column(
            pa.field(f"stats/{ANCHOR_POSES_KEY}/count", count_type),
            pa.array([[int(n)] for n in lengths], type=count_type),
        )
        pq.write_table(table, f)

    logging.info(f"Added the {ANCHOR_POSES_KEY} feature, filled with the calibration used for conversion")
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
    """Index ranges over which the wrist was turning, as [(start, stop), ...].

    A run ends on the frame the wrist stopped, which is the angle the offsets point
    at. Runs separated by less than gap_s are merged, so a turn that pauses briefly
    counts once, and runs that travel less than min_travel_deg are dropped as sensor
    noise rather than intent.
    """
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
    """Radians from each frame's wrist angle to the angle the next turn ends at.

    Every run of wrist movement is found, and the frames leading up to and including
    that run point at the angle it stopped on. After the last run there is nothing
    left to turn, so the offset is zero. With anticipate=False only the frames inside
    a run are labelled, which keeps the demonstrator's timing but leaves the channel
    near-zero almost everywhere - the sparsity that makes a rate channel hard to
    learn in the first place.
    """
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

def fuse_goal_to_room(action, gripper_pos, gripper_rot_6d, anchor_poses):
    """Fuse the per-camera goal predictions into one room-frame goal.

    Returns (goal_room, spread) where spread is the mean distance of the
    individual estimates from the fused one - a usable confidence signal, since
    the three views only agree when they agree about where the target is.
    Cameras whose pose is unknown are skipped.
    """
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
    goal = np.average(estimates, axis=0, weights=weights)
    spread = float(np.mean(np.linalg.norm(estimates - goal, axis=1)))
    return goal, spread


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
