#!/usr/bin/env python

"""How the grasp point's place in each frame is decided, shared by mine_teleop and label_video.

    room-delta            the gripper's reported position, from Positioner2 at record time
    dead-reckon           the commanded velocity, integrated away from the grasp
    dead-reckon-observed  the gantry's reported velocity, integrated the same way
    optical-flow          the pixels, tracked away from the grasp, no telemetry at all
"""

import numpy as np

from nf_robot.common.config_loader import create_default_config
from nf_robot.ml.visual_servoing.geometry import (
    CAMERA_POS_BODY, CAMERA_ROT_BODY, JAW_POS_BODY, delta_in_camera, point_in_camera,
    rotate_about_vertical,
)

# Closer than this to the lens, the projection is meaningless.
MIN_DEPTH_M = 0.02


DEFAULT_UV_METHOD = "room-delta"

# Where the jaws appear slides up the frame as the gripper climbs, so it is computed per
# range rather than fixed.
def jaw_uv(laser_rangefinder, calibration):
    """Where the jaw point sits in the frame at this range, or None if the rangefinder reads
    too little."""
    projected = project_camera(jaw_in_camera(laser_rangefinder), calibration)
    return None if projected is None else projected[:2]


def _jaw_uv_of(row, calibration):
    return jaw_uv(row["laser_rangefinder"], calibration)


def anchor_is_usable(row, calibration):
    return _jaw_uv_of(row, calibration) is not None


def jaw_in_camera(laser_rangefinder):
    """The jaw point in the camera's optical frame: straight below the jaws, on the floor."""
    drop = float(laser_rangefinder) - CAMERA_POS_BODY[2] + JAW_POS_BODY[2]
    return CAMERA_ROT_BODY.inv().apply(JAW_POS_BODY + np.array([0.0, 0.0, -drop])
                                       - CAMERA_POS_BODY)


DEFAULT_UV_METHOD = "room-delta"

# Optical flow tracks a patch, since a single carpet pixel has nothing to lock to.
FLOW_GRID = 5
# Half-width of that patch, as a fraction of the frame width.
FLOW_PATCH_FRAC = 0.05
# Minimum points surviving the forward-backward check for a step to count.
FLOW_MIN_POINTS = 6
# (pixels) forward-backward error beyond which a point didn't track.
FLOW_FB_TOLERANCE = 1.0
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(3, 30, 0.01))  # 3 == COUNT | EPS, spelled out so cv2 imports lazily


def gripper_camera_calibration():
    """Gripper camera intrinsics from camera_cal_wide as fractions of the frame: (fx, fy),
    (cx, cy)."""
    cal = create_default_config().camera_cal_wide
    K = np.array(cal.intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    width, height = cal.resolution.width, cal.resolution.height
    return (K[0, 0] / width, K[1, 1] / height), (K[0, 2] / width, K[1, 2] / height)


def project_camera(p_cam, calibration):
    """A camera-frame point as normalized pinhole (u, v) plus distance, or None if it is behind."""
    (fx, fy), (cx, cy) = calibration
    if p_cam[2] < MIN_DEPTH_M:
        return None
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return float(u), float(v), float(np.linalg.norm(p_cam))


def project(point_room, gripper_pos, spin, calibration):
    """A room point as normalized (u, v) in the gripper camera, plus its distance."""
    return project_camera(point_in_camera(point_room, gripper_pos, spin), calibration)


def project_delta(delta_room, spin, calibration):
    """The same, for a room-frame vector from the gripper to the point."""
    return project_camera(delta_in_camera(delta_room, spin), calibration)


def unproject(u, v, distance, calibration):
    """The inverse of `project_camera`: a camera-frame point at that bearing and ray distance."""
    (fx, fy), (cx, cy) = calibration
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    return ray * (float(distance) / np.linalg.norm(ray))


def camera_delta_to_room(p_cam, spin):
    """A camera-frame point as the room-frame vector from the gripper body origin to it."""
    from nf_robot.ml.visual_servoing.geometry import camera_to_body

    return rotate_about_vertical(camera_to_body(p_cam) + CAMERA_POS_BODY, -float(spin))


def grasp_point_room(row):
    """Where the object was in the room at the grasp: straight down from the jaws by the
    rangefinder reading."""
    return np.asarray(row["gripper_pos"], dtype=np.float64) + grasp_delta_room(row)


def grasp_delta_room(row):
    """The same point as a room-frame vector from the gripper, independent of the position
    estimate."""
    drop = float(row["laser_rangefinder"]) - CAMERA_POS_BODY[2] + JAW_POS_BODY[2]
    return np.asarray(JAW_POS_BODY, dtype=np.float64) + np.array([0.0, 0.0, -drop])


def jaw_delta_room(row, calibration, jaw_uv=None):
    """The room-frame vector from the gripper to the jaws, optionally aimed at `jaw_uv`
    instead of the mount."""
    if jaw_uv is None:
        return grasp_delta_room(row)
    u, v = jaw_uv
    return camera_delta_to_room(
        unproject(u, v, float(row["laser_rangefinder"]), calibration), row["spin"])


def _require(rows, field, method):
    if rows[0].get(field) is None:
        raise SystemExit(
            f"--uv_method {method} needs '{field}', which this recording does not carry. "
            f"Re-record it, or mine with --uv_method {DEFAULT_UV_METHOD}.")


def _deltas_room_delta(rows, grasp, calibration, jaw_uv):
    """The grasp point fixed in the room, differenced against the reported gripper position
    (the default method)."""
    target = grasp_point_room(rows[grasp])
    return [target - np.asarray(r["gripper_pos"], dtype=np.float64) for r in rows]


def _deltas_dead_reckon(rows, grasp, calibration, jaw_uv, field="vel_cmd", room_frame=False):
    """The jaws at the grasp, carried to the other frames by integrating commanded velocity
    outwards from the grasp."""
    _require(rows, field, "dead-reckon" if field == "vel_cmd" else "dead-reckon-observed")

    steps = []
    for k in range(len(rows) - 1):
        v = np.asarray(rows[k][field], dtype=np.float64)
        dt = float(rows[k + 1]["timestamp"] - rows[k]["timestamp"])
        # The commanded velocity is in the gripper's frame and the observed one in room
        # axes, as measured.
        steps.append((v if room_frame else rotate_about_vertical(v, -float(rows[k]["spin"]))) * dt)

    deltas = [None] * len(rows)
    deltas[grasp] = jaw_delta_room(rows[grasp], calibration, jaw_uv)
    for k in range(grasp - 1, -1, -1):
        deltas[k] = deltas[k + 1] + steps[k]
    for k in range(grasp + 1, len(rows)):
        deltas[k] = deltas[k - 1] - steps[k - 1]
    return deltas


def _deltas_dead_reckon_observed(rows, grasp, calibration, jaw_uv):
    """Dead reckoning off the gantry's reported velocity instead of the command."""
    return _deltas_dead_reckon(rows, grasp, calibration, jaw_uv,
                               field="vel_obs", room_frame=True)


def range_to_floor(u, v, row, calibration):
    """Distance along the (u, v) ray to a level floor, given the rangefinder's perpendicular
    reading."""
    ray = unproject(u, v, 1.0, calibration)
    ray = ray / np.linalg.norm(ray)
    cosine = float(ray @ CAMERA_ROT_BODY.inv().apply([0.0, 0.0, -1.0]))
    if cosine <= 1e-3:
        return None
    return float(row["laser_rangefinder"]) / cosine


def _flow_step(prev_gray, next_gray, uv):
    """Track the patch at `uv` from prev_gray to next_gray, or None if it was lost."""
    import cv2

    h, w = prev_gray.shape
    half = FLOW_PATCH_FRAC * w
    grid = np.linspace(-half, half, FLOW_GRID)
    pts = np.array([[uv[0] * w + dx, uv[1] * h + dy] for dy in grid for dx in grid],
                   dtype=np.float32).reshape(-1, 1, 2)
    forward, ok_f, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, pts, None, **_LK)
    if forward is None:
        return None
    back, ok_b, _ = cv2.calcOpticalFlowPyrLK(next_gray, prev_gray, forward, None, **_LK)
    if back is None:
        return None
    retraced = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    good = (ok_f.ravel() == 1) & (ok_b.ravel() == 1) & (retraced < FLOW_FB_TOLERANCE)
    if int(good.sum()) < FLOW_MIN_POINTS:
        return None
    # The median, so floor sliding past behind the object doesn't pull the answer.
    moved = np.median((forward - pts).reshape(-1, 2)[good], axis=0)
    return (uv[0] + float(moved[0]) / w, uv[1] + float(moved[1]) / h)


def _track_optical_flow(rows, grasp, calibration, jaw_uv, frames):
    """The target tracked through the pixels outwards from the jaws at the grasp, ending
    where it is lost."""
    import cv2

    if frames is None:
        raise SystemExit(
            "--uv_method optical-flow needs the frames, and this caller did not offer "
            "them. It is the one method that reads pixels.")

    gray = {}

    def at(i):
        if i not in gray:
            image = frames(rows[i]["frame_index"])
            gray[i] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Only two frames are ever compared, so cache a window rather than the episode.
            for stale in [k for k in gray if abs(k - i) > 2]:
                del gray[stale]
        return gray[i]

    track = [None] * len(rows)
    anchor = tuple(jaw_uv) if jaw_uv is not None else _jaw_uv_of(rows[grasp], calibration)
    if anchor is None:
        raise SystemExit(
            "the rangefinder read nothing at the grasp frame, so there is no anchor to "
            "track from. mine_episode screens these out as 'no_range' before getting here.")
    track[grasp] = anchor
    for step in (-1, 1):
        uv = track[grasp]
        k = grasp + step
        while 0 <= k < len(rows):
            uv = _flow_step(at(k - step), at(k), uv)
            if uv is None:
                break
            track[k] = uv
            k += step

    out = []
    for uv, row in zip(track, rows):
        distance = None if uv is None else range_to_floor(uv[0], uv[1], row, calibration)
        out.append(None if distance is None else (uv[0], uv[1], distance))
    return out


# The methods that produce a room-frame gripper-to-target vector, which is then projected.
DELTA_METHODS = {
    "room-delta": _deltas_room_delta,
    "dead-reckon": _deltas_dead_reckon,
    "dead-reckon-observed": _deltas_dead_reckon_observed,
}

def _as_track(deltas_of):
    """A delta-producing method wearing the track-producing contract they all share."""
    def track(rows, grasp, calibration, jaw_uv, frames):
        return [project_delta(d, r["spin"], calibration)
                for d, r in zip(deltas_of(rows, grasp, calibration, jaw_uv), rows)]
    return track


UV_METHODS = {name: _as_track(fn) for name, fn in DELTA_METHODS.items()}
UV_METHODS["optical-flow"] = _track_optical_flow

# Methods that have to decode the video, and so cost what a full mining run costs.
PIXEL_METHODS = ("optical-flow",)


def target_track(rows, grasp, calibration, method=DEFAULT_UV_METHOD, jaw_uv=None,
                 frames=None):
    """Where the grasp point sits in every frame of an episode: (u, v, distance) or None per row."""
    if method not in UV_METHODS:
        raise SystemExit(f"unknown --uv_method {method}; have {', '.join(UV_METHODS)}")
    return UV_METHODS[method](rows, grasp, calibration, jaw_uv, frames)


def add_uv_arguments(parser):
    """Add the uv method flags, identically for every tool that decides a uv."""
    parser.add_argument("--uv_method", "--uv-method", dest="uv_method",
                        default=DEFAULT_UV_METHOD, choices=sorted(UV_METHODS),
                        help="How the grasp point's place in each frame is decided. "
                             "Default %(default)s.")
    parser.add_argument("--jaw_uv", "--jaw-uv", dest="jaw_uv", type=float, nargs=2,
                        default=None, metavar=("U", "V"),
                        help="Override where in the frame the jaws are taken to be. The "
                             "default is the mount's own answer, which depends on range - "
                             "uv_methods.jaw_uv - and is what you want unless you are "
                             "testing the aim itself.")
