#!/usr/bin/env python

"""How the grasp point's place in each frame is decided, and the choices of method.

The position labels are the only ones in this dataset with no observed counterpart: the
timing labels come off pressure, the finger labels off the operator's own commands, but
`target_uv` is arithmetic all the way down. Which arithmetic is a choice, and until this
module existed it was one choice hard-coded in two tools that had to agree.

Every method answers the same question - where, in each frame of an episode, is the point
the jaws ended up closing on - and returns the same thing, a track of (u, v, distance) or
None, one entry per row. What they disagree about is which recorded quantity to trust:

    room-delta            the gripper's reported position, from Positioner2 at record time
    dead-reckon           the commanded velocity, integrated away from the grasp
    dead-reckon-observed  the gantry's reported velocity, integrated the same way
    optical-flow          the pixels, tracked away from the grasp, no telemetry at all

They are not ranked. Each is wrong in its own way and the useful thing is the comparison:
a constant gap between two methods is a mount or calibration error, a gap that opens with
time from the grasp is drift in whichever one is integrating, and a gap that appears only
when the gripper touches down is the position estimator flipping hang points.

`mine_teleop` and `label_video` both drive from here and both take the same arguments, so
a video can be rendered with the method a dataset was mined with and be a picture of those
labels rather than of a second implementation of them.
"""

import numpy as np

from nf_robot.common.config_loader import create_default_config
from nf_robot.ml.visual_servoing.geometry import (
    CAMERA_POS_BODY, CAMERA_ROT_BODY, JAW_POS_BODY, delta_in_camera, point_in_camera,
    rotate_about_vertical,
)

# Closer than this to the lens and the projection stops meaning anything, while still
# producing a plausible looking coordinate.
MIN_DEPTH_M = 0.02


DEFAULT_UV_METHOD = "room-delta"

# Where the jaws appear is not a constant - it is a fixed point in the body frame seen
# from a lens 2.7cm in front of it, so it slides up the frame as the gripper climbs. On
# naavox/red-dot it runs from v=0.72 with the fingertips on the floor to v=0.46 a metre up,
# and nothing about that is optional: aiming at one number instead teaches an approach to
# converge on the jaws only at the range that number was read at.
#
# This replaces a JAW_UV constant of (0.5, 0.692), which was never measured - it was the
# mirror of the pre-flip (0.5, 0.308) and happened to sit 2px from the truth at grasping
# range and a quarter of a frame away at half a metre.
def jaw_uv(laser_rangefinder, calibration):
    """Where the point the jaws will close on sits in the frame, at this range.

    None when the rangefinder read too little to place anything: a gripper resting on the
    floor reports a few millimetres, and the jaw point is then at or behind the lens, where
    there is no answer rather than a large one.
    """
    projected = project_camera(jaw_in_camera(laser_rangefinder), calibration)
    return None if projected is None else projected[:2]


def _jaw_uv_of(row, calibration):
    """The mount's answer for one row, which is what every anchor here defaults to."""
    return jaw_uv(row["laser_rangefinder"], calibration)


def anchor_is_usable(row, calibration):
    """Whether this frame can anchor an episode's labels.

    It cannot when the rangefinder read too little to put the jaw point in front of the
    lens, which is what a gripper closing while resting on the floor reports. Every method
    hangs the target off that reading, so the answer is a property of the frame rather than
    of the method.
    """
    return _jaw_uv_of(row, calibration) is not None


def jaw_in_camera(laser_rangefinder):
    """The jaw point in the camera's optical frame: straight below the jaws, on the floor.

    The rangefinder sits beside the lens and measures down the body axis, so it gives the
    lens's height; the jaws are `CAMERA_POS_BODY` behind the lens and the floor is that
    much further below them.
    """
    drop = float(laser_rangefinder) - CAMERA_POS_BODY[2] + JAW_POS_BODY[2]
    return CAMERA_ROT_BODY.inv().apply(JAW_POS_BODY + np.array([0.0, 0.0, -drop])
                                       - CAMERA_POS_BODY)


DEFAULT_UV_METHOD = "room-delta"

# Optical flow tracks a patch rather than the one pixel the target sits on: a point on
# carpet has nothing to lock to, and a grid whose members disagree is a grid that says so.
FLOW_GRID = 5
# Half-width of that patch, as a fraction of the frame width. Wide enough to contain
# something with texture in it, narrow enough that it is still mostly the target and not
# the floor sliding past behind it.
FLOW_PATCH_FRAC = 0.05
# Points surviving the forward-backward check, below which the step is called a failure
# rather than answered with whatever the survivors said.
FLOW_MIN_POINTS = 6
# (pixels) how far a point may land from where it started after being tracked there and
# back. The standard occlusion and flat-region test: a point that cannot retrace its own
# step did not track, whatever its status flag says.
FLOW_FB_TOLERANCE = 1.0
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(3, 30, 0.01))  # 3 == COUNT | EPS, spelled out so cv2 imports lazily


def gripper_camera_calibration():
    """Gripper camera intrinsics as fractions of the frame: (fx, fy), (cx, cy).

    The wide calibration rather than camera_cal: the gripper streams the full-sensor
    16:9 field of view, which is what camera_cal_wide was chessboard-calibrated for.

    Normalized, because that makes the labels independent of what resolution the frames
    happen to be stored at - a resize moves every pixel coordinate and leaves every
    normalized one alone. A *crop* does not, which is why the recipe that builds the
    source dataset sets center_crop and pad_clamp false.
    """
    cal = create_default_config().camera_cal_wide
    K = np.array(cal.intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    width, height = cal.resolution.width, cal.resolution.height
    return (K[0, 0] / width, K[1, 1] / height), (K[0, 2] / width, K[1, 2] / height)


def project_camera(p_cam, calibration):
    """A camera-frame point as normalized (u, v) plus its distance, or None if it is behind.

    0..1 spans the visible frame whatever resolution it is stored at.

    Pinhole only, no distortion: the wide calibration's coefficients are small (k1 is
    -0.026) next to the approximations above it, and the distortion polynomial diverges
    wildly outside the field of view - which is exactly where this has to stay sane, since
    the whole point is labelling targets past the frame edge.
    """
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
    """The same, for a room-frame vector from the gripper to the point.

    The delta is what every method here actually carries: an integrator never knows where
    the gripper is, only how far it has moved since the frame it was anchored on.
    """
    return project_camera(delta_in_camera(delta_room, spin), calibration)


def unproject(u, v, distance, calibration):
    """The inverse of `project_camera`: a camera-frame point at that bearing and range.

    `distance` is along the ray, the same quantity `project_camera` returns, so the two
    round-trip.
    """
    (fx, fy), (cx, cy) = calibration
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    return ray * (float(distance) / np.linalg.norm(ray))


def camera_delta_to_room(p_cam, spin):
    """A camera-frame point as the room-frame vector from the gripper body origin to it."""
    from nf_robot.ml.visual_servoing.geometry import camera_to_body

    return rotate_about_vertical(camera_to_body(p_cam) + CAMERA_POS_BODY, -float(spin))


def grasp_point_room(row):
    """Where the object was, in the room, at the instant of the grasp.

    Straight down from the *jaws* by what the rangefinder read, adjusted for the lens
    sitting 6mm above them. Not down from the lens, which is what this did until the
    red-dot measurement showed the jaws are 2.7cm behind it: hanging the target under the
    lens put every label 2.7cm toward the nose of the thing that was actually picked up,
    and a servo loop that then nulled the offset to *that* aimed the same 2.7cm past the
    fingers. See geometry.JAW_POS_BODY.

    Down rather than along the optical axis because the beam points down the body axis;
    the lens is what is tilted, not the sensor.
    """
    return np.asarray(row["gripper_pos"], dtype=np.float64) + grasp_delta_room(row)


def grasp_delta_room(row):
    """The same point as a room-frame vector from the gripper, which is all any method needs.

    Nothing in it comes from the position estimate - it is the mount and the rangefinder -
    so the dead-reckoning methods can anchor on it without taking on the dependency they
    exist to avoid. The wrist heading drops out too now that the point hangs from the body
    origin: straight down is straight down whatever the gripper is facing.
    """
    drop = float(row["laser_rangefinder"]) - CAMERA_POS_BODY[2] + JAW_POS_BODY[2]
    return np.asarray(JAW_POS_BODY, dtype=np.float64) + np.array([0.0, 0.0, -drop])


def jaw_delta_room(row, calibration, jaw_uv=None):
    """The room-frame vector from the gripper to the jaws.

    `jaw_uv` overrides where in the frame the jaws are taken to be, which is only useful
    for testing the aim against something other than the mount; left None it is the mount's
    own answer and this is `grasp_delta_room`.
    """
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
    """The grasp point fixed in the room, differenced against the reported gripper position.

    The original method and still the default. Its whole error budget is the position
    estimate: Positioner2's output at record time is what says where the camera was, and
    swing, video latency and a line going slack at touchdown all enter here and nowhere
    else.
    """
    target = grasp_point_room(rows[grasp])
    return [target - np.asarray(r["gripper_pos"], dtype=np.float64) for r in rows]


def _deltas_dead_reckon(rows, grasp, calibration, jaw_uv, field="vel_cmd", room_frame=False):
    """The jaws at the grasp, carried to the other frames by integrating velocity.

    The target does not move before the grasp, so the vector to it changes by exactly how
    far the gripper went, and that is a quantity the gantry reports without consulting the
    position estimate at all. Integrating outwards from the grasp rather than forwards from
    the episode start puts the anchor where the answer is known and lets the drift
    accumulate into the early frames, which are the ones that matter least.

    A zero-order hold on each command: the velocity at frame k is taken to hold until
    frame k+1, which is what a velocity command means to the gantry.

    What it cannot do is recover from a command that was not obeyed. Over the last second
    before a grasp that is a small thing; over ten it is the whole story, and the further
    from the grasp the more the track is a picture of what was asked for rather than of
    what happened.
    """
    _require(rows, field, "dead-reckon" if field == "vel_cmd" else "dead-reckon-observed")

    steps = []
    for k in range(len(rows) - 1):
        v = np.asarray(rows[k][field], dtype=np.float64)
        dt = float(rows[k + 1]["timestamp"] - rows[k]["timestamp"])
        # The commanded velocity is in the gripper's own frame and turns with it; the
        # gantry reports its own in room axes already. Measured, not assumed: integrating
        # the observed velocity through -spin scores worse against the recorded track than
        # integrating it raw, and the commanded one scores worse raw than through -spin.
        steps.append((v if room_frame else rotate_about_vertical(v, -float(rows[k]["spin"]))) * dt)

    deltas = [None] * len(rows)
    deltas[grasp] = jaw_delta_room(rows[grasp], calibration, jaw_uv)
    for k in range(grasp - 1, -1, -1):
        deltas[k] = deltas[k + 1] + steps[k]
    for k in range(grasp + 1, len(rows)):
        deltas[k] = deltas[k - 1] - steps[k - 1]
    return deltas


def _deltas_dead_reckon_observed(rows, grasp, calibration, jaw_uv):
    """Dead reckoning off the gantry's reported velocity instead of the command.

    Worth having beside the commanded one because the two fail differently: the command is
    what was asked for and the gantry's measurement is what the spools did, so where they
    agree the approach was flown as commanded, and where they part the difference is lag,
    swing or slack. Neither consults the hang-point estimate, which is the point of both.
    """
    return _deltas_dead_reckon(rows, grasp, calibration, jaw_uv,
                               field="vel_obs", room_frame=True)


def range_to_floor(u, v, row, calibration):
    """How far along the (u, v) ray the floor is, given what the rangefinder read.

    The beam measures straight down the body axis, so it gives the perpendicular distance
    to the floor plane and a ray at angle theta off it reaches that plane at L / cos theta.
    Assumes the target is on the floor and the floor is level, which is the same assumption
    `grasp_point_room` makes; what it does not assume is any knowledge of where the gripper
    is, which is what keeps optical flow free of the position estimate entirely.
    """
    ray = unproject(u, v, 1.0, calibration)
    ray = ray / np.linalg.norm(ray)
    cosine = float(ray @ CAMERA_ROT_BODY.inv().apply([0.0, 0.0, -1.0]))
    if cosine <= 1e-3:
        return None
    return float(row["laser_rangefinder"]) / cosine


def _flow_step(prev_gray, next_gray, uv):
    """One frame of tracking: where the point at `uv` in prev_gray went in next_gray.

    None when the patch could not be tracked, which is a real and frequent answer - the
    target leaving the frame, a finger crossing it, or a stretch of featureless carpet.
    Saying so beats extrapolating, because a flow track that has lost its target does not
    drift away slowly, it snaps onto whatever else is nearby.
    """
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
    # The median, not the mean: half the patch can be floor sliding past behind the object
    # and the answer should still be the object's.
    moved = np.median((forward - pts).reshape(-1, 2)[good], axis=0)
    return (uv[0] + float(moved[0]) / w, uv[1] + float(moved[1]) / h)


def _track_optical_flow(rows, grasp, calibration, jaw_uv, frames):
    """The target followed through the pixels, anchored at the jaws and tracked outwards.

    The one method here that consults no telemetry at all except the rangefinder: where the
    target is in frame t comes from frame t and frame t+1 and nothing else, so swing, video
    latency and a hang point flipping cannot enter it. That makes it the independent
    opinion the other three can be checked against rather than a better version of them.

    What it buys is paid for twice. It has to decode the video, which the others do not, and
    it can only follow a target that stays visible: once the object leaves the frame the
    track ends rather than guessing, so this method produces no off-canvas labels at all and
    cannot teach the sock-past-the-bottom-edge case the 1.25x canvas exists for. Its errors
    are also not the others' errors - it accumulates per-frame tracking error the same way
    dead reckoning accumulates velocity error, and it fails abruptly rather than gradually.
    """
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
            # Only ever two frames are compared, so the cache is a window rather than the
            # episode: holding a 675 frame video would cost half a gigabyte.
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


# The two methods that work by producing a room-frame vector from the gripper to the
# target, which is then projected. Named because the invariant between them - with exact
# velocities their tracks may differ only by their anchors - is stated in that space and
# nowhere else.
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
    """Where the grasp point sits in every frame of an episode: (u, v, distance) or None.

    One entry per row, indexed the same, so a caller can ask about any frame of the episode
    including the ones outside the mined window. None means the method has no answer there:
    at or behind the lens for the projecting methods, and the track having been lost for
    optical flow.

    A whole track rather than a per-frame call because every method but `room-delta` is
    cumulative - frame t's answer is only defined relative to the anchor at the grasp.

    `frames` is a callable taking a frame_index and returning that frame as BGR. Only the
    pixel methods need it and they raise by name when it is missing.
    """
    if method not in UV_METHODS:
        raise SystemExit(f"unknown --uv_method {method}; have {', '.join(UV_METHODS)}")
    return UV_METHODS[method](rows, grasp, calibration, jaw_uv, frames)


def add_uv_arguments(parser):
    """The method flags, added identically to every tool that decides a uv.

    Shared so that a label video and the mining run it is meant to be a picture of cannot
    be given different arguments without someone noticing.
    """
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
