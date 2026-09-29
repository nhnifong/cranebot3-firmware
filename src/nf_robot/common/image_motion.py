"""How far the scene slid between two camera frames, and what that says about headings."""

import cv2
import numpy as np

# Share of the frame, from the top, that is compared. The rest of a gripper camera frame is
# fingers and whatever is a hand's width under the camera: the part that slides furthest for
# a given move and agrees with nothing else in the view.
GRIPPER_COMPARE_CROP = 0.62
MIN_MATCHES = 20   # fewer than this and there is nothing to fit a move to


def image_shift(reference_gray, live_gray, crop=GRIPPER_COMPARE_CROP, min_matches=MIN_MATCHES):
    """How far the scene slid between two grayscale frames.

    Returns (dx, dy, confidence, inliers): the slide in pixels from reference to live, at the
    middle of the frame, the share of feature matches that agree on it (0 to 1), and how many
    that was. All zeros when there is nothing to match.

    Matched features rather than phase correlation, because the two frames are not one
    image shifted. What is a hand's width under the camera sits in the same view as a
    room several metres off, so a sideways move slides the near and far halves by quite
    different amounts, and phase correlation - which can only answer with one shift for
    the whole frame - reported peaks of 0.01 on real pairs that were obviously the same
    corner of the room. RANSAC instead picks whichever depth the bulk of the matches
    agree on and throws the rest out, passing people included. The bottom of the frame
    is cropped off first, as the nearest and so worst-parallax part of the view.
    """
    if live_gray.shape != reference_gray.shape:
        live_gray = cv2.resize(live_gray, (reference_gray.shape[1], reference_gray.shape[0]),
                               interpolation=cv2.INTER_AREA)
    keep = slice(0, int(reference_gray.shape[0] * crop))
    reference, live = reference_gray[keep], live_gray[keep]

    detector = cv2.ORB_create(nfeatures=1500)
    ref_kp, ref_desc = detector.detectAndCompute(reference, None)
    live_kp, live_desc = detector.detectAndCompute(live, None)
    if ref_desc is None or live_desc is None or len(ref_kp) < 8 or len(live_kp) < 8:
        return 0.0, 0.0, 0.0, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(ref_desc, live_desc)
    if len(matches) < min_matches:
        return 0.0, 0.0, 0.0, 0
    ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in matches])
    live_pts = np.float32([live_kp[m.trainIdx].pt for m in matches])
    # partial affine: a translation, plus the rotation and scale that a wrist a degree
    # out or a few centimetres of height put on top of it, which are not the answer but
    # do have to be absorbed before the translation is right
    transform, inliers = cv2.estimateAffinePartial2D(ref_pts, live_pts, method=cv2.RANSAC,
                                                     ransacReprojThreshold=3.0)
    if transform is None:
        return 0.0, 0.0, 0.0, 0
    # read the shift off at the middle of the frame rather than from the transform's own
    # translation, which is measured at the corner and so carries the rotation with it
    center = np.float32([reference.shape[1] / 2.0, reference.shape[0] / 2.0, 1.0])
    moved = transform @ center
    agreed = int(inliers.sum())
    return (float(moved[0] - center[0]), float(moved[1] - center[1]),
            agreed / len(matches), agreed)


def wrap_angle(angle):
    """angle in radians, brought into (-pi, pi]."""
    return float(np.pi - (np.pi - angle) % (2 * np.pi))


def heading_error(commanded_xy, measured_xy):
    """The counterclockwise angle in radians from the commanded room direction to the measured
    one, in (-pi, pi]."""
    return wrap_angle(np.arctan2(measured_xy[1], measured_xy[0])
                      - np.arctan2(commanded_xy[1], commanded_xy[0]))


def mean_heading_error(errors):
    """(circular mean, largest departure from it) of a list of heading errors in radians."""
    errors = np.asarray(errors, dtype=float)
    mean = float(np.arctan2(np.mean(np.sin(errors)), np.mean(np.cos(errors))))
    spread = max(abs(wrap_angle(e - mean)) for e in errors)
    return mean, spread
