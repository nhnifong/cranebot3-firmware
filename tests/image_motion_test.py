import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import nf_robot.common.definitions as model_constants
from nf_robot.common.image_motion import (heading_error, image_shift, mean_heading_error,
                                          wrap_angle)
from nf_robot.host.observer import AsyncObserver


def textured(width=640, height=480, seed=0):
    """A grayscale frame with floor-like texture: blurred noise at a couple of scales."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (height, width)).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), 2.0) + cv2.GaussianBlur(image, (0, 0), 8.0)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class TestImageShift(unittest.TestCase):
    def test_a_slid_frame_measures_the_slide(self):
        reference = textured()
        live = cv2.warpAffine(reference, np.float32([[1, 0, 12], [0, 1, -7]]),
                              (reference.shape[1], reference.shape[0]))
        dx, dy, confidence, inliers = image_shift(reference, live)
        self.assertAlmostEqual(dx, 12, delta=0.5)
        self.assertAlmostEqual(dy, -7, delta=0.5)
        self.assertGreater(confidence, 0.5)

    def test_a_blank_frame_has_nothing_to_match(self):
        blank = np.full((480, 640), 128, np.uint8)
        self.assertEqual(image_shift(blank, blank), (0.0, 0.0, 0.0, 0))


class TestHeadings(unittest.TestCase):
    def test_errors_are_counterclockwise_and_wrap(self):
        self.assertAlmostEqual(heading_error([1, 0], [0, 1]), np.pi / 2)
        self.assertAlmostEqual(heading_error([-1, 0.01], [-1, -0.01]), 0.02, places=3)
        self.assertAlmostEqual(wrap_angle(3 * np.pi), np.pi)

    def test_the_mean_of_errors_either_side_of_half_a_turn(self):
        mean, spread = mean_heading_error([np.pi - 0.1, -np.pi + 0.1])
        self.assertAlmostEqual(abs(mean), np.pi, places=6)
        self.assertAlmostEqual(spread, 0.1, places=6)


class FakeGripper:
    """Only the heading the observer turns camera vectors into the room with."""

    def __init__(self, believed_spin):
        self.believed_spin = believed_spin

    def gripper_body_room_rotation(self, timestamp=None):
        return Rotation.from_rotvec([0.0, 0.0, -self.believed_spin])


class TestSpinErrors(unittest.IsolatedAsyncioTestCase):
    async def test_the_error_is_the_correction_recover_spin_applies(self):
        """Round a circle, the floor slides as a camera at the true spin would see it; the
        observer, believing another spin, must find the difference as the error, which is
        what recover_spin adds to frame_room_spin."""
        true_spin, believed_spin = 0.9, 0.4
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        ob.gripper_client = FakeGripper(believed_spin)
        camera_to_body = (Rotation.from_euler('x', 90, degrees=True)
                          * Rotation.from_rotvec(model_constants.gripper_camera[0]))
        room_to_camera = (Rotation.from_rotvec([0.0, 0.0, -true_spin]) * camera_to_body).inv()
        intrinsics = ob.gripper_camera_intrinsics()
        depth = 0.8

        track, pairs = [], []
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 40, endpoint=False)):
            t = float(i)
            track.append((t, np.array([0.2 * np.cos(angle), 0.2 * np.sin(angle), 1.5])))
        for (t_a, p_a), (t_b, p_b) in zip(track, track[1:]):
            in_camera = room_to_camera.apply(p_b - p_a)
            dx = -in_camera[0] * intrinsics[0][0] / depth
            dy = -in_camera[1] * intrinsics[1][1] / depth
            pairs.append((t_a, t_b, dx, dy, 0.9, 100, depth))

        errors = ob._spin_errors(track, pairs, min_confidence=0.5, min_move_m=0.004)
        self.assertEqual(len(errors), len(pairs))
        correction, spread = mean_heading_error(errors)
        self.assertAlmostEqual(correction, true_spin - believed_spin, delta=np.radians(0.5))


if __name__ == '__main__':
    unittest.main()
