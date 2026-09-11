"""
Unit tests for measuring each anchor camera's downward pitch from the calibration cards.

The first optimizer pass needs the tilts to run at all, so this has to work from the raw tag
observations alone, with no anchor pose and no room frame. The cards are flat, so each one's
normal is the room vertical, and a plumb anchor has to see that normal along its own z; the
tilt is the single angle that makes it so.
"""

import unittest

import cv2
import numpy as np

import nf_robot.common.definitions as model_constants
from nf_robot.host.eyelet_calibration import (
    ARP_CAMERA_MODEL_TILT_DEG,
    CAM_TILT_RANGE_DEG,
    estimate_cam_tilts,
)

CAMERA_ROTATION, _ = cv2.Rodrigues(
    np.asarray(model_constants.arp_anchor_camera[0], dtype=float).reshape(3))
UP_IN_CAMERA = CAMERA_ROTATION.T @ np.array([0.0, 0.0, 1.0])


def card_pose(true_tilt_deg, spin_rad=0.0, flipped=False):
    """A synthetic sighting of a flat card by a camera at true_tilt_deg, plumb anchor."""
    extra = np.radians(ARP_CAMERA_MODEL_TILT_DEG - true_tilt_deg)
    undo, _ = cv2.Rodrigues(np.array([-extra, 0.0, 0.0]))
    normal = undo @ UP_IN_CAMERA          # the card normal this camera would see
    # any frame whose third column is that normal; the card's spin about it is arbitrary
    helper = np.array([1.0, 0.0, 0.0])
    if abs(helper @ normal) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, normal)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack([x_axis, np.cross(normal, x_axis), normal])
    spin, _ = cv2.Rodrigues(normal * spin_rad)
    rotation = spin @ rotation
    if flipped:
        # the other IPPE_SQUARE branch: the tag reflected about an in-plane axis
        branch, _ = cv2.Rodrigues(rotation[:, 0] * np.pi * 0.8)
        rotation = branch @ rotation
    rvec, _ = cv2.Rodrigues(rotation)
    return (rvec.ravel(), np.array([0.1, 0.2, 3.0]))


def obs(cam0, cam1):
    """raw_obs holding one card per camera list, as snapshot_tag_observations_still shapes it."""
    return {'origin': [list(cam0), list(cam1)]}


class EstimateCamTiltsTest(unittest.TestCase):

    def test_recovers_the_tilt_a_camera_was_synthesised_at(self):
        tilts = estimate_cam_tilts(obs([card_pose(34.0)] * 6, [card_pose(43.5)] * 6), (30.0, 30.0))
        self.assertAlmostEqual(tilts[0], 34.0, places=4)
        self.assertAlmostEqual(tilts[1], 43.5, places=4)

    def test_the_two_anchors_are_measured_independently(self):
        # the failure that started this: one adapter well off while the other is nearly right
        tilts = estimate_cam_tilts(obs([card_pose(30.0)] * 6, [card_pose(50.0)] * 6), (30.0, 30.0))
        self.assertAlmostEqual(tilts[0], 30.0, places=4)
        self.assertAlmostEqual(tilts[1], 50.0, places=4)

    def test_the_cards_own_spin_does_not_matter(self):
        poses = [card_pose(34.0, spin_rad=s) for s in (0.0, 0.7, -1.9, 2.8, 1.2, -0.4)]
        tilts = estimate_cam_tilts(obs(poses, poses), (30.0, 30.0))
        self.assertAlmostEqual(tilts[0], 34.0, places=4)

    def test_a_branch_flip_is_dropped_rather_than_averaged_in(self):
        good = [card_pose(34.0)] * 6
        tilts = estimate_cam_tilts(obs(good + [card_pose(34.0, flipped=True)] * 2, good),
                                   (30.0, 30.0))
        self.assertAlmostEqual(tilts[0], 34.0, places=4)

    def test_a_camera_with_too_few_sightings_keeps_its_configured_value(self):
        tilts = estimate_cam_tilts(obs([card_pose(34.0)] * 6, [card_pose(43.5)] * 2), (30.0, 26.0))
        self.assertAlmostEqual(tilts[0], 34.0, places=4)
        self.assertEqual(tilts[1], 26.0)

    def test_a_camera_that_saw_nothing_keeps_its_configured_value(self):
        tilts = estimate_cam_tilts(obs([card_pose(34.0)] * 6, []), (30.0, 26.0))
        self.assertEqual(tilts[1], 26.0)

    def test_an_impossible_estimate_is_refused(self):
        beyond = CAM_TILT_RANGE_DEG[1] + 15.0
        tilts = estimate_cam_tilts(obs([card_pose(beyond)] * 6, [card_pose(34.0)] * 6), (30.0, 30.0))
        self.assertEqual(tilts[0], 30.0)
        self.assertAlmostEqual(tilts[1], 34.0, places=4)

    def test_none_and_zero_sightings_are_skipped(self):
        poses = [None, np.zeros((2, 3))] + [card_pose(34.0)] * 6
        tilts = estimate_cam_tilts(obs(poses, poses), (30.0, 30.0))
        self.assertAlmostEqual(tilts[0], 34.0, places=4)

    def test_the_gantry_tag_is_not_used(self):
        # it hangs off the moving gantry, so its normal is not the room vertical
        raw = obs([card_pose(34.0)] * 6, [card_pose(34.0)] * 6)
        raw['gantry'] = [[card_pose(70.0)] * 20, [card_pose(70.0)] * 20]
        tilts = estimate_cam_tilts(raw, (30.0, 30.0))
        self.assertAlmostEqual(tilts[0], 34.0, places=4)

    def test_a_configured_value_survives_unchanged_when_nothing_is_measurable(self):
        self.assertEqual(estimate_cam_tilts({}, (30.0, 26.0)), (30.0, 26.0))


if __name__ == '__main__':
    unittest.main()
