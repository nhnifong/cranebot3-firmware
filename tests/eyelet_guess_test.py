"""
Unit tests for the external eyelet starting guess and the pass-3 plausibility guard.

The guess doubles as the eyelet_reg target, so a wrong one does not merely slow the fit down,
it holds the answer near itself. The guard exists because the gripper-card pass carries two
orders of magnitude more residuals than the pass before it, and so lowers the total cost even
when it gets there by deforming the room.
"""

import unittest

import numpy as np

from nf_robot.host.eyelet_calibration import (
    STRUCTURAL_COST_TERMS,
    rectangular_room_eyelet_guess,
    refinement_is_plausible,
)


def anchors_at(p0, p1, yaw0=0.0, yaw1=0.0):
    return np.array([[[0.0, 0.0, yaw0], list(p0)], [[0.0, 0.0, yaw1], list(p1)]], dtype=float)


class RectangularRoomEyeletGuessTest(unittest.TestCase):

    def test_square_room_recovers_the_other_two_corners(self):
        # the default config's layout: anchors on one diagonal, eyelets on the other
        guess = rectangular_room_eyelet_guess(anchors_at((3, 3, 2), (-3, -3, 2)))
        np.testing.assert_allclose(guess, [[-3, 3, 2], [3, -3, 2]], atol=1e-9)

    def test_eyelet_zero_belongs_to_anchor_zero(self):
        # the pair is symmetric about the anchor diagonal, so only the order distinguishes
        # them, and the state vector reads eyelet 0 as anchor 0's.
        guess = rectangular_room_eyelet_guess(anchors_at((3, 3, 2), (-3, -3, 2)))
        self.assertGreater(guess[0][1], 0.0)
        self.assertLess(guess[1][1], 0.0)

    def test_each_eyelet_takes_its_own_anchors_height(self):
        guess = rectangular_room_eyelet_guess(anchors_at((3, 3, 2.5), (-3, -3, 2.1)))
        self.assertAlmostEqual(guess[0][2], 2.5)
        self.assertAlmostEqual(guess[1][2], 2.1)

    def test_height_ignores_anchor_yaw(self):
        # the guess is built in the room frame, so an anchor fitted off plumb cannot tip a
        # multi-metre offset down out of the pull-point plane.
        level = rectangular_room_eyelet_guess(anchors_at((3, 3, 2), (-3, -3, 2)))
        spun = rectangular_room_eyelet_guess(
            anchors_at((3, 3, 2), (-3, -3, 2), yaw0=1.1, yaw1=-2.4))
        np.testing.assert_allclose(level, spun, atol=1e-9)

    def test_guesses_are_equidistant_from_their_anchors(self):
        guess = rectangular_room_eyelet_guess(anchors_at((1.79, -1.76, 2.51), (-2.07, 1.75, 2.36)))
        positions = np.array([[1.79, -1.76, 2.51], [-2.07, 1.75, 2.36]])
        spans = [np.linalg.norm(guess[i][:2] - positions[i][:2]) for i in (0, 1)]
        self.assertAlmostEqual(spans[0], spans[1], places=6)
        # the side of the square on the anchor diagonal
        diagonal = np.linalg.norm(positions[1][:2] - positions[0][:2])
        self.assertAlmostEqual(spans[0], diagonal / np.sqrt(2), places=6)


class RefinementGuardTest(unittest.TestCase):

    @staticmethod
    def fit(before, after):
        return {'input_costs': dict(before), 'costs': dict(after)}

    def test_rejects_a_pass_that_buys_its_fit_by_leaning_the_anchors(self):
        ok, reason = refinement_is_plausible(self.fit(
            {'anchor_tilt': 0.0647, 'anchor_planarity': 0.0159, 'shape_match': 0.0005,
             'eyelet_reg': 0.0, 'gripper_cards': 2.1080},
            {'anchor_tilt': 0.3958, 'anchor_planarity': 0.0435, 'shape_match': 0.0117,
             'eyelet_reg': 0.0538, 'gripper_cards': 0.5650}))
        self.assertFalse(ok)
        self.assertIn('anchor_tilt', reason)

    def test_total_cost_falling_is_not_enough_to_pass(self):
        before = {'anchor_tilt': 0.1260, 'anchor_planarity': 0.0213, 'shape_match': 0.0195,
                  'eyelet_reg': 0.0, 'gripper_cards': 1.3066}
        after = {'anchor_tilt': 0.1846, 'anchor_planarity': 0.0521, 'shape_match': 0.0094,
                 'eyelet_reg': 0.0604, 'gripper_cards': 0.4932}
        self.assertLess(sum(after.values()), sum(before.values()))
        ok, _ = refinement_is_plausible(self.fit(before, after))
        self.assertFalse(ok)

    def test_accepts_a_refinement_that_leaves_the_room_alone(self):
        ok, _ = refinement_is_plausible(self.fit(
            {'anchor_tilt': 0.030, 'anchor_planarity': 0.010, 'shape_match': 0.001,
             'eyelet_reg': 0.0, 'gripper_cards': 0.9},
            {'anchor_tilt': 0.032, 'anchor_planarity': 0.011, 'shape_match': 0.002,
             'eyelet_reg': 0.004, 'gripper_cards': 0.3}))
        self.assertTrue(ok)

    def test_small_absolute_growth_survives_the_ratio(self):
        # a warm start already sitting near zero would trip any ratio on rounding alone
        ok, _ = refinement_is_plausible(self.fit(
            {'anchor_tilt': 0.001, 'anchor_planarity': 0.0, 'shape_match': 0.0, 'eyelet_reg': 0.0},
            {'anchor_tilt': 0.020, 'anchor_planarity': 0.005, 'shape_match': 0.0,
             'eyelet_reg': 0.004}))
        self.assertTrue(ok)

    def test_a_pass_with_no_recorded_start_is_not_judged(self):
        ok, reason = refinement_is_plausible({'costs': {'anchor_tilt': 9.9}})
        self.assertTrue(ok)
        self.assertIn('no starting costs', reason)

    def test_every_structural_term_is_weighed(self):
        for term in STRUCTURAL_COST_TERMS:
            ok, _ = refinement_is_plausible(self.fit({term: 0.10}, {term: 0.40}))
            self.assertFalse(ok, f'{term} alone should be able to fail the guard')


if __name__ == '__main__':
    unittest.main()
