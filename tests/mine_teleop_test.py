"""Tests for what the teleop miner does with a target it cannot see.

The labels come from projecting one room point back through the approach, and nothing in
that arithmetic knows whether the point landed in the picture. Frames where it did not are
the ones these are about: the honest label there is "no target in view", not a position the
image gives no evidence for.
"""

import math
import unittest

import numpy as np

from nf_robot.ml.visual_servoing.mine_teleop import (
    CANVAS_SCALE, OFF_SCREEN_MARGIN, ShardWriter, find_lift, holding_label, in_view,
    mine_episode)


def rows_for(offsets, fps=30.0, grasp_at=60, length=90):
    """A descending approach that grasps at `grasp_at` and lifts afterwards.

    `offsets` gives the horizontal distance from the target at each frame, which is what
    decides whether the projected label lands in the frame.
    """
    rows = []
    for i in range(length):
        held = i >= grasp_at
        # down to the object, then up again carrying it - the rise after the grasp is
        # what find_grasp's caller uses to tell a real pick from closing on nothing
        height = (0.6 - 0.005 * i if not held
                  else 0.6 - 0.005 * grasp_at + 0.01 * (i - grasp_at))
        rows.append({
            "frame_index": i,
            "timestamp": i / fps,
            "gripper_pos": np.array([offsets[i], 0.0, height]),
            "spin": 0.0,
            # >= PRESSURE_THRESHOLD is a grasp; the recorded value rises with grip
            "pressure": 1.0 if held else 0.0,
            "wrist_angle": 540.0,
            "finger_angle": 0.0,
            "laser_rangefinder": max(height - 0.2, 0.05),
            "target_force": 0.0,
            "finger_speed": 0.0,
        })
    return rows


# intrinsics as fractions of the frame, the way gripper_camera_calibration returns them
CALIBRATION = ((439.3 / 684.0, 461.6 / 384.0), (0.5, 0.308))


class TestInView(unittest.TestCase):

    def test_inside_the_frame_is_in_view(self):
        self.assertTrue(in_view(0.5, 0.5))
        self.assertTrue(in_view(0.0, 1.0))

    def test_a_little_past_the_edge_is_still_worth_predicting(self):
        """The case the oversized canvas exists for: the object is in shot, the spot to
        grab it by has slipped past the edge."""
        self.assertTrue(in_view(1.0 + OFF_SCREEN_MARGIN / 2, 0.5))
        self.assertTrue(in_view(0.5, -OFF_SCREEN_MARGIN / 2))

    def test_far_outside_is_not(self):
        self.assertFalse(in_view(1.0 + OFF_SCREEN_MARGIN * 2, 0.5))
        self.assertFalse(in_view(0.5, -OFF_SCREEN_MARGIN * 2))

    def test_the_margin_is_inside_the_canvas(self):
        """Rows past the margin are still kept as frames; rows past the canvas are not
        mined at all, so the margin has to be the tighter of the two."""
        self.assertLess(OFF_SCREEN_MARGIN, (CANVAS_SCALE - 1.0) / 2.0)


class TestHoldingWindow(unittest.TestCase):
    """`holding` is positive only where the grip is proven: from the lift to the drop.

    Labelling from the grasp instead makes "an object is close in frame" sufficient for a
    positive, which is the cheapest feature in the episode and the one the head learns
    instead of the object being in the hand.
    """

    def test_before_the_grasp_is_a_negative(self):
        self.assertEqual(holding_label(5, grasp=10, lift=20, drop=None), 0)

    def test_between_the_grasp_and_the_lift_is_masked(self):
        self.assertIsNone(holding_label(15, grasp=10, lift=20, drop=None))

    def test_a_grasp_that_never_lifted_is_masked_throughout(self):
        """Closing on something and never picking it up proves nothing either way."""
        self.assertIsNone(holding_label(50, grasp=10, lift=None, drop=None))

    def test_from_the_lift_to_the_drop_is_positive(self):
        self.assertEqual(holding_label(25, grasp=10, lift=20, drop=40), 1)

    def test_the_drop_ends_the_carry(self):
        """Whether the operator opened the jaws or the object slipped out of them."""
        self.assertEqual(holding_label(40, grasp=10, lift=20, drop=40), 0)
        self.assertEqual(holding_label(45, grasp=10, lift=20, drop=40), 0)


class TestMineEpisode(unittest.TestCase):

    def _mine(self, offsets):
        return mine_episode(rows_for(offsets), 30.0, CALIBRATION,
                            approach_seconds=2.0, carry_seconds=0.5, rise_m=0.05)

    def test_a_close_approach_keeps_its_position_labels(self):
        result, dropped, blind = self._mine([0.02] * 90)
        self.assertIsNotNone(result)
        labelled = [r for r in result if r["target_uv"] is not None]
        self.assertGreater(len(labelled), 0)
        self.assertEqual(blind, 0)
        self.assertTrue(all(r["target_present"] == 1 for r in labelled))

    def test_a_target_outside_the_frame_loses_its_position_labels(self):
        """The frame is kept - it is a real picture with the object out of shot - and
        every position label is dropped rather than pointed somewhere invented."""
        # drifting in from one side, so the label crosses the margin partway through
        offsets = list(np.linspace(0.45, 0.02, 60)) + [0.02] * 30
        result, dropped, blind = self._mine(offsets)
        self.assertGreater(blind, 0)
        blind_rows = [r for r in result if r["seconds_to_grasp"] > 0
                      and r["target_uv"] is None]
        self.assertEqual(len(blind_rows), blind)
        for row in blind_rows:
            self.assertIsNone(row["target_uv"])
            self.assertIsNone(row["target_range_m"])
            self.assertIsNone(row["grasp_axis_rad"])

    def test_blind_rows_still_carry_the_labels_that_do_not_need_the_target(self):
        """Finger and holding are about the gripper, not about where the object is."""
        result, _, blind = self._mine(list(np.linspace(0.45, 0.02, 60)) + [0.02] * 30)
        blind_rows = [r for r in result if r["seconds_to_grasp"] > 0 and r["target_uv"] is None]
        self.assertGreater(len(blind_rows), 0)
        for row in blind_rows:
            self.assertIsNotNone(row["finger"])
            self.assertIsNotNone(row["holding"])

    def test_a_blind_row_does_not_claim_the_picture_is_empty(self):
        """Only that this object is out of shot. Something else graspable may well be in
        view, so present is masked rather than set to zero."""
        result, _, _ = self._mine(list(np.linspace(0.45, 0.02, 60)) + [0.02] * 30)
        blind_rows = [r for r in result if r["seconds_to_grasp"] > 0 and r["target_uv"] is None]
        self.assertGreater(len(blind_rows), 0)
        for row in blind_rows:
            self.assertIsNone(row["target_present"])

    def test_carry_frames_are_unchanged(self):
        """After the grasp the object rides in the jaws; those rows were already unlabelled
        for position and are not what the new masking is about.

        What they do keep is the labels that describe the gripper rather than where the
        object is. `holding` is one of those but has a window of its own, which
        TestHoldingWindow covers."""
        result, _, _ = self._mine([0.02] * 90)
        carried = [r for r in result if r["seconds_to_grasp"] < 0]
        self.assertGreater(len(carried), 0)
        for row in carried:
            self.assertIsNone(row["target_uv"])
            self.assertIsNone(row["target_present"])
            self.assertIsNotNone(row["finger"])

    def test_the_carry_is_masked_until_the_grip_takes_the_weight(self):
        """The grasp is not the lift. Frames between the two look exactly like held ones
        and the object really is between the jaws, but nothing there proves the grip, so
        they are masked rather than taught either way."""
        rows = rows_for([0.02] * 90)
        result, _, _ = self._mine([0.02] * 90)
        lift = find_lift(rows, 60, 30.0)

        carried = [r for r in result if r["seconds_to_grasp"] < 0]
        unproven = [r for r in carried if r["frame_index"] < lift]
        proven = [r for r in carried if r["frame_index"] >= lift]

        self.assertGreater(len(unproven), 0)
        self.assertGreater(len(proven), 0)
        self.assertTrue(all(r["holding"] is None for r in unproven))
        self.assertTrue(all(r["holding"] == 1 for r in proven))

    def test_everything_before_the_grasp_is_a_negative(self):
        """An object in view, in reach, and not in the hand is the case the head keeps
        getting wrong, so those frames are taught rather than masked."""
        result, _, _ = self._mine([0.02] * 90)
        approach = [r for r in result if r["seconds_to_grasp"] > 0]
        self.assertGreater(len(approach), 0)
        self.assertTrue(all(r["holding"] == 0 for r in approach))

    def test_an_episode_with_no_grasp_is_skipped(self):
        rows = rows_for([0.02] * 90)
        for row in rows:
            row["pressure"] = 0.0
        result, reason, blind = mine_episode(rows, 30.0, CALIBRATION, 2.0, 0.5, 0.05)
        self.assertIsNone(result)
        self.assertEqual(reason, "no_grasp")
        self.assertEqual(blind, 0)


class TestShardPrefix(unittest.TestCase):

    def test_the_miner_owns_only_its_own_shards(self):
        """Both producers write into one split, so a rerun of either must leave the
        other's files alone - emptying the directory used to delete the synthetic half
        without saying so."""
        self.assertEqual(ShardWriter.DEFAULT_PREFIX, "shard")
        self.assertNotEqual(ShardWriter.DEFAULT_PREFIX, "synth")


if __name__ == "__main__":
    unittest.main()
