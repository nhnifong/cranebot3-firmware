"""Tests for the fingerplates chroma keyer.

The capture is a wrist turn per finger angle over the green backdrop, so the fingers are
the only ungreen thing that stays on the same pixels; everything the tests here build is
that situation, plus the ways it goes wrong.
"""

import unittest
from unittest.mock import patch

import numpy as np

from nf_robot.ml.visual_servoing.finger_matte import (
    build_matte, clean_mask, green_fraction, group_by_finger_angle)

SHAPE = (120, 200)


def green_frame():
    frame = np.zeros((*SHAPE, 3), np.float32)
    frame[:, :, 1] = 200.0
    return frame


def stack_with_fingers(n=8, intruder=True):
    """A dark finger down the left edge in every frame, plus something ungreen that moves.

    The intruder stands in for whatever rotated past under the gripper - a shoe, the edge
    of the sheet - and is in a different place in each frame, as a wrist turn makes it.
    """
    frames = []
    for i in range(n):
        frame = green_frame()
        frame[40:80, 0:30] = 40.0  # the finger, touching the frame edge
        if intruder:
            x = 60 + i * 12
            frame[10:30, x:x + 18] = 40.0
        frames.append(frame)
    return np.stack(frames)


class TestBuildMatte(unittest.TestCase):

    def test_keeps_the_finger(self):
        rgba, diagnostics = build_matte(stack_with_fingers())
        alpha = rgba[:, :, 3]
        self.assertEqual(alpha[60, 10], 255)
        self.assertEqual(diagnostics["components"], 1)

    def test_median_over_the_turn_outvotes_a_moving_intruder(self):
        rgba, _ = build_matte(stack_with_fingers())
        alpha = rgba[:, :, 3]
        # the intruder is ungreen in one frame out of eight at any given pixel
        self.assertTrue((alpha[10:30, 60:] == 0).all())

    def test_an_intruder_that_never_moves_is_kept(self):
        """The median cannot tell a bolted-on part from something that sat still: the
        border test is what rejects it, and only when it floats clear of the edge."""
        stack = stack_with_fingers(intruder=False)
        stack[:, 10:30, 60:78] = 40.0
        rgba, _ = build_matte(stack, border_only=False)
        self.assertEqual(rgba[:, :, 3][20, 70], 255)
        rgba, diagnostics = build_matte(stack, border_only=True)
        self.assertEqual(rgba[:, :, 3][20, 70], 0)
        self.assertEqual(diagnostics["components"], 1)

    def test_green_fraction_reports_a_missing_backdrop(self):
        stack = stack_with_fingers(intruder=False)
        _, on_green = build_matte(stack)
        self.assertGreater(on_green["green_fraction"], 0.8)

        grey = np.full_like(stack, 90.0)
        grey[:, 40:80, 0:30] = 40.0
        _, off_green = build_matte(grey)
        self.assertLess(off_green["green_fraction"], 0.01)

    def test_spill_is_pulled_out_of_the_kept_colour(self):
        """A finger lit by the backdrop carries a green cast that must not composite."""
        stack = stack_with_fingers(intruder=False)
        stack[:, 40:80, 0:30] = (40.0, 120.0, 40.0)
        rgba, _ = build_matte(stack)
        red, green, blue = rgba[60, 10, :3]
        self.assertLessEqual(int(green), (int(red) + int(blue)) // 2 + 1)


class TestCleanMask(unittest.TestCase):

    def test_encloses_holes_without_flooding_from_a_corner(self):
        """A finger reaching the corner used to make a corner flood a no-op, which read
        the whole background as one hole."""
        raw = np.zeros(SHAPE, bool)
        raw[0:60, 0:60] = True   # reaches two corners
        raw[20:30, 20:30] = False  # a highlight inside it
        filled, kept, components = clean_mask(raw, border_only=True)
        self.assertTrue(filled[25, 25])
        self.assertFalse(kept[25, 25])
        self.assertFalse(filled[100, 150])
        self.assertEqual(components, 1)

    def test_drops_speckle(self):
        raw = np.zeros(SHAPE, bool)
        raw[40:80, 0:30] = True
        raw[100:102, 100:102] = True
        filled, _, components = clean_mask(raw)
        self.assertEqual(components, 1)
        self.assertFalse(filled[100, 100])


class TestGrouping(unittest.TestCase):

    def test_groups_on_the_commanded_angle(self):
        """The measured angle wanders by a fraction of a degree and would split every
        group into singletons."""
        rows = [
            {"attrs": {"commanded_finger_angle": -20}, "finger_angle": -20.4, "image": "a"},
            {"attrs": {"commanded_finger_angle": -20}, "finger_angle": -19.6, "image": "b"},
            {"attrs": {"commanded_finger_angle": 0}, "finger_angle": 0.1, "image": "c"},
        ]
        with patch("nf_robot.ml.visual_servoing.finger_matte.iter_run", return_value=rows):
            groups = group_by_finger_angle("plates", "run")
        self.assertEqual(groups, {-20.0: ["a", "b"], 0.0: ["c"]})


class TestGreenFraction(unittest.TestCase):

    def test_counts_only_decisive_green(self):
        frame = green_frame()
        self.assertAlmostEqual(green_fraction(frame), 1.0)
        frame[:, :100] = 90.0  # grey
        self.assertAlmostEqual(green_fraction(frame), 0.5)


if __name__ == "__main__":
    unittest.main()
