"""Tests for the objectplates chroma keyer, mostly its range-scaled vignette.

The vignette exists because the chroma key only rejects green: at the top of a height
sweep the board no longer fills the frame, and its edge and the floor past it key as
foreground.
"""

import unittest

import numpy as np

from nf_robot.ml.visual_servoing.object_matte import (
    PRINCIPAL_NORM, apply_vignette, extract_cutout, vignette_axes)

SHAPE = (384, 684)


class TestVignetteGeometry(unittest.TestCase):

    def test_shrinks_with_range(self):
        """Same diameter on the floor, so the pixels it covers fall off as 1/range."""
        near = vignette_axes(SHAPE, 0.12)
        far = vignette_axes(SHAPE, 0.60)
        self.assertLess(far[0], near[0])
        # 0.60 / 0.12 = 5x the range, 1/5 the pixels
        self.assertAlmostEqual(near[0] / far[0], 5.0, places=6)

    def test_resolution_independent(self):
        """The capture stream is a different resolution from the calibration, same FOV."""
        small = vignette_axes((384, 684), 0.44)
        large = vignette_axes((1080, 1920), 0.44)
        self.assertAlmostEqual(small[0] / 684, large[0] / 1920, places=6)
        self.assertAlmostEqual(small[1] / 384, large[1] / 1080, places=6)

    def test_keeps_whole_frame_at_the_bottom_of_a_sweep(self):
        alpha = apply_vignette(np.ones(SHAPE, np.float32), 0.12)
        self.assertTrue((alpha > 0).all())

    def test_cuts_the_frame_edge_at_the_top_of_a_sweep(self):
        alpha = apply_vignette(np.ones(SHAPE, np.float32), 0.60)
        self.assertEqual(alpha[0, 0], 0.0)
        self.assertEqual(alpha[-1, -1], 0.0)
        # the grasp point is at the principal point and always survives
        cx, cy = int(PRINCIPAL_NORM[0] * SHAPE[1]), int(PRINCIPAL_NORM[1] * SHAPE[0])
        self.assertEqual(alpha[cy, cx], 1.0)

    def test_diameter_is_in_metres_on_the_floor(self):
        """A 0.5m object at 0.5m range spans the same pixels as the 0.5m vignette."""
        axes = vignette_axes(SHAPE, 0.5, diameter_m=0.5)
        doubled = vignette_axes(SHAPE, 0.5, diameter_m=1.0)
        self.assertAlmostEqual(doubled[0] / axes[0], 2.0, places=6)


class TestExtractCutout(unittest.TestCase):
    """A green frame with a grey blob at the principal point and another out at the edge,
    standing in for an object and an intruding board edge."""

    def _frame(self):
        rgb = np.zeros((*SHAPE, 3), np.uint8)
        rgb[:, :, 1] = 200  # green board everywhere
        cx, cy = int(PRINCIPAL_NORM[0] * SHAPE[1]), int(PRINCIPAL_NORM[1] * SHAPE[0])
        rgb[cy - 20:cy + 20, cx - 20:cx + 20] = 128  # the object
        rgb[0:40, 0:40] = 128  # the corner intruder
        return rgb, (cx, cy)

    def test_vignette_drops_the_corner_intruder(self):
        rgb, (cx, cy) = self._frame()
        rgba, grasp, _ = extract_cutout(rgb, range_m=0.60)
        # the crop is now around the object alone, not the two of them together
        self.assertLess(rgba.shape[1], cx)
        # the grasp point stays inside the cutout it was cropped against
        self.assertTrue(0 <= grasp[0] < rgba.shape[1] and 0 <= grasp[1] < rgba.shape[0])

    def test_without_a_range_nothing_is_cut(self):
        rgb, (cx, cy) = self._frame()
        rgba, _, _ = extract_cutout(rgb, range_m=None)
        # the crop has to reach from the corner to the object to hold both
        self.assertGreater(rgba.shape[1], cx)

    def test_keys_to_nothing_when_the_object_is_outside_the_vignette(self):
        rgb = np.zeros((*SHAPE, 3), np.uint8)
        rgb[:, :, 1] = 200
        rgb[0:40, 0:40] = 128  # only the intruder, no object
        self.assertIsNone(extract_cutout(rgb, range_m=0.60))


if __name__ == "__main__":
    unittest.main()
