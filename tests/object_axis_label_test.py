"""Tests for the grasp axis label on object cutouts.

The label is the only thing in the pipeline that says which way an object is turned, and
it is measured rather than stored: the capture records where the wrist was for each frame,
the run records the angle the operator called ideal, and the difference is the label. When
that measurement is missing the whole thing still runs and produces a dataset whose axis
head trains on a constant - which is what these tests exist to catch.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nf_robot.ml.visual_servoing.object_matte import (
    PRINCIPAL_NORM, extract_run, wrist_offset_deg)
from nf_robot.ml.visual_servoing.plates import PlateWriter

SHAPE = (240, 320)


class TestWristOffset(unittest.TestCase):

    def test_offset_from_the_ideal_grasping_angle(self):
        self.assertAlmostEqual(wrist_offset_deg(120.0, 90.0), 30.0)
        self.assertAlmostEqual(wrist_offset_deg(60.0, 90.0), -30.0)
        self.assertAlmostEqual(wrist_offset_deg(90.0, 90.0), 0.0)

    def test_folds_across_the_wrap(self):
        """A sweep crosses 360; the raw difference would jump 359 degrees there."""
        self.assertAlmostEqual(wrist_offset_deg(370.0, 350.0), 20.0)
        self.assertAlmostEqual(wrist_offset_deg(10.0, 350.0), 20.0)
        self.assertAlmostEqual(wrist_offset_deg(350.0, 10.0), -20.0)

    def test_unknown_stays_unknown(self):
        """None, not zero: zero is a real answer that means 'ideally aligned'."""
        self.assertIsNone(wrist_offset_deg(None, 90.0))
        self.assertIsNone(wrist_offset_deg(90.0, None))


def write_run(plate_dir, wrist_angles, **run_attrs):
    """A small objectplates run: a grey blob on green, at a spread of wrist angles."""
    writer = PlateWriter(plate_dir, "objectplates", notes="test")
    for wrist in wrist_angles:
        rgb = np.zeros((*SHAPE, 3), np.uint8)
        rgb[:, :, 1] = 200
        cx = int(PRINCIPAL_NORM[0] * SHAPE[1])
        cy = int(PRINCIPAL_NORM[1] * SHAPE[0])
        rgb[cy - 25:cy + 25, cx - 25:cx + 25] = 120
        writer.add(rgb, wrist_angle=wrist, laser_rangefinder=0.3, label="sock")
    writer.close(**run_attrs)
    return writer.run_id


class TestExtractRunLabelsTheAxis(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_offsets_come_from_the_wrist_telemetry(self):
        angles = [90.0, 120.0, 180.0, 300.0]
        run_id = write_run(self.dir, angles, grasp_axis_wrist_angle=90.0, label="sock")
        entries = extract_run(self.dir, run_id, self.dir / "objects")
        self.assertEqual(len(entries), len(angles))
        got = [e["wrist_offset_deg"] for e in entries]
        self.assertEqual([round(v) for v in got], [0, 30, 90, -150])

    def test_a_capture_whose_wrist_swept_produces_a_spread(self):
        """The bug this file exists for: every cutout came out at 0."""
        run_id = write_run(self.dir, [10.0, 100.0, 190.0, 280.0],
                           grasp_axis_wrist_angle=10.0)
        entries = extract_run(self.dir, run_id, self.dir / "objects")
        offsets = {round(e["wrist_offset_deg"]) for e in entries}
        self.assertGreater(len(offsets), 1)
        self.assertNotEqual(offsets, {0})

    def test_start_wrist_angle_is_accepted_as_the_zero(self):
        """What the sweep records it as, for runs written before the name settled."""
        run_id = write_run(self.dir, [45.0, 75.0], start_wrist_angle=45.0)
        entries = extract_run(self.dir, run_id, self.dir / "objects")
        self.assertEqual([round(e["wrist_offset_deg"]) for e in entries], [0, 30])

    def test_a_run_with_no_zero_recorded_labels_nothing(self):
        """Unlabelled rather than quietly zero, so synth_frames is not handed a
        fabricated axis for every frame."""
        run_id = write_run(self.dir, [45.0, 75.0])
        entries = extract_run(self.dir, run_id, self.dir / "objects")
        self.assertTrue(all(e["wrist_offset_deg"] is None for e in entries))


if __name__ == "__main__":
    unittest.main()
