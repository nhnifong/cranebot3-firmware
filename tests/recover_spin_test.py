"""Tests for recovering the `spin` state field from what a recording kept instead.

Built by running the recorder's own arithmetic forwards - bearing = room_angle - spin,
from stringman_lerobot's state builder - so a test failing here means the recovery no
longer inverts what the robot actually wrote.
"""

import math
import unittest

import numpy as np

from nf_robot.ml.visual_servoing.recover_spin import (
    MIN_HORIZONTAL_M, TARGET_NAMES, episode_constant, episode_spin, spin_from_bearings)

STATE_NAMES = (["gripper_pos_x", "gripper_pos_y", "wrist_angle"]
               + [f"{n}_{k}" for n in TARGET_NAMES for k in ("bearing", "distance")])
INDEX = {n: i for i, n in enumerate(STATE_NAMES)}


def make_episode(positions, wrist_deg, constant_deg, target=(0.0, 0.0), detected=()):
    """Frames as the recorder would have written them, for a known spin.

    `constant_deg` is the calibration offset between spin and the wrist, `target` is where
    the named position sat - the origin unless a recording actually detected something,
    which `detected` names.
    """
    rows = []
    for (x, y), wrist in zip(positions, wrist_deg):
        spin = math.radians(wrist) + math.radians(constant_deg)
        state = [0.0] * len(STATE_NAMES)
        state[INDEX["gripper_pos_x"]] = x
        state[INDEX["gripper_pos_y"]] = y
        state[INDEX["wrist_angle"]] = wrist
        for name in TARGET_NAMES:
            here = (0.9, -1.4) if name in detected else target
            dx, dy = here[0] - x, here[1] - y
            room_angle = math.atan2(dx, dy)
            state[INDEX[f"{name}_bearing"]] = (room_angle - spin + math.pi) % (2 * math.pi) - math.pi
            state[INDEX[f"{name}_distance"]] = math.hypot(dx, dy)
        rows.append(state)
    return np.array(rows)


def ring(count=60, radius=1.2):
    """Gripper positions well clear of the origin, where the bearing is well conditioned."""
    return [(radius * math.cos(t), radius * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, count, endpoint=False)]


class TestSpinFromBearings(unittest.TestCase):

    def test_recovers_the_spin_the_recorder_started_from(self):
        wrist = list(np.linspace(0, 720, 60))
        state = make_episode(ring(), wrist, constant_deg=-111.4)
        spins, usable = spin_from_bearings(state, INDEX)
        self.assertTrue(usable.all())
        expected = np.radians(np.array(wrist)) + math.radians(-111.4)
        for row in spins:
            diff = (row - expected + math.pi) % (2 * math.pi) - math.pi
            np.testing.assert_allclose(diff, 0, atol=1e-9)

    def test_ignores_a_column_that_saw_something_real(self):
        """A detected target is not at the origin, so its bearing says nothing about spin
        without knowing where it was - and is left out rather than averaged in."""
        state = make_episode(ring(), [0.0] * 60, constant_deg=0.0, detected={"gamepad"})
        _, usable = spin_from_bearings(state, INDEX)
        gamepad = TARGET_NAMES.index("gamepad")
        self.assertFalse(usable[gamepad].any())
        self.assertTrue(usable[TARGET_NAMES.index("hamper")].all())

    def test_drops_frames_too_near_the_origin(self):
        """The direction back to a point you are standing on is noise."""
        close = [(0.01, -0.01)] * 40
        state = make_episode(close, [0.0] * 40, constant_deg=0.0)
        _, usable = spin_from_bearings(state, INDEX)
        self.assertFalse(usable.any())


class TestEpisodeConstant(unittest.TestCase):

    def test_measures_the_calibration_offset(self):
        state = make_episode(ring(), list(np.linspace(0, 360, 60)), constant_deg=10.8)
        constant, spread, frames = episode_constant(state, INDEX)
        self.assertAlmostEqual(math.degrees(constant), 10.8, places=6)
        self.assertLess(spread, 1e-6)
        self.assertGreater(frames, 0)

    def test_fills_every_frame_including_the_ill_conditioned_ones(self):
        """The wrist is exact everywhere; the bearing is only needed to find the offset."""
        positions = ring(50) + [(0.02, 0.0)] * 10          # ten frames sat on the origin
        wrist = list(np.linspace(0, 500, 60))
        state = make_episode(positions, wrist, constant_deg=-71.1)
        spin, diagnostics = episode_spin(state, INDEX)
        self.assertIsNotNone(spin)
        self.assertEqual(len(spin), 60)
        expected = np.radians(np.array(wrist)) + math.radians(-71.1)
        np.testing.assert_allclose(spin, expected, atol=1e-9)
        self.assertAlmostEqual(diagnostics["constant_deg"], -71.1, places=6)

    def test_refuses_an_episode_where_spin_is_not_the_wrist_plus_a_constant(self):
        """If that does not hold, the frames without a bearing cannot be filled, and a
        plausible number for them is worse than none."""
        state = make_episode(ring(), [0.0] * 60, constant_deg=0.0)
        # scramble the bearings so no single offset explains them
        for i, name in enumerate(TARGET_NAMES):
            state[:, INDEX[f"{name}_bearing"]] += np.linspace(0, 1.5, len(state))
        spin, diagnostics = episode_spin(state, INDEX)
        self.assertIsNone(spin)
        self.assertIn("drift", diagnostics["reason"])

    def test_refuses_an_episode_with_almost_no_usable_bearing(self):
        state = make_episode([(0.01, 0.01)] * 100, [0.0] * 100, constant_deg=0.0)
        spin, diagnostics = episode_spin(state, INDEX)
        self.assertIsNone(spin)
        self.assertIn("too few frames", diagnostics["reason"])


class TestConventionMatchesTheRecorder(unittest.TestCase):

    def test_bearing_is_room_angle_minus_spin_with_x_first(self):
        """stringman_lerobot builds room_angle as arctan2(delta_x, delta_y) - a compass
        bearing off +Y, not the usual atan2(y, x). Getting that backwards would mirror
        every recovered heading and still look self-consistent."""
        state = make_episode([(2.0, 0.0)], [0.0], constant_deg=0.0)
        # gripper at +X, target at the origin: the direction back to it is -X, which as a
        # compass bearing off +Y is -90 degrees
        bearing = state[0, INDEX["hamper_bearing"]]
        self.assertAlmostEqual(math.degrees(bearing), -90.0, places=6)
        spins, _ = spin_from_bearings(state, INDEX)
        self.assertAlmostEqual(spins[0, 0], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
