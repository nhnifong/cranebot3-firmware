import unittest

import numpy as np

import nf_robot.common.definitions as model_constants
from nf_robot.common.config_loader import create_default_config
from nf_robot.generated.nf import common, config as nf_config
from nf_robot.host import swing


class TestPoleTypes(unittest.TestCase):
    def test_every_pole_type_has_a_geometry(self):
        for pole_type in common.PoleType:
            geom = model_constants.POLE_GEOMETRY[pole_type]
            self.assertGreater(geom.swing_length, 0)
            self.assertGreater(geom.gantry_to_gripper, 0)
            self.assertIsNotNone(geom.gantry_april)

    def test_carbon_pole_swings_at_the_shorter_length(self):
        self.assertLess(model_constants.POLE_GEOMETRY[common.PoleType.CARBON400].swing_length,
                        model_constants.POLE_GEOMETRY[common.PoleType.ABS500].swing_length)

    def test_the_two_poles_carry_different_markers(self):
        self.assertIsNot(model_constants.POLE_GEOMETRY[common.PoleType.CARBON400].gantry_april,
                         model_constants.POLE_GEOMETRY[common.PoleType.ABS500].gantry_april)

    def test_a_shorter_pole_swings_faster(self):
        abs500 = swing.Pendulum(model_constants.pole_length_abs500)
        carbon = swing.Pendulum(model_constants.pole_length_carbon400)
        self.assertLess(carbon.period, abs500.period)

    def test_built_from_the_configured_pole(self):
        cfg = nf_config.StringmanPilotConfig(
            gripper=nf_config.Gripper(pole_type=common.PoleType.CARBON400))
        self.assertEqual(swing.pendulum_for(cfg).length, model_constants.pole_length_carbon400)

    def test_a_new_default_config_has_the_current_pole(self):
        cfg = create_default_config()
        self.assertEqual(cfg.gripper.pole_type, common.PoleType.CARBON270)
        self.assertEqual(swing.pendulum_for(cfg).length, model_constants.pole_length_carbon270)

    def test_an_unset_pole_type_falls_back_to_the_older_pole(self):
        """Matches the length config_loader backfills, so both paths agree."""
        cfg = nf_config.StringmanPilotConfig(gripper=nf_config.Gripper())
        self.assertEqual(swing.pendulum_for(cfg).length, model_constants.pole_length_abs500)


class TestPendulumModel(unittest.TestCase):
    def setUp(self):
        self.p = swing.Pendulum(model_constants.pole_length_abs500)

    def test_projecting_by_zero_is_the_model_itself(self):
        sm = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(self.p.project(sm, 0.0), sm)

    def test_projecting_by_a_full_period_returns_to_the_same_phase(self):
        sm = np.array([[1.0, 2.0], [-0.5, 0.25]])
        np.testing.assert_allclose(self.p.project(sm, self.p.period), sm, atol=1e-12)

    def test_projecting_by_half_a_period_inverts_the_state(self):
        sm = np.array([[1.0, 2.0], [-0.5, 0.25]])
        np.testing.assert_allclose(self.p.project(sm, self.p.half_period), -sm, atol=1e-12)

    def test_amplitude_is_phase_independent(self):
        sm = np.array([[0.3, -0.7], [0.1, 0.2]])
        amp = self.p.amplitude(sm)
        for dt in np.linspace(0, self.p.period, 17):
            self.assertAlmostEqual(self.p.amplitude(self.p.project(sm, dt)), amp)

    def test_amplitude_without_an_imu(self):
        self.assertEqual(self.p.amplitude(None), 0.0)
        self.assertEqual(self.p.amplitude(np.zeros((2, 2))), 0.0)

    def test_tilt_peaks_at_the_amplitude(self):
        sm = np.array([[0.4, 0.0], [0.0, 0.0]])
        tilts = [self.p.tilt(sm, dt)[0] for dt in np.linspace(0, self.p.period, 400)]
        self.assertAlmostEqual(max(tilts), self.p.amplitude(sm), places=4)
        self.assertEqual(self.p.tilt(sm)[2], 0)

    def test_cancel_velocity_opposes_the_swing(self):
        # column 1 is the quarter-ahead phase, which the correction opposes
        sm = np.array([[0.0, 0.5], [0.0, -0.25]])
        vel = self.p.cancel_velocity(sm, 0.0)
        self.assertLess(vel[0], 0)
        self.assertGreater(vel[1], 0)

    def test_cancel_velocity_leads_by_the_latency(self):
        """Asking for the correction dt ahead gives what a caller dt later would get now."""
        sm = np.array([[0.3, -0.2], [0.1, 0.4]])
        np.testing.assert_allclose(self.p.cancel_velocity(sm, 0.25),
                                   self.p.cancel_velocity(self.p.project(sm, 0.25), 0.0))

    def test_the_wrong_pole_lands_at_the_wrong_phase(self):
        """Why the length is configured at all: the two poles diverge over a latency."""
        sm = np.array([[0.3, -0.2], [0.1, 0.4]])
        carbon = swing.Pendulum(model_constants.pole_length_carbon400)
        self.assertFalse(np.allclose(self.p.cancel_velocity(sm, 0.3),
                                     carbon.cancel_velocity(sm, 0.3)))

    def test_no_swing_needs_no_correction(self):
        np.testing.assert_allclose(self.p.cancel_velocity(np.zeros((2, 2)), 0.1), np.zeros(2))


class TestCenteringIntegrator(unittest.TestCase):
    def test_offset_accumulates_commanded_motion(self):
        vel, offset = swing.integrate_centering(np.array([1.0, 0.0]), np.zeros(2), 0.1)
        np.testing.assert_allclose(vel, [1.0, 0.0])
        np.testing.assert_allclose(offset, [0.1, 0.0])

    def test_offset_pulls_the_velocity_back(self):
        vel, _ = swing.integrate_centering(np.zeros(2), np.array([0.5, 0.0]), 0.1)
        self.assertLess(vel[0], 0)

    def test_offset_converges_back_to_zero(self):
        offset = np.array([0.5, 0.0])
        for _ in range(2000):
            _, offset = swing.integrate_centering(np.zeros(2), offset, 0.01)
        self.assertLess(abs(offset[0]), 1e-3)

    def test_a_stalled_loop_does_not_wreck_the_integrator(self):
        for dt in (5.0, -1.0):
            vel, offset = swing.integrate_centering(np.array([1.0, 0.0]), np.zeros(2), dt)
            np.testing.assert_allclose(vel, [1.0, 0.0])
            np.testing.assert_allclose(offset, np.zeros(2))


class TestAltitudeHold(unittest.TestCase):
    def test_drives_toward_the_start_altitude(self):
        self.assertGreater(swing.altitude_hold_velocity(0.01), 0)
        self.assertLess(swing.altitude_hold_velocity(-0.01), 0)
        self.assertEqual(swing.altitude_hold_velocity(0.0), 0.0)

    def test_capped_both_ways(self):
        self.assertEqual(swing.altitude_hold_velocity(10.0), swing.ALTITUDE_HOLD_MAX_MPS)
        self.assertEqual(swing.altitude_hold_velocity(-10.0), -swing.ALTITUDE_HOLD_MAX_MPS)


class TestFineCandidates(unittest.TestCase):
    def test_spans_the_coarse_best(self):
        cands = swing.fine_candidates(0.3)
        self.assertAlmostEqual(min(cands), 0.3 - swing.FINE_HALF_WIDTH)
        self.assertAlmostEqual(max(cands), 0.3 + swing.FINE_HALF_WIDTH)
        self.assertEqual(len(cands), swing.FINE_COUNT)
        self.assertEqual(cands, sorted(cands))

    def test_clipped_to_the_sane_range_without_repeating_a_trial(self):
        cands = swing.fine_candidates(0.0)
        self.assertGreaterEqual(min(cands), swing.FINE_CLIP[0])
        self.assertLessEqual(max(cands), swing.FINE_CLIP[1])
        # the half below zero all clip to the same value, which is only worth measuring once
        self.assertEqual(len(cands), len(set(cands)))
        self.assertLess(len(cands), swing.FINE_COUNT)

    def test_drops_latencies_the_coarse_pass_already_measured(self):
        cands = swing.fine_candidates(0.3, [0.3, 0.0, 0.6])
        self.assertNotIn(0.3, cands)
        # 0.0 and 0.6 are outside this spread, so nothing else is lost to them
        self.assertEqual(len(cands), swing.FINE_COUNT - 1)
        self.assertEqual(cands, swing.fine_candidates(0.3, [0.3]))


class TestTrialResidual(unittest.TestCase):
    def setUp(self):
        self.p = swing.Pendulum(model_constants.pole_length_abs500)

    def _settle(self, amps):
        """Timestamps covering the measured window, one sample per loop iteration."""
        return list(np.linspace(0, 6.4 * self.p.period, len(amps))), amps

    def test_averages_only_the_final_periods(self):
        # loud at the start, quiet once cancellation takes hold: only the tail counts
        n = 200
        amps = [0.3] * (n // 2) + [0.02] * (n // 2)
        ts, amps = self._settle(amps)
        self.assertAlmostEqual(self.p.trial_residual(ts, amps, None), 0.02, places=6)

    def test_a_steady_swing_reads_as_its_amplitude(self):
        ts, amps = self._settle([0.1] * 100)
        self.assertAlmostEqual(self.p.trial_residual(ts, amps, None), 0.1)

    def test_pumping_past_the_cap_scores_as_definitively_bad(self):
        ts, amps = self._settle([0.05] * 100)
        self.assertEqual(self.p.trial_residual(ts, amps, 'amp_cap'), swing.SAFETY_AMP_RAD)

    def test_trials_that_never_settled_are_excluded(self):
        ts, amps = self._settle([0.05] * 100)
        self.assertIsNone(self.p.trial_residual(ts, amps, 'drift'))
        self.assertIsNone(self.p.trial_residual(ts, amps, 'tension'))

    def test_too_few_samples_to_believe(self):
        ts, amps = self._settle([0.05] * (swing.MIN_SAMPLES - 1))
        self.assertIsNone(self.p.trial_residual(ts, amps, None))


class TestSelectMinResidual(unittest.TestCase):
    def test_picks_the_center_of_the_damped_range(self):
        # 0.2 through 0.4 all bottom out at the measurement floor
        results = [(0.0, 0.30), (0.1, 0.12), (0.2, 0.021), (0.3, 0.019),
                   (0.4, 0.022), (0.5, 0.15), (0.6, 0.33)]
        self.assertAlmostEqual(swing.select_min_residual(results), 0.3)

    def test_a_single_clear_winner(self):
        results = [(0.0, 0.30), (0.1, 0.20), (0.2, 0.02), (0.3, 0.20)]
        self.assertAlmostEqual(swing.select_min_residual(results), 0.2)

    def test_one_bad_settle_does_not_reject_a_good_latency(self):
        """A latency measured twice keeps its better reading."""
        results = [(0.0, 0.30), (0.2, 0.02), (0.2, 0.28), (0.3, 0.30)]
        self.assertAlmostEqual(swing.select_min_residual(results), 0.2)

    def test_floor_range_does_not_run_past_a_gap(self):
        # 0.5 also sits at the floor but 0.3 does not, so the range stops before it
        results = [(0.1, 0.02), (0.2, 0.021), (0.3, 0.30), (0.5, 0.02)]
        self.assertAlmostEqual(swing.select_min_residual(results), 0.15)


class TestMeasurePendulum(unittest.TestCase):
    """Enough to trust the number this prints: a synthetic swing carrying what a real
    recording has - gyro bias, jittered sample times, decay, and a second axis ringing
    along with the first - has to come back out as the length that went in."""

    def _recording(self, length, duration=20.0, seed=0):
        rng = np.random.default_rng(seed)
        n = int(duration * 100)
        ts = np.sort(np.arange(n) / 100 + rng.normal(0, 0.002, n)) + 1_700_000_000.0
        omega = np.sqrt(swing.GRAVITY / length)
        envelope = 0.5 * np.exp(-(ts - ts[0]) / 12.0)
        gx = envelope * np.sin(omega * (ts - ts[0])) + 0.04 + rng.normal(0, 0.01, n)
        gy = 0.25 * envelope * np.sin(omega * (ts - ts[0]) + 0.6) - 0.02 + rng.normal(0, 0.01, n)
        return np.stack([ts, gx, gy], axis=1)

    def test_recovers_the_length_it_was_given(self):
        # 2mm is the accuracy worth running this for: well inside the 18mm between the two
        # poles, so a measurement can say which one is fitted.
        for length in (model_constants.pole_length_carbon400,
                       model_constants.pole_length_abs500):
            with self.subTest(length=length):
                freq, measured = swing.measure_pendulum(self._recording(length))
                self.assertAlmostEqual(measured, length, delta=0.002)
                self.assertAlmostEqual(swing.length_for_frequency(freq), measured)

    def test_says_nothing_rather_than_guessing(self):
        self.assertEqual(swing.measure_pendulum(np.zeros((0, 3))), (None, None))
        # too few swings to time
        self.assertEqual(swing.measure_pendulum(self._recording(0.4526, duration=4.0)),
                         (None, None))


if __name__ == '__main__':
    unittest.main()
