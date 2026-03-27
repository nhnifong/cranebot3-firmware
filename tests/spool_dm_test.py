import unittest
from unittest.mock import MagicMock, patch
import math
import numpy as np
import time

# We mock these out so the tests can run without the actual hardware or modules present
import sys
sys.modules['damiao_motor'] = MagicMock()

from damiao_motor import DaMiaoMotor
from nf_robot.robot.spools import SpiralCalculator
from nf_robot.robot.spool_dm import DamiaoSpoolController, default_conf_dm


class TestDamiaoSpoolController(unittest.TestCase):

    def setUp(self):
        """Set up a fresh controller and mock dependencies before each test."""
        self.mock_motor = MagicMock()
        
        # Patch the SpiralCalculator before the controller initializes it
        self.sc_patcher = patch('nf_robot.robot.spool_dm.SpiralCalculator')
        self.MockSpiralCalculator = self.sc_patcher.start()
        self.mock_sc_instance = self.MockSpiralCalculator.return_value
        self.mock_sc_instance.get_unspool_rate.return_value = 0.05
        self.mock_sc_instance.get_unspooled_length.return_value = 3.0
        
        self.config = {}
        self.controller = DamiaoSpoolController(
            motor=self.mock_motor,
            empty_diameter=50,
            full_diameter=100,
            full_length=20,
            config=self.config,
            direction=1
        )

    def tearDown(self):
        self.sc_patcher.stop()

    def test_initialization(self):
        """Verify the controller initializes config and spiral calculator properly."""
        self.MockSpiralCalculator.assert_called_once_with(50, 100, 20, 1, -1)
        self.assertEqual(self.controller.direction, 1)
        self.assertAlmostEqual(self.controller.meters_per_rev, 0.05)
        # Check that default config is merged
        self.assertEqual(self.config['MAX_SAFE_LINE_SPEED'], default_conf_dm['MAX_SAFE_LINE_SPEED'])

    def test_setReferenceLength_slack(self):
        """Verify reference length is ignored if line is slack (torque_err > 0)."""
        self.controller.torque_err = 0.5  # Slack
        self.controller.setReferenceLength(5.0)
        self.mock_sc_instance.set_zero_angle.assert_not_called()

    def test_setReferenceLength_out_of_bounds(self):
        """Verify reference length is ignored if outside [0, 20]."""
        self.controller.torque_err = -0.5  # Taught
        
        self.controller.setReferenceLength(-1.0)
        self.mock_sc_instance.set_zero_angle.assert_not_called()

        self.controller.setReferenceLength(25.0)
        self.mock_sc_instance.set_zero_angle.assert_not_called()

    def test_setReferenceLength_valid(self):
        """Verify valid reference length updates the SpiralCalculator and syncs target."""
        self.controller.torque_err = -0.5  # Taught
        self.controller.target_length = 3.0
        self.controller.last_angle = 10.0
        
        self.mock_sc_instance.calc_za_from_length.return_value = -2.0
        
        self.controller.setReferenceLength(5.0)
        
        self.mock_sc_instance.calc_za_from_length.assert_called_once_with(5.0, 10.0)
        self.mock_sc_instance.set_zero_angle.assert_called_once_with(-2.0)
        self.assertEqual(self.controller.target_length, 5.0)

    def test_setAimSpeed(self):
        """Verify setting aim speed clamps values and clears target length."""
        max_speed = self.config['MAX_SAFE_LINE_SPEED']
        
        # Valid speed
        self.controller.setAimSpeed(1.0)
        self.assertEqual(self.controller.aim_line_speed, 1.0)
        self.assertIsNone(self.controller.target_length)
        
        # Upper clamp
        self.controller.setAimSpeed(max_speed + 2.0)
        self.assertEqual(self.controller.aim_line_speed, max_speed)
        
        # Lower clamp
        self.controller.setAimSpeed(-max_speed - 2.0)
        self.assertEqual(self.controller.aim_line_speed, -max_speed)

    def test_jog(self):
        """Verify jogging properly increments the target length."""
        self.controller.last_length = 2.0
        
        # When target is None, it should build from last_length
        self.controller.jog(0.5)
        self.assertEqual(self.controller.target_length, 2.5)
        
        # When target exists, it adds to existing target
        self.controller.jog(0.5)
        self.assertEqual(self.controller.target_length, 3.0)

    def test_popMeasurements(self):
        """Verify popping returns the buffer and clears it."""
        self.controller.record = [1, 2, 3]
        popped = self.controller.popMeasurements()
        self.assertEqual(popped, [1, 2, 3])
        self.assertEqual(self.controller.record, [])

    def test_fastStop(self):
        """Verify fast stop flag cleanly ends the spool loop constraint."""
        self.assertTrue(self.controller.run_spool_loop)
        self.controller.fastStop()
        self.assertFalse(self.controller.run_spool_loop)

    def test_update_absolute_angle(self):
        """Verify motor wrapping/rollover detection maintains continuous revolutions."""
        # Initial pos
        ans = self.controller._update_absolute_angle(0.0)
        self.assertEqual(ans, 0.0)
        
        # Small positive move (no wrap)
        ans = self.controller._update_absolute_angle(math.pi)
        self.assertEqual(ans, 0.5)
        
        # Negative to Positive Wrap (e.g., from +12 to -12) -> +25 range diff
        self.controller.last_raw_pos = 12.0
        self.controller.rev_offset = 0.0
        ans = self.controller._update_absolute_angle(-12.0) 
        # Diff = -24 (which is < -12.5). rev_offset += 25 / 2pi
        expected = (25.0 / (2 * math.pi)) + (-12.0 / (2 * math.pi))
        self.assertAlmostEqual(ans, expected)

        # Positive to Negative Wrap (e.g., from -12 to +12) -> -25 range diff
        self.controller.last_raw_pos = -12.0
        self.controller.rev_offset = 0.0
        ans = self.controller._update_absolute_angle(12.0)
        # Diff = +24 (which is > 12.5). rev_offset -= 25 / 2pi
        expected = (-25.0 / (2 * math.pi)) + (12.0 / (2 * math.pi))
        self.assertAlmostEqual(ans, expected)

    @patch('time.sleep')
    @patch('time.time')
    def test_trackingLoop_basic_operations(self, mock_time, mock_sleep):
        """Verify the tracking loop correctly queries the motor, filters torque, and disables on exit."""
        # 3 calls: start_time, loop_start, end_loop_time
        # loop duration = 1.01 - 1.0 = 0.01s (fast enough to trigger sleep and stop the loop)
        mock_time.side_effect = [1.0, 1.0, 1.01]
        
        # Cause the loop to exit after 1 iteration via the sleep mock
        def stop_loop(*args):
            self.controller.run_spool_loop = False
        mock_sleep.side_effect = stop_loop
        
        # Setup mock motor state
        self.mock_motor.get_states.return_value = {'pos': 0.0, 'vel': 2.0, 'torq': -0.1}
        self.mock_sc_instance.get_unspooled_length.return_value = 3.0
        
        self.controller.trackingLoop()
        
        # Verify setups and teardowns
        self.mock_motor.enable.assert_called_once()
        self.mock_motor.ensure_control_mode.assert_called_with("VEL")
        self.mock_motor.disable.assert_called_once()
        
        # Verify that record tracking gathered 1 record tuple
        self.assertEqual(len(self.controller.record), 1)
        # Record format: (start_time, last_length, line_speed, tension)
        self.assertEqual(self.controller.record[0][1], 3.0)

    @patch('time.sleep')
    @patch('time.time')
    def test_trackingLoop_jog_logic_and_deadband(self, mock_time, mock_sleep):
        """Verify that position targeting (jogging) sets the correct cruise speed and respects deadband."""
        mock_time.side_effect = [1.0, 1.0, 1.01]
        
        def stop_loop(*args):
            self.controller.run_spool_loop = False
        mock_sleep.side_effect = stop_loop
        
        self.mock_motor.get_states.return_value = {'pos': 0.0, 'vel': 0.0, 'torq': -0.1}
        
        # Set a target further away
        self.controller.target_length = 5.0
        self.mock_sc_instance.get_unspooled_length.return_value = 4.0 # dist_err = 1.0 > 0
        
        self.controller.trackingLoop()
        # Should command positive cruise speed
        self.assertEqual(self.controller.aim_line_speed, self.config['CRUISE_SPEED'])

        # Reset loop logic and test Deadband
        self.controller.run_spool_loop = True
        mock_time.side_effect = [2.0, 2.0, 2.01]
        mock_sleep.side_effect = stop_loop
        
        self.controller.target_length = 5.0
        # Just barely inside the deadband
        self.mock_sc_instance.get_unspooled_length.return_value = 5.0001 
        
        self.controller.trackingLoop()
        # Deadband met: aim speed should zero and target should clear
        self.assertIsNone(self.controller.target_length)
        self.assertEqual(self.controller.aim_line_speed, 0.0)

    @patch('time.sleep')
    @patch('time.time')
    def test_trackingLoop_birdsnest_prevention(self, mock_time, mock_sleep):
        """Verify outspooling is softly muted when tension drops (birdsnest prevention)."""
        mock_time.side_effect = [1.0, 1.0, 1.01]
        
        def stop_loop(*args):
            self.controller.run_spool_loop = False
        mock_sleep.side_effect = stop_loop
        
        # Torque error > 0 means the line has gone slack.
        # Target torque is -0.01. So giving it a positive torque (+0.1) forces a slack condition.
        self.mock_motor.get_states.return_value = {'pos': 0.0, 'vel': 0.0, 'torq': 0.1}
        
        # We try to outspool line (wanted_motor_vel > 0)
        self.controller.aim_line_speed = 1.0 
        
        self.controller.trackingLoop()
        
        # Extract the velocity commanded to the motor
        cmd_vel_calls = self.mock_motor.send_cmd_vel.call_args_list
        last_cmd_vel = cmd_vel_calls[-1].kwargs['target_velocity']
        
        # Because torque err > 0 and wanted vel > 0, mute = 0.
        # Smooth_mute pulls it down heavily from 1 towards 0. 
        # So commanded velocity should be less than the raw aim requested.
        raw_wanted_vel = (1.0 / self.controller.meters_per_rev) * (2 * math.pi)
        self.assertLess(last_cmd_vel, raw_wanted_vel)

if __name__ == '__main__':
    unittest.main()