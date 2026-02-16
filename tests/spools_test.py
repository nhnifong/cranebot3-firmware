"""
Unit tests for spool controller
"""
import pytest
pytestmark = pytest.mark.pi
pytest.importorskip("gpiodevice")

import unittest
import numpy as np
from math import pi
import asyncio
import time

from nf_robot.robot.spools import *
from nf_robot.robot.debug_motor import DebugMotor
from nf_robot.robot.anchor_server import default_anchor_conf
from nf_robot.robot.gripper_server import default_gripper_conf

class TestSpoolControllerInit(unittest.TestCase):

    def setUp(self):
        self.debug_motor = DebugMotor()

    def test_init_for_anchor(self):
        conf = default_anchor_conf.copy()
        spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=30, full_length=10, conf=conf)
        self.assertEqual(False, spooler.spoolPause)
        self.assertLessEqual(0.02*pi, spooler.meters_per_rev)
        self.assertGreaterEqual(0.03*pi, spooler.meters_per_rev)

    def test_init_for_gripper(self):
        conf = default_gripper_conf.copy()
        spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=30, full_length=10, conf=conf)
        self.assertEqual(False, spooler.spoolPause)

    def test_set_ref_length(self):
        conf = default_anchor_conf.copy()
        diameter_mm = 20
        spooler = SpoolController(self.debug_motor, empty_diameter=diameter_mm, full_diameter=diameter_mm, full_length=10, conf=conf)
        self.debug_motor.position = 100
        cir_before = spooler.meters_per_rev
        za_before = spooler.sc.zero_angle

        # there are 4 meters of line unspooled, and 6 meters of line spooled.
        expected_zero_angle = self.debug_motor.position * -1 - 6/(diameter_mm*0.001*pi)
        spooler.setReferenceLength(4)

        self.assertAlmostEqual(expected_zero_angle, spooler.sc.zero_angle, 5)

        # expect the current length to be what we just set the current reference length to be
        length = spooler.sc.get_unspooled_length(self.debug_motor.position)
        self.assertAlmostEqual(4, length, 5)

    def test_command_speed(self):
        conf = default_anchor_conf.copy()
        spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=30, full_length=10, conf=conf)
        self.debug_motor.position = 100
        self.debug_motor.speed = 0
        spooler._commandSpeed(9)
        self.assertEqual(9, self.debug_motor.speed)


class TestSpoolControllerTracking(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.motor = DebugMotor()
        self.motor.position = 0
        self.conf = default_anchor_conf.copy()
        # Set start length to 0 for simplicity (empty spool logic depends on calc, 
        # but we'll just trust the motor pos 0 aligns with calculator for these tests)
        self.spooler = SpoolController(self.motor, empty_diameter=20, full_diameter=20, full_length=100, conf=self.conf)
        # override for faster test
        self.spooler.conf['CRUISE_SPEED'] = 0.5
        self.spooler.conf['LOOP_DELAY_S'] = 0.01
        self.spooler.conf['MAX_ACCEL'] = 1.5
        
        
        # Force start at 10m so we have room to reel in or out
        self.spooler.setReferenceLength(10.0)

        # Start the tracking loop in a background thread
        self.spooler.run_spool_loop = True
        self.loop_thread = asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop))

    async def asyncTearDown(self):
        self.spooler.fastStop()
        try:
            await self.loop_thread
        except:
            pass

    async def test_position_accuracy_move_out(self):
        """Test moving from 10m to 12m"""
        target = 12.0
        self.spooler.setTargetLength(target)
        
        # Allow time to travel 2 meters. 
        # at 0.5m/s (cruise) it takes 4s. Add buffer for accel.
        await asyncio.sleep(8) 
        
        _, current_len = self.spooler.currentLineLength()
        print(f"Move Out: Target {target}, Actual {current_len}")
        
        # Verify 10cm accuracy (0.1m)
        self.assertAlmostEqual(current_len, target, delta=0.1)
        
        # Verify motor stopped
        self.assertAlmostEqual(self.motor.speed, 0, delta=0.1)

    async def test_position_accuracy_reel_in(self):
        """Test moving from 10m to 8m"""
        target = 8.0
        self.spooler.setTargetLength(target)
        
        await asyncio.sleep(8) 
        
        _, current_len = self.spooler.currentLineLength()
        print(f"Reel In: Target {target}, Actual {current_len}")
        
        self.assertAlmostEqual(current_len, target, delta=0.1)

    async def test_jog_relative(self):
        """Test jogging +1 meter"""
        start_target = self.spooler.target_length
        self.spooler.jogRelativeLen(1.0)
        
        self.assertAlmostEqual(self.spooler.target_length, start_target + 1.0, delta=1e-4)
        
        await asyncio.sleep(4)
        _, current_len = self.spooler.currentLineLength()
        self.assertAlmostEqual(current_len, start_target + 1.0, delta=0.1)

    async def test_speed_tracking_mode(self):
        """Test raw speed control"""
        self.spooler.setAimSpeed(0.2) # 0.2 m/s
        
        # Wait for acceleration
        await asyncio.sleep(1)
        
        # Check internal state matches request
        self.assertEqual(self.spooler.tracking_mode, 'speed')
        
        # Check actual speed calculation
        # Motor revs/sec * meters/rev approx = 0.2
        estimated_speed = self.motor.speed * self.spooler.meters_per_rev
        self.assertAlmostEqual(estimated_speed, 0.2, delta=0.05)

        # Test reversal
        self.spooler.setAimSpeed(-0.2)
        await asyncio.sleep(2) # Wait for decel and accel
        estimated_speed = self.motor.speed * self.spooler.meters_per_rev
        self.assertAlmostEqual(estimated_speed, -0.2, delta=0.05)

    async def test_pop_measurements(self):
        """Verify data recording works in position mode"""
        self.spooler.setTargetLength(15.0)
        await asyncio.sleep(1)
        
        data = self.spooler.popMeasurements()
        
        # Should have data
        self.assertGreater(len(data), 0)
        
        # Check row structure: (time, length, speed) or (time, length, speed, tight)
        # Since tight_check_fn is None in setUp:
        first_row = data[0]
        self.assertEqual(len(first_row), 3) 
        self.assertIsInstance(first_row[0], float) # time
        self.assertIsInstance(first_row[1], float) # length
        self.assertIsInstance(first_row[2], float) # speed

        # Verify clearing behavior
        data_second = self.spooler.popMeasurements()
        # Might be 0 or very few depending on thread timing, but definitely cleared old ones
        self.assertLess(len(data_second), len(data))

class TestSpiralCalulator(unittest.TestCase):
    def testDirection(self):
        sc = SpiralCalculator(empty_diameter=25, full_diameter=27, full_length=7.5, gear_ratio=20/51, motor_orientation=-1)
        # the zero angle starts at 0. this is the encoder position (in revs) where we expect the line to be completely unspooled

        self.assertAlmostEqual(sc.get_unspooled_length(0), 7.5)
        self.assertAlmostEqual(sc.get_unspooled_length(-50), 5.9472, 3)
        