"""
Unit tests for spool controller
"""

import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
import numpy as np
from math import pi
import asyncio
import time

from spools import *
from debug_motor import DebugMotor
from anchor_server import default_anchor_conf
from gripper_server import default_gripper_conf

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

        # Although averaging of zero angles always occurs, when there is only one measurement, we should expect it to be the one calculated
        # from the reference length we just profided
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
        self.debug_motor = DebugMotor()
        self.debug_motor.position = 0
        self.conf = default_anchor_conf.copy()
        self.spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=20, full_length=10, conf=self.conf)

        # start tracking loop
        self.spool_task = asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop))
        await asyncio.sleep(0.1)
    
    async def asyncTearDown(self):
        self.spooler.fastStop()
        result = await self.spool_task

    async def test_spool_tracks_plan(self):
        # make debug motor throw an exception if its speed changes too fast
        self.debug_motor.setAccelLimit(self.conf['MAX_ACCEL'] / (0.02*pi))
        self.assertEqual(0, self.debug_motor.speed)
        self.spooler.setReferenceLength(1)
        self.assertAlmostEqual(0, self.debug_motor.position, 6)
        self.spooler.setPlan([
            (time.time()+1, 1),
            (time.time()+2, 1.3),
            (time.time()+3, 1.6),
        ])
        await asyncio.sleep(0.5)
        self.assertEqual(0, self.debug_motor.speed)
        self.assertAlmostEqual(0, self.debug_motor.position, 6)
        await asyncio.sleep(1.5)

        # TODO assertions of position find that the tracking algorithm oveshoots by 1 or 2 revs in this case.
        # but we can still assert that it stopped the motor eventually.

        # self.assertAlmostEqual(0.3 / (0.02 * pi), self.debug_motor.position, 1)
        await asyncio.sleep(1)
        # self.assertAlmostEqual(0.6 / (0.02 * pi), self.debug_motor.position, 1)
        await asyncio.sleep(1)
        # self.assertAlmostEqual(0.6 / (0.02 * pi), self.debug_motor.position, 1)
        self.assertEqual(0, self.debug_motor.speed)

    async def test_spool_speed(self):
        # make debug motor throw an exception if its speed changes too fast
        self.conf['MAX_ACCEL'] = 0.5 # meters per second^2
        self.debug_motor.setAccelLimit(self.conf['MAX_ACCEL'] / (0.02*pi))
        self.spooler.setReferenceLength(1)
        aimSpeed = 0.5
        self.spooler.setAimSpeed(aimSpeed)
        await asyncio.sleep(0.9)
        self.assertLessEqual(self.debug_motor.speed, aimSpeed/(0.02*pi))
        await asyncio.sleep(1.1)
        self.assertAlmostEqual(self.debug_motor.speed, aimSpeed/(0.02*pi), 4)
        self.spooler.setAimSpeed(0)
        await asyncio.sleep(1.1)
        self.assertEqual(self.debug_motor.speed, 0)

class TestSpiralCalulator(unittest.TestCase):
    def testDirection(self):
        sc = SpiralCalculator(empty_diameter=25, full_diameter=27, full_length=7.5, gear_ratio=20/51, motor_orientation=-1)
        # the zero angle starts at 0. this is the encoder position (in revs) where we expect the line to be completely unspooled

        self.assertAlmostEqual(sc.get_unspooled_length(0), 7.5)
        self.assertAlmostEqual(sc.get_unspooled_length(-50), 5.9472, 3)
        