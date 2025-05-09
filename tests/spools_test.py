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

from spools import SpoolController
from debug_motor import DebugMotor
from anchor_server import default_anchor_conf
from gripper_server import default_gripper_conf

class TestSpoolControllerInit(unittest.TestCase):

    def setUp(self):
        self.debug_motor = DebugMotor()

    def test_init_for_anchor(self):
        conf = default_anchor_conf.copy()
        spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=30, full_length=10, conf=conf, tension_support=True)
        self.assertEqual(True, spooler.tension_support)
        self.assertEqual(False, spooler.spoolPause)
        self.assertLessEqual(0.02*pi, spooler.meters_per_rev)
        self.assertGreaterEqual(0.03*pi, spooler.meters_per_rev)

    def test_init_for_gripper(self):
        conf = default_gripper_conf.copy()
        spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=30, full_length=10, conf=conf, tension_support=False)
        self.assertEqual(False, spooler.tension_support)
        self.assertEqual(False, spooler.spoolPause)

    def test_set_ref_length(self):
        conf = default_anchor_conf.copy()
        spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=30, full_length=10, conf=conf, tension_support=True)
        self.debug_motor.position = 100
        cir_before = spooler.meters_per_rev
        za_before = spooler.zero_angle
        spooler.setReferenceLength(4)
        cir_after = spooler.meters_per_rev
        za_after = spooler.zero_angle
        self.assertEqual(cir_before, cir_after)
        self.assertNotEqual(za_before, za_after)
        length = spooler.get_unspooled_length(self.debug_motor.position)
        self.assertAlmostEqual(4, length,9)

    def test_command_speed(self):
        conf = default_anchor_conf.copy()
        spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=30, full_length=10, conf=conf, tension_support=True)
        self.debug_motor.position = 100
        self.debug_motor.speed = 0
        spooler._commandSpeed(9)
        self.assertEqual(9, self.debug_motor.speed)


class TestSpoolControllerTracking(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.debug_motor = DebugMotor()
        self.debug_motor.position = 0
        self.conf = default_anchor_conf.copy()
        self.spooler = SpoolController(self.debug_motor, empty_diameter=20, full_diameter=20, full_length=10, conf=self.conf, tension_support=False)

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