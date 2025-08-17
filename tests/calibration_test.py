import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calibration import *
import unittest
import numpy as np
from math import pi

class TestPoseFunctions(unittest.TestCase):

    def setUp(self):
        self.anchor_poses = np.array([
            ((0, 0, -pi/4), (3, 3, 3)),
            ((0, 0, -3*pi/4), (3, -3, 3)),
            ((0, 0, pi/4), (-3, 3, 3)),
            ((0, 0, 3*pi/4), (-3, -3, 3)),
        ])

    def test_calibration(self):

        anchor_poses = np.array([
            ((0, 0, -pi/4), (3, 3, 3)),
            ((0, 0, -3*pi/4), (3, -3, 3)),
            ((0, 0, pi/4), (-3, 3, 3)),
            ((0, 0, 3*pi/4), (-3, -3, 3)),
        ])

        # TODO generate realistic observations
        observations = []
        for i in range(15):
            entry = {'encoders': np.random.uniform(-20,20, (4,))}
            entry['visuals'] = np.random.uniform(-1,1, (4, 10, 2, 3))
            observations.append(entry)
        new_params = find_cal_params(self.anchor_poses, observations, 0)

    def test_order_points_for_low_travel(self):
        points, distance = order_points_for_low_travel(np.random.uniform(-3,3,(20,3)))
        self.assertLess(distance, 40)
        self.assertEqual(points.shape, (20,3))

    def test_calibration_cost_fn(self):
        actual_gantry_position = np.array([0, 0, 1.5])
        full_length = 7.5
        diameter_mm = 26
        distance = np.linalg.norm(actual_gantry_position - self.anchor_poses[0])
        encoder_val = -(full_length - distance) / (pi * diameter_mm * 0.001)
        # 4.5 meters of line are out on each gantry.

        # generate some parameters that should be perfect and therefore have a approximately zero cost.
        params = []
        spools = []
        for ap in self.anchor_poses:
            # no spiral
            spools.append(SpiralCalculator(
                empty_diameter=diameter_mm,
                full_diameter=diameter_mm,
                full_length=full_length,
                gear_ratio=20/51,
                motor_orientation=-1))
            guess = [
                *ap[0],    # actual rotation of anchor
                ap[1][0],    # anchor x pos
                ap[1][1],    # anchor x pos
                encoder_val, # encoder angle at which line length would be zero.
            ]
            params.append(guess)
        params = np.array(params).flatten()

        # single sample location
        # single pose observed from each camera, all identical
        # TODO, calculate what the perfect pose observation would actually be.
        pose = np.array([[0.0, 0.0, 0.0], [0.5, 2.8, distance]])

        observations = []
        observations.append({
            'encoders': [0, 0, 0, 0], # encoder angles for four spool motors
            'visuals': [ # a list of poses for each anchor
                [pose], # visual observations from camera 1
                [pose], # visual observations from camera 2
                [pose], # visual observations from camera 3
                [pose], # visual observations from camera 4
            ],
        })

        # run the code under test
        cost_a = calibration_cost_fn(params, observations, spools)
        self.assertLess(cost_a, 0.5)

        # run again with some error in each encoder.
        for i in range(4):
            observations[0]['encoders'] = [0, 0, 0, 0]
            observations[0]['encoders'][i] = 1 # off by one revolution
            cost_b = calibration_cost_fn(params, observations, spools)
            self.assertLess(cost_a, cost_b)

