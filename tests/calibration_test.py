import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from calibration import *
import unittest
import numpy as np
from math import pi

class TestPoseFunctions(unittest.TestCase):
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
        new_params = find_cal_params(anchor_poses, observations)