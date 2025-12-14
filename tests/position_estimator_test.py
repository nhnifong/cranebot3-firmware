
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from position_estimator import *
from unittest.mock import MagicMock, patch
import time
from multiprocessing import Queue
from data_store import DataStore
import numpy as np
from math import pi, sqrt, sin, cos
import time
from random import random
from observer import AsyncObserver
from scipy.spatial.transform import Rotation
from config_loader import create_default_config

class TestPositionEstimator(unittest.TestCase):

    def setUp(self):
        self.datastore = DataStore(size=200)
        self.mock_observer = MagicMock()
        self.mock_observer.config = create_default_config()
        self.pe = Positioner2(self.datastore, self.mock_observer)

    def test_sphere_intersection(self):
        sphere1 = (np.array([0, 0, 0]), 5)
        sphere2 = (np.array([6, 0, 0]), 5)

        intersection = sphere_intersection(sphere1, sphere2)
        self.assertTrue(intersection is not None)

        center, normal, radius = intersection
        np.testing.assert_array_almost_equal(center, [3,0,0])
        np.testing.assert_array_almost_equal(normal, [1,0,0])
        self.assertAlmostEqual(radius, 4, 6)

    def test_sphere_intersection2(self):
        sphere1 = (np.array([0, 0, 0]), 1.1)
        sphere2 = (np.array([1, 1, 1]), 0.7)

        intersection = sphere_intersection(sphere1, sphere2)
        self.assertTrue(intersection is not None)

        center, normal, radius = intersection
        np.testing.assert_array_almost_equal(center, [0.62,0.62,0.62])
        np.testing.assert_array_almost_equal(normal, [0.57735, 0.57735, 0.57735]) # 1/sqrt(3)
        self.assertAlmostEqual(radius, 0.238327, 5)

    def test_sphere_intersection3(self):
        sphere1 = (np.array([0, 0, 10]), 3)
        sphere2 = (np.array([3, 4, 10]), 4)

        intersection = sphere_intersection(sphere1, sphere2)
        self.assertTrue(intersection is not None)

        center, normal, radius = intersection
        np.testing.assert_array_almost_equal(center, [1.08,  1.44, 10])
        np.testing.assert_array_almost_equal(normal, [0.6, 0.8, 0]) # 1/sqrt(3)
        self.assertAlmostEqual(radius, 2.4, 5)

    def test_sphere_circle_intersection(self):
        sphere_center = np.array([0, 0, 0])
        sphere_radius = 5
        circle_center = np.array([3, 0, 0])
        circle_normal = np.array([0, 0, 1])
        circle_radius = 4

        pts = sphere_circle_intersection(sphere_center, sphere_radius, circle_center, circle_normal, circle_radius)
        self.assertEqual(2, len(pts))

        np.testing.assert_array_almost_equal(pts[0], [3,-4,0])
        np.testing.assert_array_almost_equal(pts[1], [3,4,0])

    def test_lowest_point_on_circle(self):
        center = np.array([ 1, 1, 2])
        normal = np.array([-0.57735, -0.57735, -0.57735])
        radius = 2

        # circles with normals pointing up or down don't have a lowest point
        result = lowest_point_on_circle(center, normal, radius)
        self.assertTrue(result is not None)
        np.testing.assert_array_almost_equal(result, [1.816496, 1.816496, 0.367006]) # 1/sqrt(3)

    def test_lowest_point_on_circle_flat(self):
        center = np.array([ 2, 3, 4])
        normal = np.array([ 0.,  0., -1.])
        radius = 2

        # circles with normals pointing up or down don't have a lowest point
        result = lowest_point_on_circle(center, normal, radius)
        self.assertTrue(result is None)


    def test_find_hang_point(self):
        anchors = np.array([
            [-3, 3, 2],
            [ 3, 3, 2],
            [ 3,-3, 2],
            [-3,-3, 2],
        ], dtype=float)

        expected_hang_point = np.array([0,1,1])
        lengths = np.linalg.norm(anchors - expected_hang_point, axis=1)
        # make one length too long
        lengths[3] = 6

        result = find_hang_point(anchors, lengths)
        self.assertTrue(result is not None)
        point, slack_lines = result

        np.testing.assert_array_almost_equal(point, expected_hang_point)

        self.assertFalse(slack_lines[0])
        self.assertFalse(slack_lines[1])
        self.assertFalse(slack_lines[2])
        self.assertTrue(slack_lines[3])


    def test_find_hang_point_2(self):
        # two lines diagonally across from eachother are tight and two are slack

        anchors = np.array([
            [-3, 3, 2],
            [ 3, 3, 2],
            [ 3,-3, 2],
            [-3,-3, 2],
        ], dtype=float)

        expected_hang_point = np.array([0,0,1])

        # make two lengths too long
        lengths = np.array([6, 4.358, 6, 4.358])

        result = find_hang_point(anchors, lengths)
        self.assertTrue(result is not None)
        point, slack_lines = result

        np.testing.assert_array_almost_equal(point, expected_hang_point, 2)

        self.assertTrue(slack_lines[0])
        self.assertFalse(slack_lines[1])
        self.assertTrue(slack_lines[2])
        self.assertFalse(slack_lines[3])

class TestPositionEstimatorAsync(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # make an instance of observer just to use it's simulated data function.
        self.ob = AsyncObserver(terminate_with_ui=False, robot_config=create_default_config())
        self.datastore = self.ob.datastore

    async def test_positioner_main(self):
        
        task = asyncio.create_task(self.ob.pe.main())
        self.ob.run_command_loop = True
        sim_task = asyncio.create_task(self.ob.add_simulated_data_point2point())
        await asyncio.sleep(5)
        self.ob.pe.run = False
        self.ob.run_command_loop = False
        task.cancel()
        result = await asyncio.wait_for(task, 1)
        result = await asyncio.wait_for(sim_task, 1)

class TestEstimateGripper(unittest.TestCase):
    def setUp(self):
        self.mock_datastore = MagicMock()
        self.mock_observer = MagicMock()
        self.mock_observer.config = create_default_config()
        
        self.pe = Positioner2(self.mock_datastore, self.mock_observer)
        self.pe.swing_est = MagicMock()
        
        # Set gantry position
        self.pe.gant_pos = np.array([10.0, 5.0, 20.0], dtype=float)
        
        # Common valid data setup
        self.ts = 123456789.0
        self.winch_length = 5.0
        self.winch_speed = 0.1

        # Real-world "Neutral" data
        self.valid_quat = [3.94042969e-01, -5.91979980e-01, -5.81420898e-01, 3.95324707e-01]

        # Setup default mock returns
        self.pe.datastore.imu_quat.getLast.return_value = [self.ts] + self.valid_quat
        self.pe.datastore.winch_line_record.getLast.return_value = [self.ts, self.winch_length, self.winch_speed]
        self.pe.datastore.range_record.getLast.return_value = [self.ts, 0.4]

    def test_normal_operation_real_data(self):
        self.pe.tip_over.clear()
        self.pe.estimate_gripper()

        # Raw data is 90deg, but corrected is 0deg. Should NOT tip.
        self.assertFalse(self.pe.tip_over.is_set(), "Tipping should not trigger for neutral hanging (corrected) pose")

        # Should receive corrected vector (approx 0,0,0)
        self.pe.swing_est.add_rotation_vector.assert_called_once()
        call_args = self.pe.swing_est.add_rotation_vector.call_args
        # Allow small floating point margin around 0
        np.testing.assert_array_almost_equal(call_args[0][1], [0, 0, -2], decimal=1)

        # Should be hanging straight down from Gantry ([10, 5, 20] -> [10, 5, 15])
        rvec, tvec = self.pe.grip_pose
        np.testing.assert_array_almost_equal(tvec, [10.0, 5.0, 15.0], decimal=1)

    def test_tipping_trigger(self):
        """Verify tip_over.set() is called when rotation is tipped relative to neutral."""
        # Create a raw reading that corresponds to 110 degrees X (90 neutral + 20 tip)
        r_tipped = Rotation.from_euler('x', 110, degrees=True)
        
        self.pe.datastore.range_record.getLast.return_value = [self.ts, 0.1]
        self.pe.datastore.imu_quat.getLast.return_value = [self.ts] + list(r_tipped.as_quat())
        self.pe.tip_over.clear() 

        self.pe.estimate_gripper()

        self.assertTrue(self.pe.tip_over.is_set(), "Should trigger tipping for 110deg raw input (20deg corrected)")

    def test_pose_math(self):
        """
        Test mathematical correctness of pose composition.
        """
        # CASE 1: Real Neutral Input (~90 deg X)
        # Corrected: 0 deg (Vertical)
        # Expected: Gantry [10, 5, 20] + Down 5m = [10, 5, 15]
        r_neutral = Rotation.from_euler('x', 90, degrees=True)
        self.pe.datastore.imu_quat.getLast.return_value = [self.ts] + list(r_neutral.as_quat())
        
        self.pe.estimate_gripper()
        
        _, tvec = self.pe.grip_pose
        np.testing.assert_array_almost_equal(tvec, [10.0, 5.0, 15.0])