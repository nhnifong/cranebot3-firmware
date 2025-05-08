
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from position_estimator import *
import time
from multiprocessing import Queue
from data_store import DataStore
import numpy as np
from math import pi, sqrt
import time

class TestPositionEstimator(unittest.TestCase):

    def setUp(self):
        self.datastore = DataStore(horizon_s=10, n_cables=4)
        to_ui_q = Queue()
        to_pe_q = Queue()
        to_ob_q = Queue()
        to_ui_q.cancel_join_thread()
        to_pe_q.cancel_join_thread()
        to_ob_q.cancel_join_thread()
        self.pe = Positioner2(self.datastore, to_ui_q, to_pe_q, to_ob_q)

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

    def test_estimate(self):
        self.pe.estimate()

    def test_swing_angle_from_params(self):
        expected_angles = np.array([
            [1, -2],
            [0,  0],
            [-1,  2]
        ], dtype=float)
        times = np.array([100,101,102], dtype=float)
        freq = 1/4
        xamp = 1
        yamp = 2
        xphase = 0
        yphase = -pi/2

        angles = swing_angle_from_params(times, freq, xamp, yamp, xphase, yphase)
        np.testing.assert_array_almost_equal(angles, expected_angles)

        # add full circle to both phases, nothing should change
        xphase += 2*pi
        yphase += 2*pi
        angles = swing_angle_from_params(times, freq, xamp, yamp, xphase, yphase)
        np.testing.assert_array_almost_equal(angles, expected_angles)

    def test_find_swing(self):
        self.pe.stop_cutoff = 100

        self.datastore.imu_rotvec.insertList([np.array([98, 0,0,0], dtype=np.float64) for i in range(30)])
        self.datastore.imu_rotvec.insertList(np.array([
            [99,1,1,1],
            [101,0.1,0.2,0.3],
            [102,-0.4,0.5,-0.6],
            [103,-0.2,-0.1,0.1],
        ]))
        self.pe.find_swing()
        expected_params = np.array([ 1.017484, 1, 1, -pi, -1.922795])
        np.testing.assert_array_almost_equal(self.pe.swing_params, expected_params)