
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from position_estimator import Positioner2, sphere_intersection, sphere_circle_intersection, find_hang_point
import time
from multiprocessing import Queue
from data_store import DataStore
import numpy as np
from math import pi, sqrt
import time

class TestPositionEstimator(unittest.TestCase):

    def setUp(self):
        datastore = DataStore(horizon_s=10, n_cables=4)
        to_ui_q = Queue()
        to_pe_q = Queue()
        to_ob_q = Queue()
        to_ui_q.cancel_join_thread()
        to_pe_q.cancel_join_thread()
        to_ob_q.cancel_join_thread()
        self.pe = Positioner2(datastore, to_ui_q, to_pe_q, to_ob_q)

    def test_sphere_intersection(self):
        sphere1 = (np.array([0, 0, 0]), 5)
        sphere2 = (np.array([6, 0, 0]), 5)

        intersection = sphere_intersection(sphere1, sphere2)
        self.assertTrue(intersection is not None)

        center, normal, radius = intersection
        np.testing.assert_array_almost_equal(center, [3,0,0])
        np.testing.assert_array_almost_equal(normal, [1,0,0])
        self.assertAlmostEqual(radius, 4, 6)

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

    def test_find_hang_point(self):
        anchors = np.array([
            [-3, 2, 3],
            [ 3, 2, 3],
            [ 3, 2,-3],
            [-3, 2,-3],
        ], dtype=float)
        lengths = np.array([4, 4, 3.5, 8], dtype=float)
        result = find_hang_point(anchors, lengths)
        self.assertTrue(result is not None)
        point, slack_lines = result

        np.testing.assert_array_almost_equal(point, [0,1,0])

        self.assertFalse(slack_lines[0])
        self.assertFalse(slack_lines[1])
        self.assertFalse(slack_lines[2])
        self.assertTrue(slack_lines[3])

    # def test_estimate(self):
    #     ndims = 3
    #     params = np.concatenate([
    #         np.repeat(np.arange(1, self.pe.n_ctrl_pts+1), ndims),
    #         np.repeat(np.arange(1, self.pe.n_ctrl_pts+1)+10, ndims)])
    #     # noise = np.random.normal(0, 1e-6, params.shape)
    #     # params = params + noise
    #     self.pe.set_splines_from_params(params)

    #     now = time.time()-1
    #     self.pe.time_domain = (now - self.pe.horizon_s, now + self.pe.horizon_s)
    #     self.pe.estimate()