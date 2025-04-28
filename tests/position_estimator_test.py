
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from position_estimator import CDPR_position_estimator
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
        self.pe = CDPR_position_estimator(datastore, to_ui_q, to_pe_q, to_ob_q)
        self.now = 1739888888.5555555
        self.pe.time_domain = (self.now - 10, self.now + 10)

    def test_model_time(self):
        result = self.pe.model_time(self.now)
        self.assertEqual(result, 0.5)

    def test_model_time_numpy_float(self):
        times = np.array([self.now,self.now,self.now], dtype=float)
        result = self.pe.model_time(times[0])
        self.assertEqual(result, 0.5)

    def test_error_meas(self):
        def func(times):
            arr = times*2
            return arr.reshape(-1, 1) # create column of data, like bspline
        position_measurements = np.array([
            [0.1, 0.21],
            [0.2, 0.41],
            [0.3, 0.61],
            [0.4, 0.81],
        ])
        error = self.pe.error_meas(func, position_measurements, normalize_time=False)
        self.assertAlmostEqual(error, 0.01)

    def test_forces_stable(self):
        """If the gripper hangs directly below the gantry, are the forces balanced?"""
        gant_pos = np.array([[0,0,1]])
        grip_pos = np.array([[0.,0.,0.1]])
        self.pe.times = np.array([123.4])
        result = self.pe.calc_gripper_accel_from_forces(gant_pos, grip_pos)
        expected = np.array([[123.4, 0, 0, 0]])
        np.testing.assert_array_almost_equal(result, expected)


    def test_forces_model_swing(self):
        """If we integrate the forces from calc_gripper_accel_from_forces would the pendulum actually swing?"""
        gant_pos = np.array([[0,0,2]])
        p = np.array([0.4,0,0.1])
        v = np.array([0.,0.,0.])
        time = 1000
        tick = 1/50
        rope_len = np.linalg.norm(gant_pos-p)
        print(f'rope len = {rope_len}')
        expected_period = 2*pi*sqrt(rope_len/9.81)

        start = 1000
        period = []
        osc = 0

        for i in range(2000):
            grip_pos = np.array([p])
            last_sign = (p[0]>0)

            self.pe.times = np.array([time])
            result = self.pe.calc_gripper_accel_from_forces(gant_pos, grip_pos)
            accel = result[0][1:]
            v += accel * tick
            p += v * tick
            time += tick

            # measure the period of oscillation.
            now_sign = (p[0]>0)
            if last_sign != now_sign:
                if start != 1000:
                    period.append((time - start) * 2)
                start = time
                osc += 1

        mean_period = np.mean(period)
        print(f'counted {osc} half-swings')
        print(f'mean period of oscillation = {mean_period} seconds')
        self.assertTrue(osc>1)
        self.assertAlmostEqual(expected_period, mean_period, 1)

    def test_set_splines(self):
        # model consists of two 3D splines.
        # 2 splines
        # 3 dimensions
        # N control points
        nsplines = 2
        ndims = 3
        model_size = nsplines * ndims * self.pe.n_ctrl_pts
        params = np.concatenate([
            np.repeat(np.arange(1, self.pe.n_ctrl_pts+1), ndims),
            np.repeat(np.arange(1, self.pe.n_ctrl_pts+1)+10, ndims)])
        self.assertEqual(model_size, len(params))

        self.pe.set_splines_from_params(params)
        
        np.testing.assert_array_almost_equal(self.pe.gripper_pos_spline(0.5), [6,6,6])
        np.testing.assert_array_almost_equal(self.pe.gantry_pos_spline(0.5),  [16,16,16])
        
        np.testing.assert_array_almost_equal(self.pe.gripper_velocity(0.5), [9,9,9])
        np.testing.assert_array_almost_equal(self.pe.gantry_velocity(0.5),  [9,9,9])
        
        np.testing.assert_array_almost_equal(self.pe.gripper_accel_func(0.5), [0,0,0])
        np.testing.assert_array_almost_equal(self.pe.gantry_accel_func(0.5),  [0,0,0])

    def test_move_spline_domain_robust(self):
        self.pe.move_spline_domain_robust(self.pe.gripper_pos_spline, 0.04)

    def test_estimate(self):
        ndims = 3
        params = np.concatenate([
            np.repeat(np.arange(1, self.pe.n_ctrl_pts+1), ndims),
            np.repeat(np.arange(1, self.pe.n_ctrl_pts+1)+10, ndims)])
        # noise = np.random.normal(0, 1e-6, params.shape)
        # params = params + noise
        self.pe.set_splines_from_params(params)

        now = time.time()-1
        self.pe.time_domain = (now - self.pe.horizon_s, now + self.pe.horizon_s)
        self.pe.estimate()