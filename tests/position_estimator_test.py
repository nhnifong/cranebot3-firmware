
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