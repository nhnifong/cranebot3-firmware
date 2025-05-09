"""
Unit tests for datatore class
"""
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import time
import multiprocessing
from data_store import CircularBuffer, DataStore


class TestDatastore(unittest.TestCase):
    def test_circular_buffer_insert_read(self):
        cb = CircularBuffer((3,1))
        cb.insert(0.1)
        out = cb.deepCopy()
        self.assertEqual(out[0], 0.1)

    def test_circularity(self):
        cb = CircularBuffer((3,1))
        cb.insert(0.1)
        cb.insert(0.2)
        cb.insert(0.3)
        cb.insert(0.4)
        out = cb.deepCopy()
        self.assertEqual(out[0], 0.4)
        self.assertEqual(out[1], 0.2)
        self.assertEqual(out[2], 0.3)

    def test_datastore_init(self):
        dlen = 30
        anchors = 4
        d = DataStore(dlen, anchors)

        self.assertEqual(anchors, len(d.anchor_line_record))
        for aline in d.anchor_line_record:
            self.assertEqual((dlen, 4), aline.shape)

    def test_datastore_init_3(self):
        # does it still work with three anchors
        d2 = DataStore(1, 3)

    def test_get_last(self):
        d = DataStore(30, 4)
        d.winch_line_record.insert(np.array([123.4, 1.0, 0.0]))
        d.winch_line_record.insert([123.5, 2.0, 0.0])
        d.winch_line_record.insert((123.6, 3.0, 10.0))
        last = d.winch_line_record.getLast()
        self.assertEqual(123.6, last[0])
        self.assertEqual(3.0, last[1])
        self.assertEqual(10.0, last[2])

        # insert numpy array
        d.winch_line_record.insertList(np.array([
            [124.1, 4.0, 0],
            [124.2, 5.0, 0],
            [124.3, 6.0, 0],
            ]))
        # insert regular list
        d.winch_line_record.insertList([
            (124.4, 7.0, 0),
            (124.5, 8.0, 0),
            (124.6, 9.0, 10),
            ])
        last = d.winch_line_record.getLast()
        self.assertEqual(124.6, last[0])
        self.assertEqual(9.0, last[1])
        self.assertEqual(10, last[2])

    def test_overflow(self):
        d = DataStore(30, 4)
        for i in range(200):
            d.anchor_line_record[0].insert(np.array([i, 0, 0, 0]))
            last = d.anchor_line_record[0].getLast()
            self.assertEqual(i, last[0])