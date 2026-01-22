"""
Unit tests for datatore class
"""
import unittest
import numpy as np
import time
import multiprocessing

from nf_robot.host.data_store import CircularBuffer, DataStore


class TestDatastore(unittest.TestCase):
    def test_circular_buffer_insert_read(self):
        cb = CircularBuffer((3,1))
        cb.insert(0.1)
        out = cb.deepCopy()
        self.assertEqual(out[-1], 0.1)

    def test_circularity(self):
        cb = CircularBuffer((3,1))
        cb.insert(0.1)
        cb.insert(0.2)
        cb.insert(0.3)
        cb.insert(0.4)
        self.assertEqual(cb.arr[0], 0.4)
        self.assertEqual(cb.arr[1], 0.2)
        self.assertEqual(cb.arr[2], 0.3)

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
            d.anchor_line_record[0].insert(np.array([i, 0, 0, 1]))
            last = d.anchor_line_record[0].getLast()
            self.assertEqual(i, last[0])

    def test_deepcopy(self):
        """Assert that when you deep copy the array, your copy ends at the last inserted element"""
        c = CircularBuffer((30, 2))
        c.insert([7,7])
        c.insert([8,8])
        c.insert([9,9])
        a = c.deepCopy()
        np.testing.assert_array_almost_equal(a[-1], [9,9])

    def test_getClosest(self):
        c = CircularBuffer((6, 2))
        c.insertList(np.array([
            [124.1, 3.0],
            [124.2, 4.0],
            [124.3, 5.0],
            [124.4, 6.0], # included
            [124.5, 7.0], # included | desired answer
            [124.6, 8.0], # included
            [124.7, 9.0], # included
            [124.8, 9.0], # included
            [124.9, 9.0], # included
        ]))
        a = c.getClosest(124.54)
        self.assertEqual(a[0], 124.5)