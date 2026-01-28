import numpy as np
from time import time
from random import random
import asyncio

class CircularBuffer:
    """
    circular buffer implemented as a numpy array
    """
    def __init__(self, shape):
        self.shape = shape
        self.arr = np.zeros(shape, dtype=np.float64)
        # start pointed at the last item in the array. When we write the first, it will be written to 0
        # at any time, self.idx would then point to the last item to be written.
        self.idx = shape[0]-1

    def asNpa(self):
        """Return the entire array without reordering or reshaping anything"""
        return self.arr

    def deepCopy(self, cutoff=None, before=False):
        """
        Return a deep copy of the array.
        if cutoff is provided (a float timestamp)
        then only rows after that time will be returned.
        if before=True, rows before the cutoff are returned
        """
        arr = np.roll(self.arr, -1*(self.idx+1), axis=0)
        if cutoff is not None:
            if before:
                arr = arr[arr[:,0]<=cutoff]
            else:
                arr = arr[arr[:,0]>cutoff]
        return arr

    def insert(self, row):
        self.idx = (self.idx + 1) % self.shape[0]
        self.arr[self.idx] = row

    def insertList(self, row_list):
        """Insert a list of measurements. Newest at end."""
        for row in row_list:
            self.idx = (self.idx + 1) % self.shape[0]
            self.arr[self.idx] = row

    def getLast(self):
        return self.arr[self.idx]

    def getClosest(self, timestamp):
        """Get the data point closest to the given timestamp"""

        # reorder from oldest to newest
        sorted_arr = np.roll(self.arr, -1 * (self.idx + 1), axis=0)
        # Filter out any uninitialized rows
        valid_data = sorted_arr[sorted_arr[:, 0] > 0]
        insertion_index = np.searchsorted(valid_data[:, 0], timestamp)

        # Handle edge cases: timestamp is outside the range of stored data
        if insertion_index == 0:
            return valid_data[0]
        if insertion_index == len(valid_data):
            return valid_data[-1]

        # Compare the two neighbors and return the closer one.
        # In case of a tie, the later measurement (after) is returned.
        before = valid_data[insertion_index - 1]
        after = valid_data[insertion_index]
        return before if (timestamp - before[0]) < (after[0] - timestamp) else after

class DataStore:
    """
    This class is meant to store continuously collected measurable variables of the robot and store them in circular buffers.
    """

    def __init__(self, size=64, n_anchors=4):
        """
        Initialize measurement arrays with sizes proportional to the approximate number of seconds of data we expect to store.
        
        gantry_pos: shape (size, 5) T N XYZ   time, anchor_num, z, y, x
        imu_quat: shape (size, 5) each row TXYZW
        winch_line_record: shape (size, 3) TLS  # used as wrist record in arp gripper (Time, Angle, 0)
        anchor_line_record: shape (size, 4) TLST  time, length, speed, tight.  one for each line
        range_record: shape (size, 3) TL
        finger: shape (size, 3) TAV time, angle, pad_voltage
        """
        self.n_anchors = n_anchors

        self.gantry_pos = CircularBuffer((size, 5))
        self.imu_quat = CircularBuffer((size, 5))
        self.winch_line_record = CircularBuffer((size, 3))
        self.anchor_line_record = [CircularBuffer((size, 4)) for n in range(n_anchors)]
        self.range_record = CircularBuffer((size, 2))
        self.finger = CircularBuffer((size, 3))

        # events that trip when data is added to certain circular buffers
        self.anchor_line_record_event = asyncio.Event()
        self.gantry_pos_event = asyncio.Event()