import numpy as np
from time import time
from random import random

class CircularBuffer:
    """
    circular buffer implemented as a numpy
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
        arr = self.arr.copy()
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

class DataStore:
    """
    This class is meant to store continuously collected measurable variables of the robot and store them in circular buffers.
    """

    def __init__(self, size=64, n_anchors=4):
        """
        Initialize measurement arrays with sizes proportional to the approximate number of seconds of data we expect to store.
        
        gantry_pos: shape (size, 4) T XYZ
        imu_rotvec: shape (size, 5) each row TXYZ
        winch_line_record: shape (size, 3) TSL
        anchor_line_record: shape (size, 3) TSLT  time, length, speed, tension. one for each line
        range_record: shape (size, 3) TL
        """
        self.n_anchors = n_anchors

        self.gantry_pos = CircularBuffer((size, 4))
        self.imu_rotvec = CircularBuffer((size, 4))
        self.winch_line_record = CircularBuffer((size, 3))
        self.anchor_line_record = [CircularBuffer((size, 4)) for n in range(n_anchors)]
        self.range_record = CircularBuffer((size, 2))