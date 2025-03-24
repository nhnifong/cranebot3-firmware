import numpy as np
from multiprocessing import RawArray, Semaphore
from time import time
from random import random

class CircularBuffer:
    """
    continuously appending to numpy arrays is not very good.
    copying slices of python lists into numpy arrays every step is also not good.
    but we are always adding new measurements and advancing a fixed time window.
    therefore we always have a roughly constant number of measurements from each sensor type that lie within that window
    not exactly constant though because of stutter and missed charuco observations
    but a circular pre-allocated array is still viable.
    the error calculation does not care how many measurements there are or what order they are in. the are all tagged with their own timestamps.
    we could do the bookeeping to maintain only measurements within the intended time domain, but it might be simpler, even faster to not do it.
    put any new measurement you get into the circular array, and feed them all to the error function. we can either filter the old values in error_meas
    or just allow them to be used, knowing that they can't get that old anyways with finite storage and new values being constantly collected.
    For now I will opt to not filter them.

    These measurement arrays also need to be shared between multiple processes. one process periodically inserting a few new measurements,
    and one periodically taking a snapshot on which to perform error minimization
    """
    def __init__(self, shape):
        self.shape = shape
        self.arr = RawArray('d', int(np.prod(shape)))
        # start pointed at the last item in the array. When we write the first, it will be written to 0
        # at any time, self.idx would then point to the last item to be written.
        self.idx = shape[0]-1
        self.sem = Semaphore(1)

    def asNpa(self):
        """Return as numpy array with original shape"""
        return np.frombuffer(self.arr, dtype=np.dtype(self.arr)).reshape(self.shape)

    def deepCopy(self):
        with self.sem:
            arr = self.asNpa().copy()
        return arr

    def insert(self, row):
        with self.sem:
            self.idx = (self.idx + 1) % self.shape[0]
            self.asNpa()[self.idx] = row

    def insertList(self, row_list):
        with self.sem:
            arr = self.asNpa()
            for row in row_list:
                self.idx = (self.idx + 1) % self.shape[0]
                arr[self.idx] = row

    def getLast(self):
        with self.sem:
            return self.asNpa()[self.idx]

class DataStore:
    """
    This class is meant to store continuously collected measurable variables of the robot and store them in circular buffers.
    """

    def __init__(self, horizon_s, n_cables):
        """
        Initialize measurement arrays with sizes proportional to the approximate number of seconds of data we expect to store.

        all measurements
        n_measurements can be different for every array
        
        gantry_position: shape (n_measurements, 4) T ROT XYZ
        gripper_position: shape (n_measurements, 4) T ROT XYZ
        imu_accel: shape (n_measurements, 4) each row TXYZ
        winch_line_record: shape (n_measurements, 2) TL
        anchor_line_record: shape (n_measurements, n_cables+1) TLLL one L for each line
        """
        self.horizon_s = horizon_s
        self.n_cables = n_cables

        c = int(horizon_s * 3)
        self.gantry_pose = CircularBuffer((c, 7))
        self.gripper_pose = CircularBuffer((c, 7))
        self.imu_accel = CircularBuffer((c, 4))
        self.winch_line_record = CircularBuffer((c, 2))
        self.anchor_line_record = [CircularBuffer((c, 2)) for n in range(n_cables)]

        self.gantry_pose.insertList([np.array([time()+random()*20, 0,0,0, random()*0.1,random()*0.1,random()*0.1], dtype=np.float64) for i in range(c)])
        self.gripper_pose.insertList([np.array([time()+random()*20, 0,0,0, random()*0.1,random()*0.1,random()*0.1], dtype=np.float64) for i in range(c)])
        self.imu_accel.insertList([np.array([time(), 0,0,0], dtype=np.float64) for i in range(c)])
        self.winch_line_record.insertList([np.array([time(), 1], dtype=np.float64) for i in range(c)])
        for aa in self.anchor_line_record:
            aa.insertList([np.array([time(), 0]) for i in range(c)])