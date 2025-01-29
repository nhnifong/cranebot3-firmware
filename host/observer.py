import numpy as np

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
    """
    def __init__(self, shape):
        self.arr = np.zeros(shape)
        self.idx = 0

    def insert(self, row):
        self.arr[self.idx] = row
        self.idx = (self.idx + 1) % self.arr.shape[0]

class Observer:
    """
    This class is meant to continuously collect the measurable variables of the robot and store them in circular buffers.
    """

    def __init__(self, horizon_s, n_cables):
        """
        Initialize measurement arrays with sizes proportional to the approximate number of seconds of data we expect to store.

        all measurements
        n_measurements can be different for every array
        
        gantry_position: shape (n_measurements, 4) TXYZ
        gripper_position: shape (n_measurements, 4) TXYZ
        imu_accel: shape (n_measurements, 4) each row TXYZ
        winch_line_record: shape (n_measurements, 2) TL
        anchor_line_record: shape (n_measurements, n_cables+1) TLLL one L for each line
        gripper_position: shape (n_measurements, 4) TXYZ
        """
        self.n_cables = n_cables

        self.gantry_position = CircularBuffer((horizon_s * 10, 4))
        self.gripper_position = CircularBuffer((horizon_s * 10, 4))
        self.imu_accel = CircularBuffer((horizon_s * 20, 4))
        self.winch_line_record = CircularBuffer((horizon_s * 10, 2))
        self.anchor_line_record = CircularBuffer((horizon_s * 10, n_cables+1))
        self.gripper_position_desired = CircularBuffer((horizon_s * 10, 4))