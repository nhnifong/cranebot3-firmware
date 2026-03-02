import numpy as np
from nf_robot.generated.nf import common

def tonp(vec: common.Vec3):
    return np.array([vec.x, vec.y, vec.z], dtype=float)

def fromnp(arr: np.ndarray):
    return common.Vec3(float(arr[0]), float(arr[1]), float(arr[2]))

def clamp(x,small,big):
    return max(min(x,big),small)

def remap(x, ilow, ihigh, olow, ohigh):
    return (x-ilow) / (ihigh-ilow) * (ohigh-olow) + olow 

def poseTupleToProto(p):
    return common.Pose(rotation=fromnp(p[0]), position=fromnp(p[1]))

def poseProtoToTuple(p):
    return (np.array([p.rotation.x, p.rotation.y, p.rotation.z]), np.array([p.position.x, p.position.y, p.position.z]))

# A simple class for handling Proportional, Integral & Derivative (PID) control calculations
class PID:
    def __init__(self, kp, ki, kd, sample_rate):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = 0
        self._error_sum = 0
        self._last_value = 0
        self._sample_rate = sample_rate

    def calculate(self, value, value_change=None):
        error = self.setpoint - value
        self._error_sum += error * self._sample_rate
        if value_change is None:
            rate_error = (value - self._last_value) / self._sample_rate
        else:
            rate_error = value_change
        self._last_value = value

        return (error * self.kp) + (self._error_sum * self.ki) - (rate_error * self.kd)