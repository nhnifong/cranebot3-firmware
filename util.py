import numpy as np
from generated.nf import common

def tonp(vec: common.Vec3):
    return np.array([vec.x, vec.y, vec.z], dtype=float)

def fromnp(arr: np.ndarray):
    return common.Vec3(float(arr[0]), float(arr[1]), float(arr[2]))

def clamp(x,small,big):
    return max(min(x,big),small)