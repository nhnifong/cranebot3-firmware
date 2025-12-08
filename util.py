import numpy as np
from generated.nf import common

def tonp(vec: common.Vec3):
    return np.array([vec.x, vec.y, vec.z], dtype=float)

def clamp(x,small,big):
    return max(min(x,big),small)