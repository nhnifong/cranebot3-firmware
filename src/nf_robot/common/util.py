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