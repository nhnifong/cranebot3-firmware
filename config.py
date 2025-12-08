import json
import numpy as np
from pathlib import Path # Import pathlib

# why is this not a protobuf? I wanted it to be human readable

DEFAULT_CONFIG_PATH = Path(__file__).parent / 'configuration.json'

class Anchor:
    def __init__(self, num):
        self.num = num # the anchor index 0 to 4
        self.service_name = None # 123.cranebot-anchor-service.2ccf67bc3fc4
        self.pose = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float)) # rvec, tvec
        self.address = None # IP address and port where last seen
        self.port = None

        # config vars specific to each anchor, merged with common anchor vars and sent
        self.vars = {}

class Gripper:
    def __init__(self):
        self.service_name = None
        self.address = None # IP address and port where last seen
        self.port = None
        self.frame_room_spin = 50/180*np.pi # todo this is not written anywhere
        self.vars = {}

class Config:
    def __init__(self):
        self.anchors = [Anchor(i) for i in range(4)]
        self.anchor_num_map = {}
        # default configuration
        self.resolution = (1920, 1080) # half the native resolution of the raspberry pi camera module 3
        self.intrinsic_matrix = np.array(
            [[1691.33070175,    0.        , 1163.88476871],
             [   0.        , 1697.39780074,  633.90347492],
             [   0.        ,    0.        ,    1.        ]])
        self.distortion_coeff = np.array(
            [[0.021986, 0.160533, -0.003378, 0.002640, -0.356843]])
        self.commmon_anchor_vars = {}
        self.gripper = Gripper()
        self.preferred_cameras = [0,3,None] # todo this is not written anywhere
        self.robot_id = '0' # todo this is not written anywhere
        try:
            self.reload()
        except FileNotFoundError:
            f'No {DEFAULT_CONFIG_PATH} file exists, using defaults'
            self.write()

    def vars_for_anchor(self, anchor_num):
        v = self.commmon_anchor_vars.copy()
        v.update(self.anchors[anchor_num].vars)
        return v

    def reload(self):
        conf = json.loads(open(DEFAULT_CONFIG_PATH).read())
        self.anchor_num_map = {} # convenience map from service name to num
        for a in conf['anchors']:
            self.anchors[a['num']].pose = (np.array(a['rotation'], dtype=float), np.array(a['position'], dtype=float))
            self.anchors[a['num']].service_name = a.get('service_name', None)
            self.anchors[a['num']].address = a.get('address', None)
            self.anchors[a['num']].port = a.get('port', None)
            if a['service_name']:
                self.anchor_num_map[a['service_name']] = a['num']
            try:
                self.anchors[a['num']].vars = a['vars']
            except KeyError:
                pass
        cam = conf['camera_cal']
        self.resolution = tuple(cam['resolution'])
        self.intrinsic_matrix = np.array(cam['intrinsic_matrix'])
        self.distortion_coeff = np.array(cam['distortion_coeff'])
        self.commmon_anchor_vars = conf.get('common_anchor_vars', {})

        self.gripper.service_name = conf['gripper'].get('service_name', None)
        self.gripper.address = conf['gripper'].get('address', None)
        self.gripper.port = conf['gripper'].get('port', None)
        self.gripper.frame_room_spin = conf['gripper'].get('frame_room_spin', None)
        self.gripper.vars = conf['gripper'].get('vars', {})


    def write(self):
        outf = open(DEFAULT_CONFIG_PATH, 'w')
        conf = {
            'anchors': [
                {
                    'position': a.pose[1].tolist(),
                    'rotation': a.pose[0].tolist(),
                    'service_name': a.service_name,
                    'address': a.address,
                    'port': a.port,
                    'num': a.num,
                    'vars': a.vars,
                }
                for a in self.anchors],
            'camera_cal': {
                'resolution': self.resolution,
                'distortion_coeff': self.distortion_coeff.tolist(),
                'intrinsic_matrix': self.intrinsic_matrix.tolist(),
            },
            'common_anchor_vars': self.commmon_anchor_vars,
            'gripper': {
                'service_name': self.gripper.service_name,
                'address': self.gripper.address,
                'port': self.gripper.port,
                'frame_room_spin': self.gripper.frame_room_spin,
                'vars': self.gripper.vars,
            },
        }
        outf.write(json.dumps(conf, indent=2))
        outf.close()