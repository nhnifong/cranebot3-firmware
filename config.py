import json
import numpy as np

class Anchor:
    def __init__(self, num):
        self.num = num
        self.service_name = None
        self.pose = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))
        # config vars specific to each anchor, merged with common anchor vars and sent
        self.vars = {}

class Config:
    def __init__(self):
        self.anchors = [Anchor(i) for i in range(4)]
        self.anchor_num_map = {}
        # default configuration
        self.resolution = (4608, 2592)
        self.intrinsic_matrix = np.array(
            [[3382.661403509442, 0.0, 2327.7695374218133],
            [0.0, 3394.795601487364, 1267.8069498365248],
            [ 0.0, 0.0, 1.0 ]])
        self.distortion_coeff = np.array(
            [[0.021986, 0.160533, -0.003378, 0.002640, -0.356843]])
        self.commmon_anchor_vars = {}
        self.gripper_vars = {}
        try:
            self.reload()
        except FileNotFoundError:
            'No configuration.json file exists, using defaults'
            self.write()

    def vars_for_anchor(self, anchor_num):
        v = self.commmon_anchor_vars.copy()
        v.update(self.anchors[anchor_num].vars)
        return v

    def reload(self):
        conf = json.loads(open('configuration.json').read())
        self.anchor_num_map = {} # convenience map from service name to num
        for a in conf['anchors']:
            self.anchors[a['num']].pose = (np.array(a['rotation'], dtype=float), np.array(a['position'], dtype=float))
            self.anchors[a['num']].service_name = a['service_name']
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
        try:
            self.commmon_anchor_vars = conf['common_anchor_vars']
            self.gripper_vars = conf['gripper']
        except KeyError:
            pass

    def write(self):
        outf = open('configuration.json', 'w')
        conf = {
            'anchors': [
                {
                    'position': a.pose[1].tolist(),
                    'rotation': a.pose[0].tolist(),
                    'service_name': a.service_name,
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
            'gripper': self.gripper_vars,
        }
        outf.write(json.dumps(conf, indent=2))
        outf.close()