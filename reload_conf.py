import json
import numpy as np

class Anchor:
    def __init__(self, num):
        self.num = num
        self.service_name = None
        self.pose = (np.array([0,0,0], dtype=float), np.array([0,0,0], dtype=float))

class Config:
    def __init__(self):
        self.anchors = [Anchor(i) for i in range(4)]
        # default configuration
        self.resolution = (4608, 2592)
        self.intrinsic_matrix = np.array(
            [[3382.661403509442, 0.0, 2327.7695374218133],
            [0.0, 3394.795601487364, 1267.8069498365248],
            [ 0.0, 0.0, 1.0 ]])
        self.distortion_coeff = np.array(
            [[0.021986, 0.160533, -0.003378, 0.002640, -0.356843]])
        try:
            self.reload()
        except FileNotFoundError:
            'No configuration.json file exists, using defaults'
            self.write()

    def reload(self):
        conf = json.loads(open('configuration.json').read())
        for a in conf['anchors']:
            self.anchors[a['num']].pose = (np.array(a['rotation'], dtype=float), np.array(a['position'], dtype=float))
            self.anchors[a['num']].service_name = a['service_name']
        cam = conf['camera_cal']
        self.resolution = tuple(cam['resolution'])
        self.intrinsic_matrix = np.array(cam['intrinsic_matrix'])
        self.distortion_coeff = np.array(cam['distortion_coeff'])

    def write(self):
        outf = open('configuration.json', 'w')
        conf = {
            'anchors': [
                {
                    'position': a.pose[1].tolist(),
                    'rotation': a.pose[0].tolist(),
                    'service_name': a.service_name,
                    'num': a.num,
                }
                for a in self.anchors],
            'camera_cal': {
                'resolution': self.resolution,
                'distortion_coeff': self.distortion_coeff.tolist(),
                'intrinsic_matrix': self.intrinsic_matrix.tolist(),
            }
        }
        outf.write(json.dumps(conf, indent=2))
        outf.close()