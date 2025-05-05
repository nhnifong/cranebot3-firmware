
from raspi_anchor_client import ComponentClient
import numpy as np
from scipy.spatial.transform import Rotation
from cv_common import compose_poses
import model_constants
from config import Config
import json 

# number of origin detections to average
video_port = 8888
websocket_port = 8765

class RaspiGripperClient(ComponentClient):
    def __init__(self, address, datastore, to_ui_q, to_pe_q, to_ob_q, pool, stat):
        super().__init__(address, datastore, to_ui_q, to_pe_q, to_ob_q, pool, stat)
        self.conn_status = {'gripper': True}
        self.anchor_num = None

    def handle_update_from_ws(self, update):
        if 'line_record' in update:
            self.datastore.winch_line_record.insertList(update['line_record'])

        if 'imu' in update:
            timestamp = update['imu']['quat'][0]
            grip_pose = compose_poses([
                (Rotation.from_quat(update['imu']['quat'][1:]).as_rotvec(), np.array([0,0,0])),
                model_constants.gripper_imu,
            ])
            self.datastore.imu_rotvec.insert(np.concatenate([np.array([timestamp], dtype=float), grip_pose[0]]))

        if 'range' in update:
            # expect a tuple of (time, distance)
            distance_measurement = update['range']
            self.datastore.range_record.insert(distance_measurement)
            
        if 'holding' in update:
            # expect a bool. Forward it to the position estimator
            holding = update['holding'] is True
            self.to_pe_q.put({'holding': holding})

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        pass

    async def send_config(self):
        config = Config()
        if len(config.gripper_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': config.gripper_vars}))