
from raspi_anchor_client import ComponentClient
import numpy as np
from scipy.spatial.transform import Rotation
from cv_common import compose_poses
import model_constants
from config import Config
import json 

class RaspiGripperClient(ComponentClient):
    def __init__(self, address, port, datastore, to_ui_q, to_ob_q, pool, stat, pe):
        super().__init__(address, port, datastore, to_ui_q, to_ob_q, pool, stat)
        self.conn_status = {'gripper': True}
        self.anchor_num = None
        self.pe = pe

    def handle_update_from_ws(self, update):
        if 'line_record' in update:
            self.datastore.winch_line_record.insertList(update['line_record'])

        if 'imu' in update:
            timestamp = update['imu']['time']
            grip_pose = compose_poses([
                (Rotation.from_quat(update['imu']['quat']).as_rotvec(), np.array([0,0,0])),
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
            self.pe.notify_update({'holding': holding})

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1

    async def send_config(self):
        config = Config()
        if len(config.gripper_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': config.gripper_vars}))