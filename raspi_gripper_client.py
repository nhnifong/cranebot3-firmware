
from raspi_anchor_client import ComponentClient
import numpy as np
from scipy.spatial.transform import Rotation
from cv_common import compose_poses
import model_constants

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
            accel = update['imu']['accel'] # timestamp, x, y, z
            self.datastore.imu_accel.insert(np.array(accel, dtype=float))

            grip_pose = compose_poses([
                (Rotation.from_quat(update['imu']['quat']).as_rotvec(), np.array([0,0,0])),
                model_constants.gripper_imu,
            ])

            self.to_ui_q.put({'gripper_rvec': grip_pose[0]})

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        pass