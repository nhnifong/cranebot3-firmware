
from raspi_anchor_client import ComponentClient

# number of origin detections to average
video_port = 8888
websocket_port = 8765

class RaspiGripperClient(ComponentClient):
    def __init__(self, address, datastore, to_ui_q, to_pe_q, pool, stat):
        super().__init__(address, datastore, to_ui_q, to_pe_q, pool, stat)
        self.conn_status = {'gripper': True}
        self.anchor_num = None

    def handle_update_from_ws(self, update):
        if 'line_record' in update:
            self.datastore.winch_line_record.insertList(update['line_record'])
        if 'imu' in update:
            accel = update['imu']['accel'] # timestamp, x, y, z
            self.datastore.imu_accel.insert(np.array(accel, dtype=float))
            self.to_ui_q({'gripper_quat': update['imu']['quat']})

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        pass