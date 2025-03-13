
from raspi_anchor_client import ComponentClient

# number of origin detections to average
video_port = 8888
websocket_port = 8765

class RaspiGripperClient(ComponentClient):
    def __init__(self, address, datastore, to_ui_q, to_pe_q, pool, stat):
        super().__init__(address, datastore, to_ui_q, to_pe_q, pool, stat, None)
        self.conn_status = {'gripper': True}

    def handle_update_from_ws(self, update):
        if 'line_record' in update:
            self.datastore.winch_line_record.insertList(update['line_record'])

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        pass