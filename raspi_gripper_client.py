import asyncio
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

        if 'grip_sensors' in update:
            gs = update['grip_sensors']
            timestamp = gs['time']
            grip_pose = compose_poses([
                (Rotation.from_quat(gs['quat']).as_rotvec(), np.array([0,0,0])),
                model_constants.gripper_imu,
            ])
            self.datastore.imu_rotvec.insert(np.concatenate([np.array([timestamp], dtype=float), grip_pose[0]]))

            if 'range' in gs:
                distance_measurement = float(gs['range'])
                self.datastore.range_record.insert([timestamp, distance_measurement])

            self.datastore.finger.insert([timestamp, float(gs['fing_a']), float(gs['fing_v'])])
            
        if 'holding' in update:
            # expect a bool. Forward it to the position estimator
            holding = update['holding'] is True
            self.pe.notify_update({'holding': holding})

        if 'winch_zero_success' in update:
            print(f'winch_zero_success = {update["winch_zero_success"]}')
            if update['winch_zero_success']:
                self.winch_zero_event.set()

        if 'episode_button_pushed' in update:
            print('episode_button_pushed')
            self.to_ob_q.put({'episode': True})

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1

    async def send_config(self):
        config = Config()
        if len(config.gripper_vars) > 0:
            await self.websocket.send(json.dumps({'set_config_vars': config.gripper_vars}))

    async def zero_winch(self):
        """Send the command to zero the winch line and wait for it to complete"""
        print('Zero Winch Line')
        self.winch_zero_event = asyncio.Event()
        await self.send_commands({'zero_winch_line': None})
        await asyncio.wait_for(self.winch_zero_event.wait(), timeout=20)