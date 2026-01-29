import asyncio
import numpy as np
from scipy.spatial.transform import Rotation
import json
import cv2

from nf_robot.host.anchor_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.cv_common import SF_INPUT_SHAPE, stabilize_frame

class RaspiGripperClient(ComponentClient):
    def __init__(self, address, port, datastore, ob, pool, stat, pe, local_telemetry):
        super().__init__(address, port, datastore, ob, pool, stat, local_telemetry)
        self.conn_status = telemetry.ComponentConnStatus(
            is_gripper=True,
            websocket_status=telemetry.ConnStatus.NOT_DETECTED,
            video_status=telemetry.ConnStatus.NOT_DETECTED,
        )
        self.anchor_num = None
        self.pe = pe

    async def handle_update_from_ws(self, update):
        if 'line_record' in update:
            self.datastore.winch_line_record.insertList(update['line_record'])

        if 'grip_sensors' in update:
            gs = update['grip_sensors']
            timestamp = gs['time']
            self.datastore.imu_quat.insert(np.concatenate([np.array([timestamp], dtype=float), gs['quat']]))

            distance_measurement = 0
            if 'range' in gs:
                distance_measurement = float(gs['range'])
                self.datastore.range_record.insert([timestamp, distance_measurement])


            # Note that finger angles are returned in the range of (-90, 90)
            # this is because that's the range we use when talking to the inventor hat mini.
            # fully open is about -90 and fully closed is about 80
            # the actual servo installed in the Pilot hardware is a 270 degree servo
            # and it is connected to the fingers with a reduction gear.

            angle = float(gs['fing_a'])
            voltage = float(gs['fing_v'])

            self.datastore.finger.insert([timestamp, angle, voltage])
            self.ob.send_ui(grip_sensors=telemetry.GripperSensors(
                range = distance_measurement,
                angle = angle,
                pressure = voltage,
            ))
            
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

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1

    async def send_config(self):
        pass

    async def zero_winch(self):
        """Send the command to zero the winch line and wait for it to complete"""
        print('Zero Winch Line')
        self.winch_zero_event = asyncio.Event()
        await self.send_commands({'zero_winch_line': None})
        await asyncio.wait_for(self.winch_zero_event.wait(), timeout=20)

    def process_frame(self, frame_to_encode):
        # stabilize and resize for centering network input
        temp_image = cv2.resize(frame_to_encode, SF_INPUT_SHAPE, interpolation=cv2.INTER_AREA)
        fudge_latency =  0.3
        try:
            gripper_quat = self.datastore.imu_quat.getClosest(self.last_frame_cap_time - fudge_latency)[1:]
        except IndexError:
            gripper_quat = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat()
        if self.calibrating_room_spin or self.config.gripper.frame_room_spin is None:
            # roomspin = 15/180*np.pi
            roomspin = 0
        else:
            roomspin = self.config.gripper.frame_room_spin
        range_to_object = self.datastore.range_record.getLast()[1]
        return stabilize_frame(temp_image, gripper_quat, self.config.camera_cal, roomspin, range_dist=range_to_object)
