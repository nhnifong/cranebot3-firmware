import asyncio
import numpy as np
from scipy.spatial.transform import Rotation
import json

from nf_robot.host.anchor_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, common

"""
"Arpeggio" is the codename of the 2nd revision of the Stringman gripper

It differs from the previous gripper in that it has a wrist instead of a winch.
Since it uses smart servos it can report the exact angle of either the fingers or wrist
It does not send 'line records' because there is no changing length of line, but wherever line
records were being used as a heartbeat signal, the grip sensors can be used instead.

It has a wide angle camera instead of standard, and the camera is pointed inward at a point 1m below the gripper

The gripper and gantry are now one model, with the gripper's origin being 57cm below the gantry's.
They are related by a chain of poses from the gantry tags, through the wrist rotation, 

"""

class ArpeggioGripperClient(ComponentClient):
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
        if 'grip_sensors' in update:
            gs = update['grip_sensors']
            timestamp = gs['time']

            # rotation of gripper as quaternion. not present if IMU not installed.
            if 'quat' in gs:
                self.datastore.imu_quat.insert(np.concatenate([np.array([timestamp], dtype=float), gs['quat']]))

            distance_measurement = 0
            if 'range' in gs:
                distance_measurement = float(gs['range'])
                self.datastore.range_record.insert([timestamp, distance_measurement])

            # Note that finger angles are returned in the range of (-90, 90) even though these are not the actual angle
            # -90 is open
            finger_angle = float(gs['fing_a'])

            # finger pad pressure is indicated by this voltage with 3.3 being no pressure.
            # lower values indicate more pressure.
            voltage = float(gs['fing_v'])

            # wrist angle in degrees of rotation from the original zero point. can be more than one revolution.
            # the zero point is probably a safe bet for where the wire would be least twisted.
            # the angle at which it aligns with the gantry or the room must be determined in calibration
            wrist_angle = float(gs['wrist_a'])

            self.datastore.winch_line_record.insertList([timestamp, wrist_angle, 0])
            self.datastore.finger.insert([timestamp, finger_angle, voltage])
            
            self.ob.send_ui(grip_sensors=telemetry.GripperSensors(
                range = distance_measurement,
                angle = finger_angle,
                pressure = voltage,
                wrist = wrist_angle,
            ))

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1

    async def send_config(self):
        pass

    def process_frame(self, frame_to_encode):
        # stabilize and resize for centering network input
        temp_image = cv2.resize(frame_to_encode, SF_INPUT_SHAPE, interpolation=cv2.INTER_AREA)
        fudge_latency =  0.3
        try:
            gripper_quat = self.datastore.imu_quat.getClosest(self.last_frame_cap_time - fudge_latency)[1:]
        except IndexError:
            gripper_quat = Rotation.from_euler('xyz', [-90, 0, 0], degrees=True).as_quat()

        # how should we spin the frame around the room z axis so that room +Y is up
        if self.calibrating_room_spin or self.config.gripper.frame_room_spin is None:
            # roomspin = 15/180*np.pi
            roomspin = 0
        else:
            # undo the rotation added by the wrist joint
            wrist = self.datastore.winch_line_record.getClosest(self.last_frame_cap_time - fudge_latency)[1]
            roomspin = wrist / 180 * np.pi
            # then undro the rotation that the room would appear to have at the wrist's zero position
            roomspin += self.config.gripper.frame_room_spin

        range_to_object = self.datastore.range_record.getLast()[1]
        return stabilize_frame(temp_image, gripper_quat, self.config.camera_cal, roomspin, range_dist=range_to_object)