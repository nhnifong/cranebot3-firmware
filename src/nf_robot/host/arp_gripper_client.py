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

R_imu_to_cam = np.array([
    [-1, 0,  0],
    [0,  0, -1],
    [0, -1,  0]
])

# omega is the constant angular frequency of the 53cm pendulum.
OMEGA = np.sqrt(9.81 / 0.53)
SWING_CANCEL_GAIN = -0.4

def rotate_vector(vec, rad):
    """Rotates a 2D vector [x, y] by a given angle in radians."""
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    return np.array([
        vec[0] * cos_a - vec[1] * sin_a,
        vec[0] * sin_a + vec[1] * cos_a
    ])

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
        self.park_pose_relative_to_camera = None
        self.gripper_swing_model = np.zeros((2,2))
        self.swing_model_ts = time.time()

    async def handle_update_from_ws(self, update):
        if 'st' in update:
            self.swing_model_ts = float(update['st'])

        if 'sm' in update:
            self.gripper_swing_model = np.array(update['sm'])
            
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
            
            self.datastore.winch_line_record.insert([timestamp, wrist_angle, 0])
            self.datastore.finger.insert([timestamp, finger_angle, voltage])
            
            self.ob.send_ui(grip_sensors=telemetry.GripperSensors(
                range = distance_measurement,
                angle = finger_angle,
                pressure = voltage,
                wrist = wrist_angle,
            ))

    def compute_swing_correction(self, future_time):
        """Compute a corrective velocity to be applied at a future time in order to cancel the swing"""
        sm = self.gripper_swing_model
        st = self.swing_model_ts
        if sm is None or st is None:
            return None

        # we calculate the angle of the system at a future time.
        latency_comp = future_time - st
        look_ahead_angle = OMEGA * latency_comp
        c_future, s_future = np.cos(look_ahead_angle), np.sin(look_ahead_angle)
        
        # The angular acceleration (alpha) is the derivative of the velocity (gyro).
        # For this model, the derivative is omega * [-sin(theta), cos(theta)].
        future_accel = OMEGA * (sm[:, 1] * c_future - sm[:, 0] * s_future)

        # A corrective velocity to the gantry inversely proportional to the angular velocity of the gripper cancels the swing
        vel = future_accel * SWING_CANCEL_GAIN

        # rotate vector into room frame of reference
        wrist = self.datastore.winch_line_record.getLast()[1]
        imu_to_room_z = wrist / 180 * np.pi + self.config.gripper.frame_room_spin + np.pi/2
        return rotate_vector(vel, -imu_to_room_z)

    def handle_detections(self, detections, timestamp):
        """
        handle a list of tag detections from the pool
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
        return stabilize_frame(temp_image, gripper_quat, self.config.camera_cal_wide, R_imu_to_cam, roomspin,
            range_dist=range_to_object, cam_offset_mm=(0, 41.97), cam_tilt_deg=-4.67) # next model would be 4.67 degrees