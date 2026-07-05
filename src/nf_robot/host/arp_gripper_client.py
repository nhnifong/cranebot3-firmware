import asyncio
import logging
import numpy as np
from scipy.spatial.transform import Rotation
import json
import cv2
import time
import math

from nf_robot.host.anchor_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.cv_common import SF_TARGET_SHAPE, stabilize_frame_2, OTHER_MARKERS, CAL_MARKERS

logger = logging.getLogger(__name__)

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
    [1, 0,  0],
    [0,  -1, 0],
    [0,  0,  1]
])

# omega is the constant angular frequency of the pendulum. Effectuve length from pivot to center of gripper mass is 0.4526 meters
LENGTH = 0.4526
OMEGA = np.sqrt(9.81 / LENGTH)
SWING_CANCEL_GAIN = -0.12
CENTERING_GAIN = 0.4

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
            gripper_model=telemetry.GripperModel.ARPEGGIO,
        )
        self.anchor_num = None
        self.pe = pe
        self.park_pose_relative_to_camera = None
        # latest (rotvec, position) pose of each route-point tag relative to the gripper
        # camera, keyed by tag name. Reset every frame so a present entry means in-frame now.
        self.route_tag_poses_relative_to_camera = {}
        self.gripper_swing_model = np.zeros((2,2))
        self.swing_model_ts = time.time()
        self.finger_contact_calibration_complete = asyncio.Event()
        # set when the gripper replies to a query_angle_from_vertical request
        self.angle_from_vertical_received = asyncio.Event()
        self.last_angle_from_vertical = None
        
        # State variables added to track and prevent platform drift
        self._swing_position_offset = np.zeros(2)
        self._last_future_time = 0

        # State for looking in direction of motion
        self.smoothed_error = 0.0
        self.ema_alpha = 0.3  # Smoothing factor (0 to 1)
        self.deadband = 0.02  # Radians (~1.1 degrees)
        self.p_gain = 2.0     # Proportional gain for speed calculation

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

            distance_measurement = self.datastore.range_record.getLast()[1]
            if 'range' in gs:
                distance_measurement = float(gs['range'])
                self.datastore.range_record.insert([timestamp, distance_measurement])

            if 'raw_accel' in gs:
                logger.debug(f"raw_accel: {gs['raw_accel']}")

            if 'vel_from_imu' in gs:
                self.vel_from_imu = np.array(gs['vel_from_imu'])

            target_force = 0
            if 'dforce' in gs:
                target_force = float(gs['dforce'])

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
                target_force = target_force,
            ))

        if 'finger_contact_calibration_complete' in update:
            self.finger_contact_calibration_complete.set()

        if 'angle_from_vertical' in update:
            self.last_angle_from_vertical = float(update['angle_from_vertical'])
            self.angle_from_vertical_received.set()

    async def query_angle_from_vertical(self, timeout=2.0):
        """Ask the gripper for a one-shot reading of how many degrees its pole is
        tilted from vertical (from the accelerometer) and return it. Returns None
        if the gripper does not reply within `timeout` seconds."""
        self.angle_from_vertical_received.clear()
        await self.send_commands({'query_angle_from_vertical': None})
        try:
            await asyncio.wait_for(self.angle_from_vertical_received.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning('Timed out waiting for angle_from_vertical reply from gripper')
            return None
        return self.last_angle_from_vertical

    def compute_swing_correction(self, future_time):
        """Compute a corrective velocity to be applied at a future time in order to cancel the swing"""
        sm = self.gripper_swing_model
        st = self.swing_model_ts
        if sm is None or st is None:
            return None

        # calculate swing cancellation
        latency_comp = future_time - st
        look_ahead_angle = OMEGA * latency_comp
        c_future, s_future = np.cos(look_ahead_angle), np.sin(look_ahead_angle)
        
        # The angular acceleration (alpha) is the derivative of the velocity (gyro).
        # For this model, the derivative is omega * [-sin(theta), cos(theta)].
        future_accel = OMEGA * (sm[:, 1] * c_future - sm[:, 0] * s_future)

        # A corrective velocity to the gantry inversely proportional to the angular velocity of the gripper cancels the swing
        raw_vel = future_accel * SWING_CANCEL_GAIN

        # cancel accumulated drift introduced from swing cancellation
        # Calculate time elapsed since last call to update the integrator
        dt = future_time - self._last_future_time
        self._last_future_time = future_time

        # Ignore massive jumps in time if the control loop paused
        if dt > 0.5 or dt < 0:
            dt = 0.0

        # Apply a centering restorative velocity proportional to the accumulated position offset
        centering_vel = self._swing_position_offset * CENTERING_GAIN
        vel = raw_vel - centering_vel

        # Track the accumulated position offset based on the velocity we are actually commanding
        self._swing_position_offset += vel * dt

        # rotate vector into room frame of reference
        wrist = self.datastore.winch_line_record.getLast()[1]
        imu_to_room_z = wrist / 180 * np.pi + self.config.gripper.frame_room_spin - np.pi/2
        return rotate_vector(vel, -imu_to_room_z)

    def handle_detections(self, detections, timestamp):
        """
        handle a list of tag detections from the pool
        """
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)
        # setting to none every frame so we know whether it's in frame by looking at this variable
        self.park_pose_relative_to_camera = None
        self.route_tag_poses_relative_to_camera = {}

        for detection in detections:
            name = detection['n']
            self.last_known_centers[name] = detection['center']

            if name == 'park_target':
                # pose of parking target relative to gripper camera
                self.park_pose_relative_to_camera = detection['p']
            elif name in OTHER_MARKERS or name in CAL_MARKERS:
                # (rotvec, position) of a route-point tag relative to the gripper camera.
                # CAL_MARKERS (origin + cal_assist_*) are included so that the end-of-calibration
                # gripper card survey (collect_gripper_card_observations) can read where each
                # calibration card sits relative to the gripper camera. These poses are in the
                # raw (unstabilized, tilted) camera optical frame, not the room frame.
                self.route_tag_poses_relative_to_camera[name] = detection['p']

    async def send_config(self):
        pass

    def get_gripper_rvec(self, timestamp=None):
        """
        Calculates the rotation of the gripper in its local frame of reference
        at a specific timestamp, not counting the wrist.
        """
        if timestamp is None:
            projected_state = self.gripper_swing_model
        else:
            # Calculate how much the phase has evolved between the model's last update and the requested timestamp.
            dt = timestamp - self.swing_model_ts
            angle = OMEGA * dt
            c, s = np.cos(angle), np.sin(angle)
            # Project the state matrix to the target timestamp using a rotation matrix.
            # This allows us to find the A*sin and A*cos components at that exact moment.
            projected_state = self.gripper_swing_model @ np.array([[c, -s], [s, c]])
        
        # In a harmonic oscillator, displacement is the integral of velocity.
        # For a model where Col 0 is Velocity (A*sin), the displacement is -A/omega * cos.
        # This corresponds to the negative of the phase tracker (Col 1) divided by omega.
        theta_x = projected_state[0, 1] / OMEGA
        theta_y = projected_state[1, 1] / OMEGA
        return np.array([theta_x, theta_y, 0])

    def get_swing_amplitude(self):
        """Return the current angular amplitude of the gripper's swing, in radians.
        
        This is a phase-independent measure of "how much it is swinging" that can
        be read at any instant without watching for peaks over a full period.
        Returns 0.0 when there is no swing (or no IMU populating the model).
        """
        sm = self.gripper_swing_model
        if sm is None:
            return 0.0
        return float(np.linalg.norm(sm) / OMEGA)

    def get_spin(self, debug=False):
        # return the rotation of the gripper camera relative to the room in radians
        roomspin = self.datastore.winch_line_record.getLast()[1] / 180 * np.pi
        if not self.calibrating_room_spin and self.config.gripper.frame_room_spin is not None:
            # undo the rotation that the room would appear to have at the wrist's 540 position
            extra = self.config.gripper.frame_room_spin - np.pi
            if debug:
                print(f'gripper spin should be wrist {roomspin} plus extra spin from config {extra}')
            roomspin = roomspin + extra
        return roomspin

    def gripper_body_room_rotation(self):
        """Rotation taking a vector in the z-up gripper body frame (the frame the position
        estimator uses: pole pointing down -z, x/y horizontal) to the room frame, right now.

        Two parts:
        * heading about the room's vertical axis, straight from get_spin(). An earlier version
          added a +pi base here; on-robot data (the visual-centering nudge diverging by a factor
          of ~1+gain, plus the reconstructed gantry landing ~180 deg opposite the true offset for
          off-center cards) showed that base was wrong -- get_spin() already carries the correct
          heading. The +pi was invisible while cards happened to sit near the image center.
        * the small swing tilt from get_gripper_rvec (z-component always 0), the same rotvec
          the position estimator applies to the pole; applied inside the heading.
        """
        R_heading = Rotation.from_rotvec([0.0, 0.0, self.get_spin()])
        R_tilt = Rotation.from_rotvec(self.get_gripper_rvec())
        return R_heading * R_tilt

    def measure_gantry_minus_card(self, pose_cam):
        """Given a calibration card's pose in the raw gripper camera optical frame (rvec, tvec,
        as stored in route_tag_poses_relative_to_camera), return the measured room-frame vector
        from the card to the gantry point (i.e. gantry_position - card_position).

        The gantry is the point where the four lines converge, at the top of the rigid pole; the
        camera hangs a fixed distance below it. Because both the camera offset and the observed
        card are rigidly tied to the gripper body, the gantry's absolute room position cancels
        and this vector depends only on the body orientation and the observed card pose. That is
        exactly the close-range, accurate measurement the calibration anchors on: the caller adds
        the card's known room position to recover the gantry position at that hover.

        Frame chain (see model_constants and the position estimator's grip_pose):
        * model_constants.gripper_camera places the card in the CAD 'gripper frame', which is
          y-up (grommet at +y, optical axis looking down -y).
        * Rx(90 deg) re-expresses that in the z-up body frame the rest of the system uses.
        * the gantry sits arp_pole_length up the +z body axis from the gripper origin.
        * gripper_body_room_rotation() rotates the body frame into the room.

        NOTE: this open-loop camera->room conversion is new and has not been validated on
        hardware; collect_gripper_card_observations logs the reconstructed gantry position next
        to the position estimator's live gant_pos so the convention (heading base, Rx sign) can
        be checked and corrected on the robot.
        """
        # card position in the CAD y-up gripper frame
        card_in_gripper = compose_poses([model_constants.gripper_camera, pose_cam])[1]
        # re-express in the z-up body frame, then measure relative to the gantry (pole up +z)
        y_up_to_z_up = Rotation.from_euler('x', 90, degrees=True)
        card_in_body = y_up_to_z_up.apply(card_in_gripper) - np.array([0.0, 0.0, model_constants.arp_pole_length])
        # rotate the card-relative-to-gantry vector into the room, then negate for gantry-card
        card_minus_gantry_room = self.gripper_body_room_rotation().apply(card_in_body)
        return -card_minus_gantry_room

    def look_towards_vector(self, vec2):
        """
        Turn the head to face in the direction of the given XY vector in room space.
        vec2: A numpy array [x, y]
        """
        # Calculate target angle from vector
        target_angle_base = math.atan2(vec2[0], vec2[1]) # Result in (-pi, pi]

        # Spin ranges from 0 to 6*pi. Nose @ +Y is spin % 2pi == 0.
        current_spin = self.get_spin()
        
        # Determine the best target within the [0, 6*pi] range
        # There are 3 possible rotations that face the same direction:
        # base_angle (normalized to [0, 2pi]), base_angle + 2pi, and base_angle + 4pi.
        norm_target = target_angle_base % (2 * math.pi)
        candidates = [norm_target, norm_target + 2 * math.pi, norm_target + 4 * math.pi]
        
        # Determine proximity to bounds
        lower_bound = 0.5 * math.pi
        upper_bound = 5.5 * math.pi
        center_point = 3 * math.pi

        if current_spin < lower_bound:
            # Near lower limit: Force selection of a candidate that moves us toward center
            # Typically picking the candidate > current_spin
            target = min([c for c in candidates if c > current_spin] or [candidates[-1]])
        elif current_spin > upper_bound:
            # Near upper limit: Force selection of a candidate that moves us toward center
            target = max([c for c in candidates if c < current_spin] or [candidates[0]])
        else:
            # Normal operation: Pick the closest candidate
            target = min(candidates, key=lambda c: abs(c - current_spin))

        # Calculate raw error
        raw_error = target - current_spin

        # Apply Deadband
        if abs(raw_error) < self.deadband:
            raw_error = 0.0

        # Exponential Moving Average (EMA) Smoothing
        # smoothed = alpha * new + (1 - alpha) * old
        self.smoothed_error = (self.ema_alpha * raw_error) + (1.0 - self.ema_alpha) * self.smoothed_error

        # Convert Error to Speed (Degrees per Second)
        # Convert radians to degrees: radians * (180 / pi)
        # Apply a proportional gain
        wrist_speed_deg = self.smoothed_error * self.p_gain * (180.0 / math.pi)

        # 8. Clamp and Send
        wrist_speed = clamp(wrist_speed_deg, -120, 120)
        asyncio.create_task(self.send_commands({'set_wrist_speed': wrist_speed}))

    def process_frame(self, frame_to_encode):
        # an action space in which the gripper camera is not stabilized or rotated.
        # no matter what perspective the operator is driving with, the network is always seeing
        # control inputs relative to the gripper image. it will see a +Y direction when the motion is up in
        # the gripper image.
        input_shape = (frame_to_encode.shape[1], frame_to_encode.shape[0])
        if input_shape != SF_TARGET_SHAPE:
            temp_image = cv2.resize(frame_to_encode, SF_TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        else:
            temp_image = frame_to_encode
        return temp_image