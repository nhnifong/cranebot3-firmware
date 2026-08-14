import asyncio
import logging
from collections import defaultdict, deque
import numpy as np
from scipy.spatial.transform import Rotation
import json
import cv2
import time
import math

from nf_robot.host.component_client import ComponentClient
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, common
from nf_robot.common.cv_common import SF_TARGET_SHAPE, OTHER_MARKERS, CAL_MARKERS
from nf_robot.robot.component_server import stream_modes

logger = logging.getLogger(__name__)

"""Host-side client for "Arpeggio", the 2nd revision of the Stringman gripper.

A wrist instead of a winch, smart servos that report exact finger and wrist angles, and a
wide angle camera aimed 1m below the gripper. It sends no line records; grip sensor
messages are the heartbeat instead. Gripper and gantry are one model whose origins are
57cm apart, related by a chain of poses from the gantry tags through the wrist rotation.
"""

R_imu_to_cam = np.array([
    [1, 0,  0],
    [0,  -1, 0],
    [0,  0,  1]
])

# pivot to center of gripper mass, which sets the pendulum's angular frequency
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

# How long a route/cal tag sighting stays usable: a full-scan interval plus detection
# latency, so a run of frames that miss the tag doesn't blank its pose.
ROUTE_TAG_MAX_AGE_S = 0.6
# ~8s of sightings at a 30Hz detection rate, the longest window any caller asks for
ROUTE_TAG_HISTORY = 240

# The camera's two modes, by the names the gripper knows them by. Resolution, bitrate and
# framerate live in the robot side's stream_modes table, where they have to be: they are
# only meaningful next to a measured dts_zero_offset. Imported from component_server
# rather than gripper_arp_server, which pulls in i2c and servo libraries at import.
CONTROL_STREAM_MODE = 'gripper_control'
CAPTURE_STREAM_MODE = 'gripper_capture'
# (width, height) capture mode delivers, so a caller can tell which mode a frame came from
CAPTURE_RESOLUTION_SIZE = (stream_modes[CAPTURE_STREAM_MODE].width,
                           stream_modes[CAPTURE_STREAM_MODE].height)


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
        # tag name -> (capture timestamp, (rotvec, position)) relative to the gripper
        # camera, oldest first. Each detection is appended once, so a window read out of
        # here counts each sighting once.
        self.route_tag_samples = defaultdict(lambda: deque(maxlen=ROUTE_TAG_HISTORY))
        self.gripper_swing_model = np.zeros((2,2))
        self.swing_model_ts = time.time()
        self.finger_contact_calibration_complete = asyncio.Event()
        # set when the gripper replies to a query_angle_from_vertical request
        self.angle_from_vertical_received = asyncio.Event()
        self.last_angle_from_vertical = None
        
        # integrated drift from swing cancellation, which compute_swing_correction
        # subtracts back out so the platform holds its place
        self._swing_position_offset = np.zeros(2)
        self._last_future_time = 0

        # look_towards_vector's controller
        self.smoothed_error = 0.0
        self.ema_alpha = 0.3
        self.deadband = 0.02  # radians, ~1.1 degrees
        self.p_gain = 2.0

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

            # (-90, 90), not the true angle; -90 is open
            finger_angle = float(gs['fing_a'])

            # finger pad pressure, inverted: 3.3V is no pressure, lower is more
            voltage = float(gs['fing_v'])

            # degrees from the servo's zero point, possibly past a full revolution. Zero is
            # where the wire is least twisted; how it aligns with the gantry or the room
            # comes out of calibration.
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
        """Degrees the pole is tilted from vertical, from the accelerometer, or None if
        the gripper does not reply within `timeout`."""
        self.angle_from_vertical_received.clear()
        await self.send_commands({'query_angle_from_vertical': None})
        try:
            await asyncio.wait_for(self.angle_from_vertical_received.wait(), timeout)
        except asyncio.TimeoutError:
            logger.warning('Timed out waiting for angle_from_vertical reply from gripper')
            return None
        return self.last_angle_from_vertical

    def compute_swing_correction(self, future_time):
        """Room-frame gantry velocity that will cancel the gripper's swing at future_time.

        The model is projected forward to future_time rather than used as-is, which is
        what compensates for control latency.
        """
        sm = self.gripper_swing_model
        st = self.swing_model_ts
        if sm is None or st is None:
            return None

        latency_comp = future_time - st
        look_ahead_angle = OMEGA * latency_comp
        c_future, s_future = np.cos(look_ahead_angle), np.sin(look_ahead_angle)

        # angular acceleration is the derivative of the gyro velocity, which for this
        # model is omega * [-sin(theta), cos(theta)]
        future_accel = OMEGA * (sm[:, 1] * c_future - sm[:, 0] * s_future)

        # a gantry velocity opposing the gripper's angular velocity cancels the swing
        raw_vel = future_accel * SWING_CANCEL_GAIN

        dt = future_time - self._last_future_time
        self._last_future_time = future_time

        # a paused control loop leaves a huge dt that would wreck the integrator
        if dt > 0.5 or dt < 0:
            dt = 0.0

        # cancelling swing drifts the platform, so pull back toward where it started
        centering_vel = self._swing_position_offset * CENTERING_GAIN
        vel = raw_vel - centering_vel
        self._swing_position_offset += vel * dt

        wrist = self.datastore.winch_line_record.getLast()[1]
        imu_to_room_z = wrist / 180 * np.pi + self.config.gripper.frame_room_spin - np.pi/2
        return rotate_vector(vel, -imu_to_room_z)

    def handle_detections(self, detections, timestamp):
        """File one frame's tag detections, called back from the detector pool."""
        self.stat.pending_frames_in_pool -= 1
        self.stat.detection_count += len(detections)
        # cleared every frame, so this doubles as "is the park target in view"
        self.park_pose_relative_to_camera = None

        for detection in detections:
            name = detection['n']
            self.last_known_centers[name] = detection['center']
            self.last_known_half_extents[name] = detection.get('half_extent')

            if name == 'park_target':
                self.park_pose_relative_to_camera = detection['p']
            elif name in OTHER_MARKERS or name in CAL_MARKERS:
                # poses stay in the raw (unstabilized, tilted) camera optical frame, with
                # their capture time; readers apply their own staleness bound. CAL_MARKERS
                # are here for the gripper card survey.
                self.route_tag_samples[name].append((timestamp, detection['p']))

    def get_route_tag_pose(self, name, max_age_s=ROUTE_TAG_MAX_AGE_S):
        """Latest (rotvec, position) of a route-point or calibration tag relative to the
        gripper camera, or None if it has not been seen within max_age_s.

        Ages run from the frame's capture time, so they include streaming and detection
        latency, not just the gap since the last sighting.
        """
        samples = self.route_tag_samples.get(name)
        if not samples:
            return None
        seen_at, pose = samples[-1]
        if time.time() - seen_at > max_age_s:
            return None
        return pose

    def get_route_tag_samples(self, name, since=None, max_age_s=None):
        """Buffered (capture timestamp, pose) sightings of a tag, oldest first, bounded by
        since / max_age_s.

        Copies the deque before filtering: detections are appended from a pool callback
        thread.
        """
        samples = list(self.route_tag_samples.get(name, ()))
        if since is not None:
            samples = [s for s in samples if s[0] >= since]
        if max_age_s is not None:
            cutoff = time.time() - max_age_s
            samples = [s for s in samples if s[0] >= cutoff]
        return samples

    async def send_config(self):
        pass

    def get_gripper_rvec(self, timestamp=None):
        """Tilt of the gripper in its own frame, wrist rotation excluded. timestamp reads
        it at that moment instead of now."""
        if timestamp is None:
            projected_state = self.gripper_swing_model
        else:
            # rotate the state matrix by however far the pendulum's phase has evolved
            # since the model was last updated, giving A*sin and A*cos at that instant
            dt = timestamp - self.swing_model_ts
            angle = OMEGA * dt
            c, s = np.cos(angle), np.sin(angle)
            projected_state = self.gripper_swing_model @ np.array([[c, -s], [s, c]])

        # displacement is the integral of velocity, so with col 0 velocity (A*sin) it is
        # -A/omega*cos: the phase tracker in col 1, over omega
        theta_x = projected_state[0, 1] / OMEGA
        theta_y = projected_state[1, 1] / OMEGA
        return np.array([theta_x, theta_y, 0])

    def get_swing_amplitude(self):
        """Angular amplitude of the swing in radians, 0.0 if there is none (or no IMU).

        Phase independent, so it can be read at any instant rather than by watching for a
        peak over a full period.
        """
        sm = self.gripper_swing_model
        if sm is None:
            return 0.0
        return float(np.linalg.norm(sm) / OMEGA)

    async def use_capture_stream(self):
        """Switch the gripper camera to stills-capture mode: higher res, lower fps.

        For the synthetic dataset's plates, which want detail and don't care about
        latency. The low framerate comes with the mode - a pi zero 2w cannot encode this
        resolution at streaming rates - and the captures hold still at each stop anyway.
        Both modes share a sensor mode, so the field of view is unchanged and captures
        stay geometrically comparable to the control stream.

        Costs a stream restart, so frames stop for a second or two. Re-selecting the
        running mode is a no-op, and the mode holds until something asks for the control
        stream back.
        """
        await self.send_commands({'set_config_vars': {'STREAM_MODE': CAPTURE_STREAM_MODE}})

    async def capture_raw_frame(self, after_ts, timeout=5.0, expect_size=None):
        """The newest camera frame captured after after_ts, as (timestamp, RGB array), or
        (None, None) on timeout.

        Reads self.frame, not last_output_frame, which process_frame has already resized
        to SF_TARGET_SHAPE - throwing away the pixels capture mode exists to get. Waiting
        on capture time rather than sleeping guarantees the frame is from after whatever
        motion the caller just commanded, however far the stream is lagging.

        expect_size (width, height) is the only reliable test that a resolution change has
        landed: the old stream keeps delivering for seconds afterwards, and those frames
        are new enough to pass any timestamp test at the wrong size.
        """
        deadline = time.time() + timeout
        while True:
            with self.frame_lock:
                timestamp, frame = self.last_frame_cap_time, self.frame
                if frame is not None and timestamp is not None and timestamp > after_ts:
                    if expect_size is None or (frame.shape[1], frame.shape[0]) == tuple(expect_size):
                        return timestamp, frame.copy()
            if time.time() > deadline:
                return None, None
            await asyncio.sleep(0.02)

    async def restore_default_stream(self):
        """Put the camera back to the control stream."""
        await self.send_commands({'set_config_vars': {'STREAM_MODE': CONTROL_STREAM_MODE}})

    def get_spin(self, debug=False, timestamp=None):
        # Rotation of the gripper camera relative to the room, in radians. timestamp picks
        # the wrist reading nearest that capture time, for measurements made after the fact.
        if timestamp is None:
            wrist = self.datastore.winch_line_record.getLast()[1]
        else:
            wrist = self.datastore.winch_line_record.getClosest(timestamp)[1]
        roomspin = wrist / 180 * np.pi
        if not self.calibrating_room_spin and self.config.gripper.frame_room_spin is not None:
            # undo the rotation that the room would appear to have at the wrist's 540 position
            extra = self.config.gripper.frame_room_spin - np.pi
            if debug:
                print(f'gripper spin should be wrist {roomspin} plus extra spin from config {extra}')
            roomspin = roomspin + extra
        return roomspin

    def gripper_body_room_rotation(self, timestamp=None):
        """Rotation taking a vector in the z-up gripper body frame (pole down -z, x/y horizontal)
        to the room frame: the room heading from get_spin() composed with the swing tilt from
        get_gripper_rvec(). timestamp evaluates both at that moment instead of now.

        get_spin() is a clockwise bearing, so gripper->room is a rotation by -spin."""
        R_heading = Rotation.from_rotvec([0.0, 0.0, -self.get_spin(timestamp=timestamp)])
        R_tilt = Rotation.from_rotvec(self.get_gripper_rvec(timestamp))
        return R_heading * R_tilt

    def measure_gantry_minus_card(self, pose_cam, timestamp=None):
        """Room-frame (gantry_position - card_position), from a calibration card's pose in
        the raw camera optical frame as stored in route_tag_samples.

        The gantry's absolute room position cancels, so this needs only the body
        orientation and the observed pose; the caller adds the card's known room position
        to recover where the gantry is.

        Frame chain:
        * model_constants.gripper_camera places the card in the CAD 'gripper frame' (y-up:
          grommet at +y, optical axis looking down -y).
        * Rx(90 deg) re-expresses that in the z-up body frame the rest of the system uses.
        * the gantry sits arp_pole_length up the +z body axis from the gripper origin.
        * gripper_body_room_rotation() rotates the body frame into the room.

        Pass timestamp (the pose's capture time) when working from a buffered sample: the
        body orientation that belongs with the card pose is the one from when the frame
        was taken, not whatever the gripper is doing now.
        """
        # card position in the CAD y-up gripper frame
        card_in_gripper = compose_poses([model_constants.gripper_camera, pose_cam])[1]
        # re-express in the z-up body frame, then measure relative to the gantry (pole up +z)
        y_up_to_z_up = Rotation.from_euler('x', 90, degrees=True)
        card_in_body = y_up_to_z_up.apply(card_in_gripper) - np.array([0.0, 0.0, model_constants.arp_pole_length])
        # rotate the card-relative-to-gantry vector into the room, then negate for gantry-card
        card_minus_gantry_room = self.gripper_body_room_rotation(timestamp).apply(card_in_body)
        return -card_minus_gantry_room

    def look_towards_vector(self, vec2):
        """Turn the wrist to face along the room-space XY vector [x, y].

        Spin runs 0 to 6*pi (nose at +Y is spin % 2pi == 0), so three wrist angles face
        the same way. The choice between them is what keeps the cable from winding up
        against either limit.
        """
        target_angle_base = math.atan2(vec2[0], vec2[1]) # (-pi, pi]
        current_spin = self.get_spin()

        norm_target = target_angle_base % (2 * math.pi)
        candidates = [norm_target, norm_target + 2 * math.pi, norm_target + 4 * math.pi]

        lower_bound = 0.5 * math.pi
        upper_bound = 5.5 * math.pi
        center_point = 3 * math.pi

        # near either limit, take a candidate that moves back toward the centre even if a
        # closer one exists; otherwise just the closest
        if current_spin < lower_bound:
            target = min([c for c in candidates if c > current_spin] or [candidates[-1]])
        elif current_spin > upper_bound:
            target = max([c for c in candidates if c < current_spin] or [candidates[0]])
        else:
            target = min(candidates, key=lambda c: abs(c - current_spin))

        raw_error = target - current_spin

        if abs(raw_error) < self.deadband:
            raw_error = 0.0

        self.smoothed_error = (self.ema_alpha * raw_error) + (1.0 - self.ema_alpha) * self.smoothed_error

        wrist_speed_deg = self.smoothed_error * self.p_gain * (180.0 / math.pi)

        wrist_speed = clamp(wrist_speed_deg, -120, 120)
        asyncio.create_task(self.send_commands({'set_wrist_speed': wrist_speed}))

    def process_frame(self, frame_to_encode):
        # Downscale only; the image is deliberately left unstabilized and unrotated. The
        # network's action space is then relative to the gripper image whatever
        # perspective the operator drives with: +Y means up in this frame.
        input_shape = (frame_to_encode.shape[1], frame_to_encode.shape[0])
        if input_shape != SF_TARGET_SHAPE:
            temp_image = cv2.resize(frame_to_encode, SF_TARGET_SHAPE, interpolation=cv2.INTER_AREA)
        else:
            temp_image = frame_to_encode
        return temp_image