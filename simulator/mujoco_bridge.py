"""MuJoCo-backed hardware for the robot component simulator.

robot_simulator.py already replaces every piece of anchor and gripper hardware with a
stub that returns constants. This module supplies the same interfaces backed by the
MuJoCo model in stringman_arp_carbon270.xml, so the real AnchorArpServer and
GripperArpServer drive a simulated robot that actually swings, goes slack, and pulls
back. Nothing in nf_robot changes; the servers cannot tell the difference.

    python simulator/robot_simulator.py --mujoco

What each stub becomes
    DaMiaoController / DaMiaoMotor -> a spool whose shaft angle is integrated from the
        velocity commands the real spool loop sends, converted to line length through the
        same SpiralCalculator the firmware uses, and written to a MuJoCo tendon actuator.
        Reported torque comes back from the tendon's actual tension.
    SimpleSTS3215 -> the gripper's wrist and finger joints.
    MPU6050       -> the gripper's gyro/accelerometer sensors.
    VL53L1X       -> a MuJoCo rangefinder ray cast down from the gripper.
    ADS1015/AnalogIn -> the right pad's touch sensor, mapped back onto the FSR's
        voltage curve so findTouchPoint (and so 'fingercal') has something to feel.
    ffmpeg test pattern -> MuJoCo renders from the model's own cameras, at the
        resolution and field of view the real streams have (see CameraStreamer).

The physics runs in its own thread, paced against the wall clock, because the servers are
real-time: their loops sleep on time.time() and expect the world to move while they do.

WHAT THIS IS NOT, YET. Reinforcement learning wants the opposite of real-time pacing and
does not want a websocket stack in the loop. See the note at the bottom of this file.
"""

import logging
import math
import os
import threading
import time

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover - a clearer message than the raw ImportError
    raise ImportError(
        'the mujoco bridge needs the mujoco python package: pip install mujoco'
    ) from e

import nf_robot.common.definitions as model_constants
from nf_robot.robot.spools import SpiralCalculator
from nf_robot.robot.server_conf import read_hold_torque
from nf_robot.robot.spool_dm import HOLD_TORQUE_MOTION_VEL


DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'stringman_arp_carbon270.xml')

# Which MuJoCo actuator each (anchor, spool) drives, in (anchor, spool) order.
LINE_MAP = [
    'spool_0_direct',    # anchor 0, spool 0, high/direct
    'spool_1_indirect',  # anchor 0, spool 1, low/indirect
    'spool_2_direct',    # anchor 1, spool 0
    'spool_3_indirect',  # anchor 1, spool 1
]
# The MuJoCo actuators are commanded in free span, eyelet to gantry, while a spool pays
# out that plus a fixed run along the wall on the indirect lines. That run depends on the
# room, so it is measured off the compiled model rather than written down here; see
# MujocoWorld._measure_geometry.
INDIRECT_LINES = (1, 3)

# The room and camera adapter the XML is written for. Passing something else to
# MujocoWorld reconfigures the compiled model to match.
DEFAULT_ROOM_SIDE_M = 4.0
DEFAULT_CAM_TILT_DEG = 30.0

# Bodies that sit in the room's corners, and the sign of the corner each occupies.
CORNER_BODIES = {
    'anchor0': (1, 1), 'eyelet_post_A': (-1, 1),
    'anchor1': (-1, -1), 'eyelet_post_B': (1, -1),
}

# Gripper servo ids, matching gripper_arp_server.
FINGER, WRIST = 1, 2
STEPS_PER_REV = 4096

# The finger servo's real scale: 4096 steps to a motor revolution through the 10/45
# reduction, so one step is 0.0195 deg at the finger. This used to be an invented
# +/-1000 range spanning the whole travel, which is neither the right resolution nor the
# right direction.
GEAR_RATIO = 10.0 / 45.0
FINGER_RAD_PER_STEP = (2.0 * math.pi / STEPS_PER_REV) * GEAR_RATIO

# INCREASING STEPS OPEN THE FINGER. gripper_arp_server.findTouchPoint closes by
# decrementing the commanded position, and measureFingerContact then records
# finger_open_pos = finger_closed_pos + FINGER_TRAVEL_STEPS, so open sits above closed.
# Having this backwards inverts every finger command, and makes the reported angle run
# the wrong way over a fraction of its range.
#
# Zero on the joint is closed, the pose gripper.glb is drawn in. Where that falls on the
# encoder is arbitrary -- the firmware finds it by feel and stores it in
# arp_gripper_state.json -- but it is put low in the single-turn range so the finger can
# open upward without wrapping.
FINGER_CLOSED_RAD = 0.0
FINGER_CLOSED_STEPS = 128.0

# How close to the lens a camera still renders, in metres. MuJoCo expresses the near
# clipping plane as a fraction of the model extent, so the usable value moves with the
# room; this is pinned in absolute terms instead. It has to clear the gripper's own
# fingertips, which come within about 40 mm of its palm camera.
NEAR_PLANE_M = 0.004

# The DaMiao reports shaft position wrapped into this range, and spool_dm unwraps it.
# Reproduced here so that unwrapping code is actually exercised.
POS_WRAP_RAD = 25.0

logger = logging.getLogger(__name__)


class MujocoWorld:
    """The MuJoCo model, a thread that steps it, and locked accessors.

    Every server thread -- two spool loops per anchor, the gripper's motor loop, the
    asyncio loops -- touches this concurrently, so every read and write takes the lock
    and the physics thread holds it only while stepping.
    """

    def __init__(self, path=DEFAULT_MODEL, realtime=1.0, min_spool_accel=5.0,
                 room_side=DEFAULT_ROOM_SIDE_M, cam_tilt_deg=DEFAULT_CAM_TILT_DEG):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.room_side = float(room_side)
        self.cam_tilt_deg = float(cam_tilt_deg)
        self.realtime = realtime
        self.min_spool_accel = min_spool_accel
        self.lock = threading.RLock()
        self._run = False
        self._thread = None
        # callables the physics thread runs before each step, one per spool motor, so
        # shaft angles integrate against the same clock as the physics
        self._pre_step = []

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self._act = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                     for n in LINE_MAP}
        self._act['wrist'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'wrist')
        self._act['finger'] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'finger')

        def jnt(name):
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        self._wrist_q, self._wrist_v = jnt('wrist')
        self._finger_q, self._finger_v = jnt('finger_left')

        def sens(name):
            s = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            return self.model.sensor_adr[s], self.model.sensor_dim[s]
        self._gyro = sens('gripper_gyro')
        self._accel = sens('gripper_accel')
        self._range = sens('gripper_range')
        self._touch = {'left': sens('pad_touch_left'), 'right': sens('pad_touch_right')}

        self._pad_geoms = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                           for n in ('pad_left', 'pad_right')}

        # extent scales with the room, so express the near plane in metres and convert
        self.model.vis.map.znear = NEAR_PLANE_M / float(self.model.stat.extent)

        self._set_room_size(self.room_side)
        self._set_camera_tilt(self.cam_tilt_deg)
        self._measure_geometry()

    # -- configuring the room ---------------------------------------------------

    def _set_room_size(self, side):
        """Move the four corner posts onto a square `side` metres across, and match the
        floor to it. The corner azimuths do not change, so the gantry's resting yaw and
        the line-to-arm rigging are unaffected."""
        half = side / 2.0
        for name, (sx, sy) in CORNER_BODIES.items():
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.model.body_pos[b][0] = sx * half
            self.model.body_pos[b][1] = sy * half
        f = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.model.geom_size[f][0] = half
        self.model.geom_size[f][1] = half

    def _set_camera_tilt(self, deg):
        """Point both anchor cameras `deg` below horizontal, the angle of the tilt adapter
        fitted to the anchor.

        MuJoCo cameras look down local -z with +y up, so the frame is
            x = (-1, 0, 0)      y = (0, -sin t, cos t)      z = (0, cos t, sin t)
        which is definitions.arp_anchor_camera's rotation with its OpenCV y and z axes
        negated. The host has to be told the same angle (config indirectLine.camTilt),
        or calibration solves against a camera pointing somewhere else."""
        t = math.radians(deg)
        st, ct = math.sin(t), math.cos(t)
        mat = np.array([[-1.0, 0.0, 0.0],
                        [0.0, -st,  ct],
                        [0.0,  ct,  st]], dtype=np.float64)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        for name in ('anchor0_cam', 'anchor1_cam'):
            c = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.model.cam_quat[c] = quat

    def _measure_geometry(self):
        """Read the run lengths and taut spans this room implies, and push them into the
        parts of the model that depend on them.

        Everything here used to be a constant written down for one particular room: the
        fixed cross-room run baked into each indirect actuator's bias, the tendon length
        limits, and the keyframe's commanded spans. Measuring instead means the room and
        the camera adapter can move without any of them going quietly stale."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        site = lambda n: self.data.site_xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)]

        self.fixed_run = [0.0] * len(LINE_MAP)
        for i, (leave, post) in ((1, ('a0_indirect', 'eyelet_A')),
                                 (3, ('a1_indirect', 'eyelet_B'))):
            self.fixed_run[i] = float(np.linalg.norm(site(leave) - site(post)))

        full = model_constants.assumed_full_line_length
        self.keyframe_spans = []
        for i, name in enumerate(LINE_MAP):
            a = self._act[name]
            total = float(self.data.ten_length[a])
            self.keyframe_spans.append(total - self.fixed_run[i])
            # the affine bias is what converts a free-span command into a tendon length
            kp = float(self.model.actuator_gainprm[a][0])
            self.model.actuator_biasprm[a][0] = kp * self.fixed_run[i]
            # a spool holds `full` metres of line, beyond its fixed run
            self.model.tendon_range[a][0] = 0.0
            self.model.tendon_range[a][1] = full + self.fixed_run[i]
            if self.model.nkey:
                self.model.key_ctrl[0][a] = self.keyframe_spans[i]
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        logger.info('room %.2f m, camera tilt %.1f deg: fixed runs %s, keyframe spans %s',
                    self.room_side, self.cam_tilt_deg,
                    [round(r, 4) for r in self.fixed_run],
                    [round(v, 4) for v in self.keyframe_spans])

    def keyframe_spool_length(self, line_index):
        """Total line out at the keyframe pose, which is what a spool starts wound to."""
        return self.keyframe_spans[line_index] + self.fixed_run[line_index]

    # -- lifecycle -------------------------------------------------------------

    def register_pre_step(self, fn):
        with self.lock:
            self._pre_step.append(fn)

    def start(self):
        if self._thread is not None:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, name='mujoco-physics', daemon=True)
        self._thread.start()
        logger.info('mujoco physics thread started (realtime x%.2f, timestep %.4f s)',
                    self.realtime, self.model.opt.timestep)

    def stop(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        dt = self.model.opt.timestep
        next_wall = time.perf_counter()
        while self._run:
            with self.lock:
                for fn in self._pre_step:
                    fn(dt)
                mujoco.mj_step(self.model, self.data)
            next_wall += dt / max(self.realtime, 1e-6)
            slack = next_wall - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                # fell behind; give up the lost time rather than spiral
                next_wall = time.perf_counter()

    # -- lines -----------------------------------------------------------------

    def set_line_length(self, line_index, metres):
        """Command the total line a spool has paid out.

        The MuJoCo actuators are commanded in free span, eyelet to gantry, so the fixed
        cross-room run an indirect line carries comes off here. Getting this wrong is
        silent: the command just clips to the actuator's 0-7 m range and that line hangs
        slack while the others take its share of the load."""
        i = self._act[LINE_MAP[line_index]]
        run = self.fixed_run[line_index]
        lo, hi = self.model.actuator_ctrlrange[i]
        with self.lock:
            self.data.ctrl[i] = float(np.clip(metres - run, lo, hi))

    def get_line_tension(self, line_index):
        """Newtons, positive. The actuators only ever pull, so force is <= 0."""
        with self.lock:
            return float(-self.data.actuator_force[self._act[LINE_MAP[line_index]]])

    def get_line_length(self, line_index):
        """Total line out, the same quantity set_line_length takes."""
        with self.lock:
            return float(self.data.ten_length[self._act[LINE_MAP[line_index]]])

    # -- gripper joints --------------------------------------------------------

    def set_wrist(self, rad):
        i = self._act['wrist']
        lo, hi = self.model.actuator_ctrlrange[i]
        with self.lock:
            self.data.ctrl[i] = float(np.clip(rad, lo, hi))

    def get_wrist(self):
        with self.lock:
            return float(self.data.qpos[self._wrist_q]), float(self.data.qvel[self._wrist_v])

    def get_wrist_force(self):
        with self.lock:
            return float(self.data.actuator_force[self._act['wrist']])

    def set_finger(self, rad):
        i = self._act['finger']
        lo, hi = self.model.actuator_ctrlrange[i]
        with self.lock:
            self.data.ctrl[i] = float(np.clip(rad, lo, hi))

    def get_finger(self):
        with self.lock:
            return float(self.data.qpos[self._finger_q]), float(self.data.qvel[self._finger_v])

    def get_finger_force(self):
        with self.lock:
            return float(self.data.actuator_force[self._act['finger']])

    # -- gripper sensors -------------------------------------------------------

    def get_gyro(self):
        a, n = self._gyro
        with self.lock:
            return np.array(self.data.sensordata[a:a + n])

    def get_accel(self):
        a, n = self._accel
        with self.lock:
            return np.array(self.data.sensordata[a:a + n])

    def get_range(self):
        """Metres to whatever the gripper's rangefinder ray hits. -1 when nothing."""
        a, _ = self._range
        with self.lock:
            return float(self.data.sensordata[a])

    def get_pad_touch(self, side='right'):
        """Normal force on one finger pad, newtons, from that pad's touch sensor.

        Only the right pad carries an FSR on the real gripper, so that is the one the
        pressure sensor reads; the left is here for symmetry and for grasp diagnostics."""
        a, _ = self._touch[side]
        with self.lock:
            return float(self.data.sensordata[a])

    def get_pad_force(self):
        """Normal force on both pads together, newtons."""
        with self.lock:
            return sum(float(self.data.sensordata[self._touch[s][0]])
                       for s in ('left', 'right'))

    def gantry_position(self):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'gantry_origin')
        with self.lock:
            return np.array(self.data.site_xpos[sid])


class MujocoSpoolMotor:
    """Stands in for damiao_motor.DaMiaoMotor, backed by one MuJoCo line.

    The real spool loop in spool_dm.py sends velocity commands and reads back position
    and torque; it derives line length from position through a SpiralCalculator whose
    zero angle it calibrates from the host. This class keeps its own SpiralCalculator
    with the *true* zero angle and uses it to decide how much line is physically out.

    Keeping the two separate is deliberate: an uncalibrated spool loop believing the
    wrong length while the physics does something else is exactly the failure the real
    robot has, and setReferenceLength is what fixes it in both.
    """

    def __init__(self, world, line_index, empty_diameter, full_diameter, full_length,
                 direction, initial_length):
        self.world = world
        self.line_index = line_index
        self.direction = direction
        self.lock = threading.RLock()

        # same construction as DamiaoSpoolController: gear ratio 1, orientation -1
        self.sc = SpiralCalculator(empty_diameter, full_diameter, full_length, 1, -1)

        self.raw_rad = 0.0        # true shaft angle, motor frame, unwrapped
        self.cmd_vel = 0.0        # commanded shaft velocity, rad/s, motor frame
        self.vel = 0.0            # actual, after the acceleration ramp
        self.accel = None         # rad/s^2, from set_acceleration
        self.enabled = False
        self.mode = None
        self.last_turn = 0        # way the shaft last turned, for the reported friction

        # Seed the true zero angle so that shaft angle 0 means initial_length of line is
        # out, which is where the model's keyframe has the gantry.
        self.sc.set_zero_angle(self.sc.calc_za_from_length(initial_length, 0.0))
        self._push_length()
        world.register_pre_step(self._integrate)

    # -- physics side ----------------------------------------------------------

    def _shaft_revs(self):
        return self.direction * self.raw_rad / (2 * math.pi)

    def true_length(self):
        with self.lock:
            return self.sc.get_unspooled_length(self._shaft_revs())

    def _push_length(self):
        self.world.set_line_length(self.line_index, self.true_length())

    def _integrate(self, dt):
        """Called by the physics thread before each step, already holding its lock."""
        with self.lock:
            if not self.enabled:
                self.vel = 0.0
                return
            target = self.cmd_vel
            accel = self.accel
            if accel is None or accel <= 0:
                self.vel = target
            else:
                # The firmware's MAX_ACCEL default is 0.01 rad/s^2, which would take
                # minutes to reach cruise. min_spool_accel floors it so the simulator is
                # usable out of the box; pass min_spool_accel=None for the value verbatim.
                floor = self.world.min_spool_accel
                if floor is not None:
                    accel = max(accel, floor)
                step = accel * dt
                self.vel += float(np.clip(target - self.vel, -step, step))
            self.raw_rad += self.vel * dt
        self._push_length()

    # -- DaMiaoMotor interface -------------------------------------------------

    def enable(self):
        with self.lock:
            self.enabled = True

    def disable(self):
        with self.lock:
            self.enabled = False
            self.vel = 0.0

    def ensure_control_mode(self, mode):
        with self.lock:
            self.mode = mode

    def set_acceleration(self, a):
        with self.lock:
            self.accel = abs(a)

    def set_deceleration(self, d):
        with self.lock:
            self.accel = abs(d)

    def send_cmd_vel(self, target_velocity=0.0):
        with self.lock:
            self.cmd_vel = float(target_velocity)

    def send_cmd_mit(self, *a, **k):
        """check_motor_ids() probes with this; answering is enough to pass."""

    @property
    def state(self):
        return {'can_id': self.motor_id, 'arbitration_id': self.feedback_id}

    def get_states(self):
        """{'pos': rad (wrapped), 'vel': rad/s, 'torq': N.m}, all in the motor frame.

        spool_dm turns torque into tension as  T = -direction * torq * 2pi / m_per_rev,
        so the torque reported here is that relation run backwards from the tension
        MuJoCo is actually applying to this line.
        """
        tension = self.world.get_line_tension(self.line_index)
        with self.lock:
            revs = self._shaft_revs()
            m_per_rev = self.sc.get_unspool_rate(revs)
            torq = -tension * m_per_rev / (2 * math.pi * self.direction)
            # a real motor's torque reading also carries the friction it last pushed through,
            # which spool_dm takes back off. report the same amount so the two cancel.
            if self.vel > HOLD_TORQUE_MOTION_VEL:
                self.last_turn = 1
            elif self.vel < -HOLD_TORQUE_MOTION_VEL:
                self.last_turn = -1
            hold = self.hold_torque
            torq += {1: hold[0], -1: hold[1]}.get(self.last_turn, (hold[0] + hold[1]) / 2)
            # wrap position the way the real motor reports it
            pos = self.raw_rad
            pos = (pos + POS_WRAP_RAD / 2) % POS_WRAP_RAD - POS_WRAP_RAD / 2
            return {'pos': pos, 'vel': self.vel, 'torq': torq,
                    'status_code': 0, 't_mos': 25.0, 't_rotor': 25.0}


class MujocoDaMiaoController:
    """Stands in for damiao_motor.DaMiaoController.

    One per anchor. add_motor() is called for the high spool (id 0x02) then the low
    (0x01), matching anchor_arp_server, and each gets the MuJoCo line for that anchor.
    """

    # anchors are constructed in order by robot_simulator, so handing out line pairs in
    # construction order lines them up with the model's actuators
    _next_anchor = 0
    _lock = threading.Lock()

    def __init__(self, world, power=None, winding='short', **kwargs):
        self.world = world
        with MujocoDaMiaoController._lock:
            self.anchor_index = MujocoDaMiaoController._next_anchor
            MujocoDaMiaoController._next_anchor += 1
        # anchor 0 is the one with the power line, matching robot_simulator
        self.has_power = (self.anchor_index == 0) if power is None else power
        self.winding = winding
        self.motors = {}

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._next_anchor = 0

    def add_motor(self, motor_id=None, feedback_id=None, motor_type=None, **kwargs):
        # 0x02 is the high spool (direct line), 0x01 the low (indirect)
        spool = 0 if motor_id == 0x02 else 1
        line_index = self.anchor_index * 2 + spool
        kind = 'high' if spool == 0 else 'low'
        line_type = 'power' if (kind == 'high' and self.has_power) else 'fishing'
        full_length, full_diameter = model_constants.damiao_spool_geometry[
            (self.winding, kind, line_type)]
        # direction matches anchor_arp_server: -1 on the high spool, +1 on the low
        direction = -1 if spool == 0 else 1

        m = MujocoSpoolMotor(
            self.world, line_index,
            empty_diameter=model_constants.damiao_empty_spool_diameter,
            full_diameter=full_diameter, full_length=full_length,
            direction=direction,
            initial_length=self.world.keyframe_spool_length(line_index))
        m.motor_id = motor_id
        m.feedback_id = feedback_id
        # the same value spool_dm will load for this id
        m.hold_torque = read_hold_torque(motor_id)
        self.motors[motor_id] = m
        logger.info('anchor %d %s spool -> mujoco %s (%.1f m spool, start %.3f m out)',
                    self.anchor_index, kind, LINE_MAP[line_index], full_length,
                    m.true_length())
        return m

    def shutdown(self):
        for m in self.motors.values():
            m.disable()


class MujocoServoBus:
    """Stands in for nf_robot.robot.simple_st3215.SimpleSTS3215.

    The finger is modelled as the servo actually is: a step count at a fixed angular
    resolution, with an encoder reference that reset_encoder_to_midpoint renumbers. The
    firmware's -90..90 is a fictitious range it maps onto whatever open/closed step
    positions its own calibration found, so the bridge must not assume that range spans
    the travel - it only has to turn steps into an angle correctly. The wrist is
    different: the server subtracts a boot-time step offset before commanding, so only
    *relative* motion is meaningful here. The first commanded position is taken as the
    wrist's zero and everything after is relative to it, which keeps the server's own
    bookkeeping intact.
    """

    def __init__(self, world, *args, **kwargs):
        self.world = world
        self.torque = {FINGER: False, WRIST: False}
        self._wrist_zero_steps = None
        self._wrist_cmd_steps = 0.0
        # which step count corresponds to which joint angle
        self._finger_ref_steps = FINGER_CLOSED_STEPS
        self._finger_ref_rad = FINGER_CLOSED_RAD

    # -- conversions -----------------------------------------------------------

    def _finger_steps_to_rad(self, steps):
        """Steps up means open, and open is negative on this joint."""
        return self._finger_ref_rad - (steps - self._finger_ref_steps) * FINGER_RAD_PER_STEP

    def _finger_rad_to_steps(self, rad):
        return self._finger_ref_steps + (self._finger_ref_rad - rad) / FINGER_RAD_PER_STEP

    def _wrist_steps_to_rad(self, steps):
        if self._wrist_zero_steps is None:
            self._wrist_zero_steps = steps
        return math.radians((steps - self._wrist_zero_steps) / STEPS_PER_REV * 360.0)

    def _wrist_rad_to_steps(self, rad):
        zero = 0.0 if self._wrist_zero_steps is None else self._wrist_zero_steps
        return zero + math.degrees(rad) / 360.0 * STEPS_PER_REV

    # -- SimpleSTS3215 interface ----------------------------------------------

    def set_position(self, servo_id, position, speed=2400, acc=50):
        if servo_id == FINGER:
            self.world.set_finger(self._finger_steps_to_rad(position))
        else:
            self._wrist_cmd_steps = position
            self.world.set_wrist(self._wrist_steps_to_rad(position))

    def get_position(self, servo_id):
        return self.get_feedback(servo_id)['position']

    def get_feedback(self, servo_id):
        if servo_id == FINGER:
            rad, vel = self.world.get_finger()
            pos = self._finger_rad_to_steps(rad)
            spd = -vel / FINGER_RAD_PER_STEP
            force = self.world.get_finger_force()
            # the servo reports 0-1000 for load in the closing direction and 1024+ for
            # load the other way; the actuator's forcerange is +/-4 N.m
            mag = min(abs(force) / 4.0 * 1000.0, 1000.0)
            load = mag if force <= 0 else 1024 + mag
        else:
            rad, vel = self.world.get_wrist()
            pos = self._wrist_rad_to_steps(rad)
            spd = math.degrees(vel) / 360.0 * STEPS_PER_REV
            force = self.world.get_wrist_force()
            mag = min(abs(force) / 2.0 * 1000.0, 1000.0)
            load = mag if force <= 0 else 1024 + mag
        return {'position': pos, 'speed': spd, 'load': load,
                'voltage': 7.4, 'temp': 25, 'moving': int(abs(spd) > 1.0)}

    def set_speed(self, servo_id, speed):
        """The servers only ever use this to stop a motor."""

    def torque_enable(self, servo_id, enable=True):
        self.torque[servo_id] = bool(enable)

    def set_mode(self, servo_id, mode):
        pass

    def configure_multiturn(self, servo_id):
        pass

    def reset_encoder_to_midpoint(self, servo_id):
        """Renumber the encoder so the current position reads mid-range. Nothing moves;
        only the numbering changes. findTouchPoint leans on this when a close runs off
        the bottom of the servo's range."""
        if servo_id == WRIST:
            self._wrist_zero_steps = None
            self._wrist_cmd_steps = 0.0
        else:
            rad, _ = self.world.get_finger()
            self._finger_ref_steps = STEPS_PER_REV / 2.0
            self._finger_ref_rad = rad

    def ping(self, servo_id):
        return True

    def scan(self, maxid=253):
        return [FINGER, WRIST]


class MujocoIMU:
    """Stands in for adafruit_mpu6050.MPU6050."""

    def __init__(self, world, *a, **k):
        self.world = world

    @property
    def gyro(self):
        return tuple(self.world.get_gyro())

    @property
    def acceleration(self):
        return tuple(self.world.get_accel())

    @property
    def temperature(self):
        return 25.0


class MujocoRangefinder:
    """Stands in for adafruit_vl53l1x.VL53L1X. distance is centimetres."""

    def __init__(self, world, *a, **k):
        self.world = world
        self.distance_mode = 2
        self.timing_budget = 100

    def start_ranging(self):
        pass

    def stop_ranging(self):
        pass

    @property
    def model_info(self):
        return (1, 2, 3)

    @property
    def data_ready(self):
        return True

    def clear_interrupt(self):
        pass

    @property
    def distance(self):
        m = self.world.get_range()
        if m < 0:          # ray hit nothing
            return None
        return m * 100.0


class MujocoPressure:
    """Stands in for adafruit_ads1x15.AnalogIn on the finger pad FSR.

    Only the gripper's RIGHT pad has a sensor. It reads 3.3 V untouched and falls toward
    0 as the pad is pressed.

    The response is not linear in force: the server flattens it back out with
    `norm_pressure = ((3.3 - v) / 3.3) ** 2.5`, which says the raw voltage drops steeply
    under a light touch and barely moves under a hard one. Using the inverse exponent
    here makes the server's normalised pressure come out linear in pad force, which is
    the behaviour that curve was written to recover.

    Sensitivity matters for more than realism: findTouchPoint closes until the voltage
    falls below 2.2, so if a light touch does not move the voltage the fingers keep
    closing and 'fingercal' never terminates.

    FULL_SCALE_N is set from that threshold rather than from any datasheet. The real
    pads are urethane foam and press against each other hard enough to read about 2.0 V,
    which is why 2.2 is the threshold. Pressed together in here the pads develop about
    0.56 N, so full scale is chosen to put that force at ~1.9 V. At 10 N it read 2.26 and
    findTouchPoint closed straight past it.
    """

    FULL_SCALE_N = 5.0
    RESPONSE_EXP = 0.4          # inverse of the server's 2.5

    def __init__(self, world, *a, **k):
        self.world = world

    @property
    def voltage(self):
        f = min(max(self.world.get_pad_touch('right'), 0.0) / self.FULL_SCALE_N, 1.0)
        return 3.3 * (1.0 - f ** self.RESPONSE_EXP)

    @property
    def value(self):
        return int(self.voltage / 3.3 * 32767)



# ---------------------------------------------------------------------------
# Camera streams
# ---------------------------------------------------------------------------

class StreamSpec:
    """One video stream: which MuJoCo camera, at what size, rate and TCP port.

    width/height/fps/bitrate come from component_server.stream_modes so the simulated
    stream matches what the real component would produce. The field of view is not set
    here -- it is baked into the model's cameras, taken from the same camera calibration
    the detection pipeline interprets the frames with.
    """

    def __init__(self, camera, width, height, fps, bitrate, port, name=''):
        self.camera = camera
        self.width = width
        self.height = height
        self.fps = fps
        # stream_modes writes bitrates the way rpicam-vid wants them ('1000kbps').
        # ffmpeg's -b:v wants '1000k' and refuses the 'bps', with the only symptom being
        # an encoder that never opens and a stream that serves no packets.
        self.bitrate = str(bitrate).replace('bps', '')
        self.port = port
        self.name = name or camera
        self.queue = None     # asyncio.Queue, created on the loop that drains it
        self.dropped = 0
        self.sent = 0


def stream_specs(anchor_ports, gripper_port, anchor_mode='anchor_control',
                 gripper_mode='gripper_control'):
    """The streams a simulated robot should serve, sized from the real stream modes."""
    from nf_robot.robot.component_server import stream_modes
    a = stream_modes[anchor_mode]
    g = stream_modes[gripper_mode]
    specs = [StreamSpec('anchor%d_cam' % i, a.width, a.height, a.framerate, a.bitrate,
                        port, 'anchor%d' % i)
             for i, port in enumerate(anchor_ports)]
    specs.append(StreamSpec('gripper_cam', g.width, g.height, g.framerate, g.bitrate,
                            gripper_port, 'gripper'))
    return specs


class CameraStreamer:
    """Renders the model's cameras on one thread and feeds each stream's frame queue.

    All rendering happens on this single thread: a mujoco.Renderer owns an OpenGL
    context bound to the thread that made it, so they cannot be shared around. One
    Renderer per distinct frame size, shared by streams of that size.

    Frames go into a bounded queue per stream and an asyncio task pushes them into
    ffmpeg. If a consumer stalls the queue fills and frames are dropped rather than
    blocking the renderer, because blocking here would stall every other camera too.
    """

    def __init__(self, world, specs, loop):
        self.world = world
        self.specs = specs
        self.loop = loop
        self._run = False
        self._thread = None
        self._renderers = {}

        # A camera sees the robot, not the annotations. Sites are on by default and one
        # of them is a translucent overlay sitting right on the AprilTag; the rangefinder
        # ray is a debugging aid that draws into the frame. Geom group 3, the collision
        # primitives, is already off in the default options - set explicitly so that a
        # later change to the model's groups cannot quietly put them in the picture.
        self.scene_option = mujoco.MjvOption()
        self.scene_option.sitegroup[:] = 0
        self.scene_option.geomgroup[3] = 0
        # Group 2 is the anchor and eyelet-post bodies. Those are placeholder boxes, not
        # geometry extracted from anything, and an anchor's camera is mounted on its own
        # body, so once the near plane came in the box appeared in that anchor's own
        # frame.
        self.scene_option.geomgroup[2] = 0
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        # Tendons off. MuJoCo draws a line as an opaque 2.5 mm tube, and one of them
        # passes right across the marker card from the anchors' point of view, cutting
        # the tag's quad in half so the detector finds nothing. The real lines are thin
        # fishing line at close range on a lens focused metres away: not visible enough
        # to break a tag. Drawn tendons are useful in the interactive viewer, so this
        # only turns them off for the camera feeds.
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False

    def start(self):
        self._run = True
        self._thread = threading.Thread(target=self._loop_fn, name='mujoco-render', daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _renderer(self, w, h):
        if (w, h) not in self._renderers:
            self._renderers[(w, h)] = mujoco.Renderer(self.world.model, height=h, width=w)
        return self._renderers[(w, h)]

    def _submit(self, spec, frame):
        q = spec.queue
        if q is None:
            return

        def put():
            if q.full():
                try:
                    q.get_nowait()          # drop the stalest frame, keep the newest
                    spec.dropped += 1
                except Exception:
                    pass
            try:
                q.put_nowait(frame)
                spec.sent += 1
            except Exception:
                spec.dropped += 1

        try:
            self.loop.call_soon_threadsafe(put)
        except RuntimeError:
            pass                            # loop is shutting down

    def _loop_fn(self):
        try:
            # Every renderer must exist before any of them draws. Constructing one makes
            # its GL context current, so building a second one mid-run leaves the first
            # rendering into the wrong context and it returns black frames.
            for spec in self.specs:
                self._renderer(spec.width, spec.height)
        except Exception as e:
            logger.error('could not create a MuJoCo renderer (%s). Camera streams are '
                         'off. Headless? try MUJOCO_GL=egl, or MUJOCO_GL=osmesa for '
                         'software rendering.', e)
            return

        next_at = {spec.name: time.perf_counter() for spec in self.specs}
        behind = 0
        warned = False
        while self._run:
            now = time.perf_counter()
            due = [s for s in self.specs if next_at[s.name] <= now]
            if not due:
                time.sleep(min(next_at[s.name] for s in self.specs) - now)
                continue
            for spec in due:
                r = self._renderer(spec.width, spec.height)
                # Each Renderer owns its own GL context and creating one leaves it
                # current, so with more than one size in play the others quietly render
                # black frames. Bind this renderer's context before drawing into it.
                r._gl_context.make_current()
                # hold the physics lock only for the scene copy, not the render itself
                with self.world.lock:
                    r.update_scene(self.world.data, camera=spec.camera,
                                   scene_option=self.scene_option)
                self._submit(spec, r.render().tobytes())
                next_at[spec.name] += 1.0 / spec.fps
                if next_at[spec.name] < now:        # cannot keep up; resync
                    next_at[spec.name] = now + 1.0 / spec.fps
                    behind += 1
            if behind > 60 and not warned:
                warned = True
                logger.warning(
                    'the render thread is behind: %d frames resynced. The cameras are '
                    'running slower than their nominal rate. Lower the stream mode, or '
                    'check that MUJOCO_GL is using a GPU rather than software rendering.',
                    behind)

    def stats(self):
        return {s.name: (s.sent, s.dropped) for s in self.specs}


async def serve_stream(spec, world_streamer=None):
    """Keep an ffmpeg h264/mpegts listener on spec.port fed with rendered frames.

    ffmpeg exits when its client disconnects, and this restarts it. Raw frames come in on
    stdin instead of from lavfi.
    """
    import asyncio

    spec.queue = asyncio.Queue(maxsize=3)
    cmd = [
        'ffmpeg',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '%dx%d' % (spec.width, spec.height),
        '-r', str(spec.fps),
        '-i', '-',
        '-vcodec', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
        '-b:v', str(spec.bitrate),
        '-f', 'mpegts',
        'tcp://0.0.0.0:%d?listen=1' % spec.port,
        '-y', '-loglevel', 'warning',
    ]
    proc = None
    try:
        while True:
            logger.info('%s stream on port %d: %dx%d @ %d fps from mujoco camera %s',
                        spec.name, spec.port, spec.width, spec.height, spec.fps, spec.camera)
            proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE)
            try:
                while proc.returncode is None:
                    frame = await spec.queue.get()
                    proc.stdin.write(frame)
                    await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass                     # client went away; ffmpeg exits, we restart it
            except Exception as e:
                logger.warning('%s stream write failed: %s', spec.name, e)
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
            proc = None
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        if proc is not None:
            proc.kill()
            await proc.wait()
        raise


def seed_reference_lengths(anchor_servers):
    """Tell each spool loop how much line is really out.

    A freshly started spool loop has no zero angle, so it believes its whole winding is
    paid out -- 7.5 m against the 4.2 m the model actually hangs at. On the real robot
    the host fixes that during calibration by calling setReferenceLength; doing the same
    here means the simulator starts consistent instead of spending its first minute
    reeling in line that was never out.
    """
    for a_i, server in enumerate(anchor_servers):
        for s_i, spool in enumerate(server.spools):
            motor = spool.motor
            if not isinstance(motor, MujocoSpoolMotor):
                continue
            true_len = motor.true_length()
            # the loop needs one pass to have read an angle before this means anything
            spool.last_angle = motor._shaft_revs()
            spool.setReferenceLength(true_len)
            logger.info('anchor %d spool %d seeded to %.3f m', a_i, s_i, true_len)


def make_patchers(world):
    """The unittest.mock patches that swap MuJoCo in for the stub hardware.

    Mirrors the patch list in robot_simulator.main so the two stay comparable.
    """
    from unittest.mock import patch

    MujocoDaMiaoController.reset()

    return [
        patch('nf_robot.robot.anchor_arp_server.DaMiaoController',
              lambda *a, **k: MujocoDaMiaoController(world, **k)),

        patch('nf_robot.robot.gripper_arp_server.SimpleSTS3215',
              lambda *a, **k: MujocoServoBus(world)),
        patch('nf_robot.robot.gripper_arp_server.board.SCL', None, create=True),
        patch('nf_robot.robot.gripper_arp_server.board.SDA', None, create=True),
        patch('nf_robot.robot.gripper_arp_server.busio.I2C', lambda a, b: None),
        patch('nf_robot.robot.gripper_arp_server.MPU6050',
              lambda *a, **k: MujocoIMU(world)),
        patch('nf_robot.robot.gripper_arp_server.VL53L1X',
              lambda *a, **k: MujocoRangefinder(world)),
        patch('nf_robot.robot.gripper_arp_server.ADS1015', lambda *a, **k: object()),
        patch('nf_robot.robot.gripper_arp_server.AnalogIn',
              lambda *a, **k: MujocoPressure(world)),
    ]


# -- Where this goes next, for reinforcement learning --------------------------
#
# This bridge puts MuJoCo under the real firmware, which is the right thing for testing
# the firmware and the host: everything above the hardware line is genuinely exercised,
# including the spool tension logic, the swing filter and the vision pipeline's
# consumers. It is the wrong shape for RL training, for two reasons:
#
#   1. It is paced to the wall clock. Training wants thousands of episodes as fast as
#      the CPU allows. MujocoWorld(realtime=N) buys a linear speedup and nothing more,
#      because the servers' own loops sleep on time.time() and will not keep up.
#   2. Every observation and action crosses a websocket and a protobuf, through asyncio
#      loops in three processes. That is milliseconds per step of pure overhead and a
#      lot of moving parts to keep alive across a training run.
#
# The usual split is to keep both: this bridge as the fidelity check, and a separate
# gymnasium Env that drives MujocoWorld directly -- reset() to a keyframe, step()
# writing the four line lengths and the two gripper joints, observations read straight
# off mjData. Policies train against the Env, then run against this bridge to confirm
# they survive contact with the real control stack before they touch hardware.
