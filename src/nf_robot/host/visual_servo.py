#!/usr/bin/env python

"""Grasping with the visual servoing model, and the debugging modes around it.

The model (ml/visual_servoing/readme.md) answers "where is the object and what is the
grip doing", not "how fast to move". This module is the other half: it turns one gripper
frame per pass into a gantry velocity, a wrist setpoint and a finger speed, and decides
when the object is in hand.

Three modes, of which only the first can grasp anything:

    grasp     centre, descend, close, lift, report
    observe   run the model and report it, commanding nothing at all
    center    steer sideways and turn the wrist, but never descend or touch the fingers

The two debugging modes exist because this model can only really be judged on a robot,
and watching what it wants is cheaper than watching what it does.

`VisualServo` holds no robot state of its own beyond the loaded checkpoint - the gripper
client, the datastore, the position estimate and the motion primitives all belong to the
observer it is constructed with, and it drives them the same way the observer's own
motion tasks do. Its coroutines follow the motion task contract: cancellable at any
await, cleaning up in a finally.
"""

import asyncio
import logging
import pathlib
import time

import numpy as np

from nf_robot.common.util import clamp
from nf_robot.generated.nf import telemetry
from nf_robot.host.arp_gripper_client import OPEN, CLOSED, RANGE_MAX_AGE_S
from nf_robot.host.fake_progress import FakeProgress

logger = logging.getLogger(__name__)

# What the routine is allowed to do. Only GRASP descends, closes the fingers or reports
# success; the other two run until cancelled.
SERVO_MODE_GRASP = 'grasp'
SERVO_MODE_OBSERVE = 'observe'
SERVO_MODE_CENTER = 'center'
SERVO_MODES = (SERVO_MODE_GRASP, SERVO_MODE_OBSERVE, SERVO_MODE_CENTER)

# ---------------------------------------------------------------------------
# Approach and descent
# ---------------------------------------------------------------------------
# (m) range to the item below which the grip is started, and where the descent profile
# aims to run out - so the two agree on where the approach ends.
RANGE_ITEM = 0.02
# (m) below this, a low target-present score no longer aborts: too far in to go looking
# for something better.
COMMIT_RANGE_M = 0.3
# Descent speed as a function of how far there is left to fall, rather than one number
# for the whole approach. A constant slow enough to be safe at the bottom spends the
# whole descent being slow where nothing is at stake, and one fast enough to be quick at
# the top arrives at the object still moving. Speed proportional to the range still to
# close makes the approach exponential: fast while there is room, self-limiting as the
# floor comes up, and no braking phase to tune.
#
#   range   0.60  0.40  0.20  0.15  0.10  0.06
#   m/s     0.12  0.12  0.12  0.08  0.05  0.03
DESCENT_GAIN = 0.85         # (1/s) speed asked for per metre of range left to close
DESCENT_SPEED_MAX = 0.14    # (m/s) cap while there is plenty of room below
DESCENT_SPEED_MIN = 0.07    # (m/s) floor, or the last centimetres never arrive
LATERAL_GAIN = 0.8          # (1/s) fraction of the remaining offset commanded per second
LATERAL_SPEED_MAX = 0.15    # (m/s)
# Descent is gated on being centered, and the tolerance is a fraction of the range, which
# makes it an angular tolerance: 3cm of error at 60cm up is a correction the rest of the
# descent absorbs, and the same 3cm at 8cm up is a miss.
CENTER_TOL_FRACTION = 0.12
CENTER_TOL_MIN_M = 0.012
PRESENT_THRESHOLD = 0.5     # target-present probability below which the loop holds still
NOTHING_SEEN_FRAMES = 15    # consecutive unsure frames before an attempt is abandoned

# (s) time constant of the filter on the target's room position, at about one pendulum
# period. The gripper hangs on half a metre of pole and swings, so a per-frame prediction
# is a picture of where the target was from wherever the lens happened to be at that
# instant; anything faster than this steers at the swing rather than at the object.
TARGET_EMA_S = 1.0
# (m) a jump further than this is a different object, not swing. Slewing across the gap
# would spend seconds pointing at the floor between the two, so the filter re-seeds.
TARGET_EMA_RESET_M = 0.5

# (m, s) how little the gantry may move, for how long, before an approach calls itself
# stuck. The threshold sits well above the position estimate's noise and well below what
# even the slowest part of a descent covers in this time, so it means "not moving" rather
# than "moving slowly".
STUCK_DISTANCE_M = 0.05
STUCK_TIMEOUT_S = 5.0

APPROACH_TIMEOUT_S = 30.0
PRESSURE_SENSE_WAIT = 6.0  # (s) how long a close may spend looking for a grip
NUM_ATTEMPTS = 3            # tries per call, unless a caller says otherwise
LOOP_DELAY = 0.03           # (s) added to inference time; the loop runs at what that allows
WATCH_LOG_S = 2.0           # (s) how often the debugging modes print

# ---------------------------------------------------------------------------
# Fingers
# ---------------------------------------------------------------------------
# The finger head is trained against commanded finger speed divided by 90
# (mine_teleop.FINGER_SPEED_FULL_SCALE), so this undoes that and nothing else.
FINGER_SPEED_FULL_SCALE = 90.0
# Predictions smaller than this are treated as zero. The head has a bias like any
# regressor, and a standing 0.02 is 1.8 deg/s - eleven degrees of unasked-for finger
# travel over a six second approach. Set to 0 to hand it the head raw.
FINGER_DEADBAND = 0.05
# Finger head output above which it is asking to close, and how many consecutive frames
# of asking it takes to stop the gantry - so one noisy frame cannot end an approach.
FINGER_CLOSE_THRESHOLD = 0.5
CLOSE_CONFIRM_FRAMES = 10
# The open head is trained against the operator's finger angle divided by 90, the same
# scale the gripper clamps angle commands to, so this undoes that and nothing else.
FINGER_ANGLE_FULL_SCALE = 90.0

# The pre-open, from the open-angle head.
#
# The model says how wide a human opened the jaws coming in to an object like this one,
# which is the one thing about the object worth knowing while there is still half a metre
# of descent left. Spreading the fingers to it during the approach is what makes the jaws
# arrive *around* the object instead of reaching it half shut and pushing it away.
#
# Gated on the approach having settled, because a finger angle command is a move the loop
# cannot see the end of, and one issued mid-correction is predicted from a picture that is
# about to be wrong. With the gantry and the wrist both nearly still, it is not.
PREOPEN_LATERAL_MAX = 0.02      # (m/s) commanded lateral speed that counts as settled
PREOPEN_WRIST_MAX_DEG = 8.0     # (deg) predicted axis correction that counts as the same
# (deg) how much wider than the model asked for. Being a little too wide costs nothing;
# being too narrow is how an object gets shouldered aside on the way in.
PREOPEN_MARGIN_DEG = 5.0
# (deg) how far the prediction has to move before the fingers are sent somewhere new. The
# head wobbles a degree or two frame to frame and the jaws should not chase it.
PREOPEN_MIN_CHANGE_DEG = 4.0
# (deg) close enough to the commanded angle to call the move done and go back to holding.
PREOPEN_TOLERANCE_DEG = 3.0
# (deg) the wide end of the gripper's travel, kept off the hard stop.
PREOPEN_MIN_DEG = -85.0
# Model may narrow the fingers up to this amount if it chooes to
PREOPEN_MAX_DEG = 60

# The close-heads program, used when the checkpoint carries a close-onset head.
#
# It replaces a per-frame rate with a decision and a target: hold still until the model
# says the close should have begun, then close at one speed until the grip is carrying
# the force the model expects this object to need. A rate label is a teleoperator's thumb
# and is different on every frame of the same situation; when to start and how hard to
# finish are properties of the object, which is what makes them worth predicting.
CLOSE_PROB_THRESHOLD = 0.45
# (normalized finger speed) how fast the close runs once it starts, and how long it takes
# to get there. The ramp is not politeness: a step to full speed puts a jolt through half
# a metre of hanging pole at the moment the jaws are arriving at the object.
CLOSE_SPEED = 0.45
CLOSE_RAMP_S = 0.3
# (sensor units) how close the commanded force has to get to the predicted one before the
# close stops pushing. The gripper turns finger speed into a force ramp once it feels
# contact, so this is the point where the ask has reached what the model expects to need.
CLOSE_PRESSURE_TOLERANCE = 0.01

# What it takes to call the object held, and so to lift.
#
# The pressure half is a ratio, not a level: the gripper turns a finger speed into a force
# ramp once it feels contact, so target_force is what the grip *asked* for and
# filtered_force is what it got. Those matching is the difference between a grip that is
# loaded and one still closing on air, and a ratio says that at whatever force this
# particular object wants, where a fixed level has to be set for the softest thing that
# will ever be picked up.
#
# The minimum on the commanded side is what keeps the ratio meaningful: with nothing asked
# for, nothing achieved satisfies any ratio. During teleop carries the median commanded
# force is 0.11 and 41% of frames ask for nothing at all.
HOLD_FORCE_FRACTION = 0.8
HOLD_FORCE_MIN_COMMANDED = 0.02
# Always hold this much more than the model wants
HOLD_FORCE_EXTRA = 0.05
#
# The second way to be loaded needs no reference to what was commanded: a felt force this
# high that has stopped moving is something solid between the pads. The force loop does
# not always converge on its setpoint - a thin or compliant object can leave it short
# indefinitely - and a grip that has settled at a real force is held, however far under
# the setpoint it came to rest.
HOLD_FORCE_MIN_FELT = 0.15
HOLD_FORCE_SETTLE_S = 0.5
# Peak-to-peak the felt force may wander over that window and still count as still.
HOLD_FORCE_SETTLE_BAND = 0.02
# Below this many samples in the window there is nothing to judge stillness from.
HOLD_FORCE_SETTLE_MIN_SAMPLES = 5
# And the model's own opinion, which is the half that can tell a loaded grip on an object
# from a loaded grip on a finger, a carpet edge or the floor.
HOLD_PROB_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Wrist
# ---------------------------------------------------------------------------
WRIST_GAIN = 0.5            # fraction of the predicted angle commanded per pass
WRIST_MAX_STEP_DEG = 20.0
# (von Mises kappa) how sure the axis head has to be before the wrist acts on it. Below
# this the head has no real opinion about orientation and its decoded angle is a hedge.
WRIST_MIN_KAPPA = 1.5
WRIST_LOCK_RANGE_M = 0.08   # (m) stop turning the wrist this close in
# (seconds) how far a wrist record may be from a frame's capture time and still describe
# where the wrist was when that frame was taken.
# take this with a grain of salt, video latency estimates aren't even accurate to 0.1s
WRIST_ANCHOR_MAX_AGE_S = 0.4
# The wrist's command range, in degrees from the servo's zero: three revolutions with
# neutral in the middle. gripper_arp_server.setWrist *clamps* to this, silently, so a
# setpoint past the end does not turn the wrist a little way and stop - it stops the wrist
# entirely, and every later correction clamps to the same place.
WRIST_RANGE_DEG = (0.0, 1080.0)
WRIST_NEUTRAL_DEG = 540.0
# (degrees) how far from each end to stay when there is a choice. A setpoint sitting on
# the limit has no room for the next correction in one of its two directions.
WRIST_LIMIT_MARGIN_DEG = 20.0
# How much a degree away from neutral costs, against a degree of travel, when choosing
# between equivalent setpoints. Small: it should decide a near-tie and bias a long run
# back toward the middle, not spend a half turn chasing the centre.
WRIST_NEUTRAL_PULL = 0.25

# ---------------------------------------------------------------------------
# Scoring runs
# ---------------------------------------------------------------------------
# How many times a scoring run will redraw a reposition that landed outside the work area
# before giving up on the hop and staying where it is.
REPOSITION_DRAW_TRIES = 20


def choose_wrist_setpoint(goal_deg, wrist_now, neutral=WRIST_NEUTRAL_DEG,
                          limits=WRIST_RANGE_DEG, margin=WRIST_LIMIT_MARGIN_DEG,
                          neutral_pull=WRIST_NEUTRAL_PULL):
    """A reachable wrist angle that puts the jaws on the same line as `goal_deg`.

    A two finger grasp axis is pi-periodic - the jaws close along the same line at X and
    at X+180 - so every target has several equivalent wrist angles, 180 degrees apart,
    and the wrist's three revolutions hold about six of them. That is what makes the
    limits survivable: a goal off the end of the range is still reachable by turning
    around and coming at it the other way.

    Which one to take is a trade between two costs. Travel is the obvious one. The other
    is distance from neutral, because a wrist parked near a limit has nowhere to go the
    next time a correction points that way, and because the cable twists. Weighting them
    together means a near-tie goes to the middle of the range while a clear winner on
    travel still wins - `neutral_pull` sets where that line falls.

    Candidates within `margin` of either limit are dropped when anything else is
    available, so the choice leaves room for the next correction to be acted on rather
    than clamped away.
    """
    low, high = limits
    base = goal_deg + round((wrist_now - goal_deg) / 180.0) * 180.0
    candidates = [base + 180.0 * k for k in range(-8, 9)]
    reachable = [c for c in candidates if low <= c <= high]
    if not reachable:
        return clamp(goal_deg, low, high)
    roomy = [c for c in reachable if low + margin <= c <= high - margin]
    return min(roomy or reachable,
               key=lambda c: abs(c - wrist_now) + neutral_pull * abs(c - neutral))


class TargetFilter:
    """The target's room position, smoothed over about one swing period.

    What is filtered is the position, not the offset to it. They differ the moment the
    gantry moves: an offset measured a second ago was measured from somewhere else, so
    averaging offsets fights the loop's own corrections and reads as lag, while averaging
    the point they name does not - the point holds still while the robot moves toward it.
    The current offset is taken fresh from the smoothed point every pass.

    Sampled against real elapsed time rather than a fixed per-frame weight, because the
    loop rate is set by how long inference took and a fixed alpha would mean a different
    time constant on a busy machine than on an idle one.
    """

    def __init__(self, tau_s=TARGET_EMA_S, reset_m=TARGET_EMA_RESET_M):
        self.tau_s = tau_s
        self.reset_m = reset_m
        self.reset()

    def reset(self):
        """Forget the run so far. Every attempt starts over: where the last one was
        pointing is not evidence about this one."""
        self.point = None
        self.at = None

    def error(self, prediction):
        """The lateral error to close, in room axes, from this frame's prediction."""
        now = time.time()
        target = prediction['point_room']
        # this frame's lens position in the room: the point the offset is measured from
        lens = target - prediction['room_offset']

        if (self.point is None or self.at is None
                or np.linalg.norm(target - self.point) > self.reset_m):
            self.point = target
        else:
            alpha = 1.0 - float(np.exp(-(now - self.at) / self.tau_s))
            self.point = self.point + alpha * (target - self.point)
        self.at = now
        return (self.point - lens)[:2]


class VisualServo:
    """Runs the visual servoing model against a robot, in one of SERVO_MODES.

    Constructed with the observer, whose gripper client, datastore, position estimate and
    motion primitives it drives. Holds the checkpoint and nothing else.
    """

    def __init__(self, observer):
        self.ob = observer
        self.model = None
        self.filter = TargetFilter()
        # Where the staged finger program has got to. Reset per attempt: a close that
        # began on the last one says nothing about this one.
        self.close_started_at = None
        self.close_arrived = False
        # The angle the jaws were last sent to by the pre-open, or None if they have not
        # been. Reset per attempt for the same reason the close state is.
        self.preopen_deg = None

    def reset_close(self):
        self.close_started_at = None
        self.close_arrived = False
        self.preopen_deg = None

    # -- model ------------------------------------------------------------

    async def ensure_model(self):
        """Load the checkpoint if it is not loaded. True if there is one to run."""
        if self.model is None:
            await self.load_model()
        return self.model is not None

    async def load_model(self):
        """Load the visual servoing checkpoint onto the eval device, or leave it None.

        Everything happens in the worker thread, including `import torch` and the hub
        lookup. Both are slow enough to matter: the first torch import is most of a
        second, and a download is however long the network takes, and on the event loop
        thread either one stalls the whole robot - the telemetry, the position estimator
        and every other motion task - for exactly that long.

        Nothing here raises. A model that will not load is a robot that cannot use this
        grasping routine, which is a thing to report and decline over, not a traceback out
        of whatever motion task happened to ask for it first.
        """
        def load_sync():
            import torch

            from nf_robot.ml.visual_servoing import servo

            device = self.ob._device or ("cuda" if torch.cuda.is_available()
                                         else "mps" if torch.backends.mps.is_available() else "cpu")
            if device == "cpu":
                return None, device, (
                    "The visual servoing grasp cannot be used without some kind of hardware "
                    "acceleration. Loading was aborted because the torch device is CPU.")

            model, checkpoint = servo.load_model(device, local_models=self.ob.local_models)
            logger.info(f"Visual servoing model ready: epoch {checkpoint.get('epoch')}, "
                        f"input {tuple(checkpoint['image_size'])}, "
                        f"axis loss {checkpoint.get('axis_loss')}, "
                        f"metrics {checkpoint.get('metrics')}")
            return model, device, None

        self.model = None
        # A grasp is what asked for this, so the wait happens mid-motion with no other
        # explanation on screen; the bar is on a timer because load_sync reports nothing.
        # It ends quietly - the failure paths below raise their own popups, which say far
        # more than a completion notice would.
        async with FakeProgress(
            self.ob.send_ui,
            name="Visual Servoing Model",
            current_action="Loading visual servoing model...",
            done_action="Visual servoing model ready",
            failed_action="Could not load the visual servoing model",
            expected_s=5.0,
            interval_s=0.2,
            suppress_completion_popup=True,
        ) as progress:
            try:
                model, device, refusal = await asyncio.to_thread(load_sync)
            except Exception as e:
                logger.error(f'Could not load the visual servoing model: {e!r}')
                self.ob.send_ui(pop_message=telemetry.Popup(message=self._load_failure_message(e)))
                progress.fail()
                return

            self.ob._device = device
            if refusal:
                logger.warning(refusal)
                self.ob.send_ui(pop_message=telemetry.Popup(message=refusal))
                progress.fail(refusal)
                return
            self.model = model

    def _load_failure_message(self, error):
        """What to tell a person about a checkpoint that would not load.

        The three ways this actually fails have three different answers, and a traceback
        gives none of them: the hub repo does not exist yet, the local file is not there,
        or the install is missing a package the model needs.
        """
        from nf_robot.ml.visual_servoing.servo import (
            LOCAL_MODEL_PATH, SERVO_MODEL_FILENAME, SERVO_MODEL_REPOID)

        name = type(error).__name__
        if isinstance(error, ImportError):
            return (f"The visual servoing model could not be loaded: {error}. It needs the "
                    f"'transformers' package, which is not part of the standard host "
                    f"install; add it with 'pip install transformers'.")
        if isinstance(error, FileNotFoundError) or name == 'EntryNotFoundError':
            return (f"No visual servoing checkpoint at {LOCAL_MODEL_PATH}. Train one, or drop "
                    f"a downloaded {SERVO_MODEL_FILENAME} there, or run without "
                    f"--local_models to fetch it from the hub.")
        if name in ('RepositoryNotFoundError', 'GatedRepoError', 'HfHubHTTPError',
                    'LocalEntryNotFoundError'):
            local = pathlib.Path(LOCAL_MODEL_PATH)
            here = (f" A local copy is sitting at {LOCAL_MODEL_PATH}; restart with "
                    f"--local_models to use it.") if local.exists() else ""
            return (f"The visual servoing model is not available from {SERVO_MODEL_REPOID} "
                    f"on huggingface ({name}). It may not be published yet.{here}")
        return (f"The visual servoing model could not be loaded ({name}: {error}). "
                f"See the log for the full traceback.")

    # -- what the model says ----------------------------------------------

    async def predict(self):
        """One prediction from the newest gripper frame, or None if there isn't one.

        Everything it reads goes to the UI as GripCamPredictions on the way out, so the
        overlay on the gripper feed shows what the loops below are acting on. Every call
        site is here, which is why the send is too.
        """
        from nf_robot.ml.visual_servoing import servo

        gripper = self.ob.gripper_client
        frame = gripper.last_output_frame
        # Read beside the frame, not later: it is what anchors the wrist setpoint to the
        # moment this picture was taken. The demux thread can be one frame ahead of the
        # streaming thread that publishes last_output_frame, which at 60fps is 16ms of
        # slack against a quarter second of video latency.
        captured_at = gripper.last_frame_cap_time
        if frame is None:
            return None
        _, finger_angle, _ = self.ob.datastore.finger.getLast()
        state = {
            'laser_rangefinder': self.ob.datastore.range_record.getLast()[1],
            'finger_angle': finger_angle,
            'target_force': gripper.last_target_force,
        }
        prediction = await asyncio.to_thread(
            servo.predict_frame, self.model, frame, state, self.ob._device,
            gripper.get_spin(), self.ob.pe.grip_pose[1])
        prediction['captured_at'] = captured_at
        self.ob.send_ui(grip_cam_preditions=telemetry.GripCamPredictions(
            # the overlay draws the arrow from the centre of the frame, so it wants a
            # displacement rather than a position. uv may fall outside 0..1 - that is the
            # target being off the edge, and an arrow leaving the frame is the right
            # picture of it.
            move_x=float(prediction['uv'][0] - 0.5),
            move_y=float(prediction['uv'][1] - 0.5),
            prob_target_in_view=prediction['present'],
            prob_holding=prediction['holding'],
            # the bar is drawn along (cos, sin) of this in image axes, which is the
            # convention the mined labels and their previews already use; folded into
            # [0, pi) because the grasp axis is pi-periodic and the field says so.
            grip_angle=float(prediction['grasp_axis_rad'] % np.pi),
        ))
        return prediction

    # -- what the robot does about it --------------------------------------

    async def drive_fingers(self, prediction, may_preopen=False):
        """Move the fingers the way this checkpoint's heads say to. Returns the speed sent.

        Two programs, chosen by what the model carries rather than by a flag, so one
        deployment path serves both kinds of checkpoint and the older one keeps behaving
        exactly as it did.

        `may_preopen` is the caller saying the approach is settled enough to spend the
        time on a finger angle move. Only the staged program can use it: a rate model is
        commanding finger speeds every pass, and an angle command dropped in among them
        would be fighting the head that is meant to be driving.
        """
        if 'close' in prediction:
            # we are using this one right now
            return await self._drive_fingers_staged(prediction, may_preopen)
        return await self._drive_fingers_rate(prediction)

    async def _drive_fingers_rate(self, prediction):
        """The finger head's rate, straight onto the fingers, in the robot's units.

        The head predicts a rate in the same normalized units the teleop labels were
        recorded in, and during a grasp this is the only thing that moves the fingers.
        Nothing else has a better claim: the fixed closing speed it replaces was a
        constant chosen for a routine with no opinion about fingers.
        """
        speed = prediction['finger']
        if abs(speed) < FINGER_DEADBAND:
            speed = 0.0
        await self.ob.gripper_client.send_commands(
            {'set_finger_speed': speed * FINGER_SPEED_FULL_SCALE})
        return speed

    async def _drive_fingers_staged(self, prediction, may_preopen=False):
        """Pre-open, hold, close to the predicted grip force, then stop and wait for load"""
        now = time.time()
        commanded = float(self.ob.gripper_client.last_target_force)
        target = prediction['grasp_pressure'] + HOLD_FORCE_EXTRA

        if self.close_arrived:
            # Terminal: hold the force already commanded. Sending zero speed is what does
            # that - the firmware only changes desired_force while a speed is commanded,
            # so zero means "stay here", not "let go".
            await self.ob.gripper_client.send_commands({'set_finger_speed': 0.0})
            return 0.0

        if self.close_started_at is None:
            if prediction['close'] < CLOSE_PROB_THRESHOLD:
                return await self._hold_or_preopen(prediction, may_preopen)
            self.close_started_at = now
            logger.info(f"Close head says go ({prediction['close']:.2f}); closing to "
                        f"{target:.3f} of grip force")

        # commanded > 0 is the gripper saying it is in force mode, which is to say the
        # fingers are on the object rather than still travelling toward it
        if commanded > 0.0 and commanded >= target - CLOSE_PRESSURE_TOLERANCE:
            self.close_arrived = True
            logger.info(f'Commanded force {commanded:.3f} reached the predicted '
                        f'{target:.3f}; holding there and waiting for the grip to load')
            await self.ob.gripper_client.send_commands({'set_finger_speed': 0.0})
            return 0.0

        ramp = min(1.0, (now - self.close_started_at) / CLOSE_RAMP_S) if CLOSE_RAMP_S else 1.0
        speed = CLOSE_SPEED * ramp
        await self.ob.gripper_client.send_commands(
            {'set_finger_speed': speed * FINGER_SPEED_FULL_SCALE})
        return speed

    def preopen_target(self, prediction):
        """The finger angle the open head is asking for, in degrees, or None.

        None when the checkpoint has no such head, which is how an older one goes on
        approaching with the fingers wherever _grasp parked them.
        """
        if 'open_angle' not in prediction:
            return None
        widest = prediction['open_angle'] * FINGER_ANGLE_FULL_SCALE
        return clamp(widest - PREOPEN_MARGIN_DEG, PREOPEN_MIN_DEG, PREOPEN_MAX_DEG)

    def approach_is_settled(self, prediction, lateral_speed):
        """Whether the gantry and the wrist have both nearly stopped asking for anything.

        The wrist half reads the prediction rather than what was commanded, because an
        axis head below its kappa gate is not turning anything: no opinion about
        orientation is the same stillness as a confident zero, and both mean the picture
        an angle is predicted from will still be the picture when the fingers arrive.
        """
        if lateral_speed > PREOPEN_LATERAL_MAX:
            return False
        if prediction['axis_concentration'] < WRIST_MIN_KAPPA:
            return True
        return abs(np.degrees(prediction['grasp_axis_rad'])) < PREOPEN_WRIST_MAX_DEG

    def preopen_arrived(self):
        """Whether the jaws have reached the angle the pre-open sent them to."""
        if self.preopen_deg is None:
            return True
        angle = float(self.ob.datastore.finger.getLast()[1])
        return abs(angle - self.preopen_deg) <= PREOPEN_TOLERANCE_DEG

    async def _hold_or_preopen(self, prediction, may_preopen):
        """The fingers before the close: spread to the open head's angle, or held still.

        One branch because the two share a command slot. A zero finger speed is what tells
        the gripper to stay where it is, and sending one on top of an angle move would
        stop that move partway - so while a pre-open is still travelling this commands
        nothing at all and lets the angle controller finish. Nothing is a safe thing to
        send: speed commands expire on their own after 200ms.
        """
        target = self.preopen_target(prediction) if may_preopen else None
        moved_enough = target is not None and (
            self.preopen_deg is None or abs(target - self.preopen_deg) > PREOPEN_MIN_CHANGE_DEG)
        if moved_enough:
            logger.info(f'Pre-opening the jaws to {target:.0f} deg for what the model sees')
            self.preopen_deg = target
            await self.ob.gripper_client.send_commands({'set_finger_angle': target})
            return 0.0
        if not self.preopen_arrived():
            return 0.0
        await self.ob.gripper_client.send_commands({'set_finger_speed': 0.0})
        return 0.0

    def wants_to_close(self, prediction):
        """Whether the model is asking for the fingers to close on this frame.

        The two kinds of checkpoint answer it with different heads, and the loop above
        cares about the answer rather than which head gave it. A rate model says so by
        commanding a fast close; a close-heads model says so directly, which is the whole
        reason those heads exist.
        """
        if 'close' in prediction:
            return prediction['close'] > CLOSE_PROB_THRESHOLD
        return prediction['finger'] > FINGER_CLOSE_THRESHOLD

    async def servo_wrist(self, prediction):
        """Turn the wrist toward the predicted grasp axis, if the head means it.

        The command is an absolute setpoint measured from where the wrist was **when the
        frame was captured**, not an increment on where it is now. That distinction is the
        whole of this method.

        The gripper camera runs about a quarter of a second behind the world, so a
        prediction is an angle relative to a wrist position from a quarter second ago,
        while the wrist telemetry is current - it already contains whatever motion is in
        flight. Adding a stale error to a fresh angle re-commands a correction that is
        already happening, once per pass for as long as the video takes to catch up, and
        the wrist arrives past the mark, reverses, and does it again. That is a limit
        cycle: filtering only slows it down, and sampling faster makes it worse.

        Anchoring to the capture-time angle removes the accumulation instead of damping
        it. `wrist_at_capture + error` names one fixed angle, so every frame taken before
        the wrist finished moving computes the same setpoint and re-sending it is
        idempotent. The lookup is exact rather than estimated: frame capture times and
        grip sensor records are both stamped by the gripper's own clock, so it crosses no
        clock boundary and needs no latency constant. Where the telemetry has a hole, the
        wrist is left alone - a wrong anchor is worse than no correction.

        The kappa gate is the other half. The axis head is trained as a von Mises
        likelihood, so the length of its output vector is a concentration: how sure it is,
        not merely what it thinks. An unsure head still decodes to *some* angle - hedging
        looks like a confident zero once atan2 has thrown the length away - and a wrist
        that acts on that is turning to a number the network did not mean.

        Returns True if the wrist was actually commanded, so a caller can say which.
        """
        if prediction['axis_concentration'] < WRIST_MIN_KAPPA:
            return False
        captured_at = prediction.get('captured_at')
        if captured_at is None:
            return False
        record = self.ob.datastore.winch_line_record.getClosest(captured_at)
        if abs(record[0] - captured_at) > WRIST_ANCHOR_MAX_AGE_S:
            logger.debug(f'No wrist telemetry within {WRIST_ANCHOR_MAX_AGE_S}s of the frame '
                         f'({record[0] - captured_at:+.2f}s off); leaving the wrist alone')
            return False
        wrist_at_capture = record[1]
        offset = clamp(np.degrees(prediction['grasp_axis_rad']) * WRIST_GAIN,
                       -WRIST_MAX_STEP_DEG, WRIST_MAX_STEP_DEG)
        # Not the raw sum: the wrist has three revolutions of travel and the server clamps
        # anything past them, so an approach that has walked toward a limit would otherwise
        # go quiet, every correction clamping to the angle it already holds.
        goal = choose_wrist_setpoint(wrist_at_capture + offset, wrist_at_capture)
        await self.ob.gripper_client.send_commands({'set_wrist_angle': goal})
        return True

    def descent_speed(self, range_to_target, lateral_speed):
        """How fast to descend right now, in m/s, from the range and what the lateral
        correction is already spending.

        The profile is proportional to the range still to close, aiming to run out where
        the grip starts, so the descent slows itself as the floor comes up rather than
        braking on a schedule.

        The headroom term does nothing at these speeds and is there for the day the cap is
        raised. move_direction_speed enforces the machine's limit by scaling the *whole*
        velocity vector, so a descent asked for beyond it does not simply get shortened -
        it shrinks the lateral correction summed with it in the same proportion, and the
        loop stops steering exactly when it is moving fastest.
        """
        profile = DESCENT_GAIN * max(0.0, range_to_target - RANGE_ITEM)
        headroom = float(np.sqrt(max(0.0, self.ob.speed_limit() ** 2 - lateral_speed ** 2)))
        return clamp(min(profile, headroom), DESCENT_SPEED_MIN, DESCENT_SPEED_MAX)

    # -- whether we have it -------------------------------------------------

    def force_settled(self):
        """Whether the felt force has held still lately, and the spread it held within.

        Peak-to-peak over the window rather than a slope: the value arrives already
        low-passed from the gripper, so what is left to catch is drift, and a band catches
        drift in either direction. The window is measured against host time the way the
        rangefinder's staleness check is, which means a gripper that stopped sending
        empties the window and reads as unsettled rather than as a perfectly steady force.
        Spread is None when there was not enough to judge.
        """
        recent = self.ob.datastore.finger.deepCopy(cutoff=time.time() - HOLD_FORCE_SETTLE_S)
        if len(recent) < HOLD_FORCE_SETTLE_MIN_SAMPLES:
            return False, None
        spread = float(recent[:, 2].max() - recent[:, 2].min())
        return spread <= HOLD_FORCE_SETTLE_BAND, spread

    def grip_loaded(self):
        """Whether the grip is carrying a load, by either of two independent tests.

        It is loaded if it is carrying the force it asked for (a ratio against the
        commanded force), or if it has simply come to rest at a real force, whatever was
        asked for. Either is enough: the first cannot fire when little or nothing was
        commanded, and the second cannot fire while the force is still moving.
        """
        commanded = float(self.ob.gripper_client.last_target_force)
        felt = float(self.ob.datastore.finger.getLast()[2])
        ratio = felt / commanded if commanded > 0 else 0.0

        tracking = (commanded >= HOLD_FORCE_MIN_COMMANDED
                    and ratio >= HOLD_FORCE_FRACTION)
        settled, spread = self.force_settled()
        resting = felt >= HOLD_FORCE_MIN_FELT and settled and commanded > HOLD_FORCE_MIN_COMMANDED
        return tracking or resting, commanded, felt, ratio, spread

    @property
    def uses_close_heads(self):
        """Whether the loaded checkpoint carries the close-onset and pressure heads.

        Read from the model rather than sniffed from a prediction, so it still answers on
        a pass where no frame was available and there is no prediction to look at.
        """
        return bool(getattr(self.model, 'close_heads', False))

    def holding_now(self, prediction):
        """This routine's own answer to "we have it, go up", and its evidence.
        
        TODO: this predicate has a hard time with paper
        It often grabs the paper but fails to detect enough pressure.
        If no pressure is felt but the prediction of holding is high,
        There's one way to know for sure if you go it. go up.
        If you do that and the visual holding head still says you have it, then you got it.
        """
        loaded, commanded, felt, ratio, spread = self.grip_loaded()
        probability = prediction['holding'] if prediction is not None else 0.0
        steadiness = 'too few samples' if spread is None else f'{spread:.3f}'
        evidence = (f'felt {felt:.3f} of {commanded:.3f} commanded ({ratio:.0%}, needs '
                    f'{HOLD_FORCE_FRACTION:.0%}), spread {steadiness} over {HOLD_FORCE_SETTLE_S}s '
                    f'(needs <{HOLD_FORCE_SETTLE_BAND:.3f} at felt >{HOLD_FORCE_MIN_FELT:.2f}), '
                    f'holding head {probability:.2f}')

        # A close-heads model has a defined end to its close, so nothing before that end
        # counts as holding.
        if self.uses_close_heads and not self.close_arrived:
            return False, evidence + ', close still running'

        return loaded and probability > HOLD_PROB_THRESHOLD, evidence

    # -- the modes ---------------------------------------------------------

    async def run(self, mode=SERVO_MODE_GRASP, attempts=None):
        """Run one of SERVO_MODES. True only if a grasp ended with the object held.

        `attempts` overrides how many tries a grasp gets before it reports failure.

        The debugging modes run until the motion task is cancelled and return False, since
        nothing was grasped.
        """
        if mode not in SERVO_MODES:
            raise ValueError(f"unknown visual servo mode {mode!r}; expected one of {SERVO_MODES}")
        if self.model is None:
            logger.warning('No visual servoing model loaded; cannot run')
            return False
        try:
            if mode in (SERVO_MODE_OBSERVE, SERVO_MODE_CENTER):
                return await self._watch(steering=mode == SERVO_MODE_CENTER)
            return await self._grasp(NUM_ATTEMPTS if attempts is None else attempts)
        except asyncio.CancelledError:
            raise
        finally:
            self.ob.slow_stop_all_spools()

    async def _watch(self, steering):
        """Report what the model wants, and optionally steer sideways for it.

        A separate loop from the grasp below, so that "never descends and never touches
        the fingers" is a property of the code rather than of every branch in it: there is
        no descent and no finger command in here to reach. Both modes run until cancelled
        and judge nothing - no attempt limit, no giving up, no stopping the spools when
        the target is lost.
        """
        logger.info('Visual servo: %s. Cancel the motion task to stop.',
                    'centering laterally, never descending'
                    if steering else 'watching only, commanding nothing')
        self.filter.reset()
        next_log = 0.0
        while self.ob.run_command_loop:
            prediction = await self.predict()
            if prediction is None:
                logger.debug('No frame available from gripper')
                await asyncio.sleep(LOOP_DELAY)
                continue

            error_xy = self.filter.error(prediction)
            error = float(np.linalg.norm(error_xy))
            turned = None
            if steering:
                confident = prediction['present'] >= PRESENT_THRESHOLD
                lateral = error_xy * LATERAL_GAIN if confident else np.zeros(2)
                lateral_speed = float(np.linalg.norm(lateral))
                if lateral_speed > LATERAL_SPEED_MAX:
                    lateral = lateral * (LATERAL_SPEED_MAX / lateral_speed)
                # downward_bias zeroed, unlike everywhere else. The bias exists to stop a
                # lateral move drifting upward against slack lines, but here it would mean
                # a "purely lateral" command sinking a few percent of its speed every
                # pass, which over a long centering session is a descent by another name.
                await self.ob.move_direction_speed([lateral[0], lateral[1], 0.0],
                                                   downward_bias=0.0)
                turned = await self.servo_wrist(prediction)

            if time.time() > next_log:
                # the overlay carries the detail; this is just proof of life for a
                # headless run, so it is throttled well below the loop rate
                next_log = time.time() + WATCH_LOG_S
                logger.info(
                    f"servo {'center' if steering else 'watch'} "
                    f"{self._describe(prediction)} offset {np.round(error_xy, 3)}m "
                    f"({error*100:.1f}cm)"
                    + ("" if turned is None else
                       f" wrist {'turning' if turned else 'held (unsure)'}"))
            await asyncio.sleep(LOOP_DELAY)
        return False

    def _describe(self, prediction):
        """The model's answer as one line, for the logs."""
        opened = prediction.get('open_angle')
        return (f"uv {np.round(prediction['uv'], 3)} range {prediction['range_m']:.3f}m "
                f"present {prediction['present']:.2f} holding {prediction['holding']:.2f} "
                f"finger {prediction['finger']:+.2f} "
                + ("" if opened is None else
                   f"open {opened * FINGER_ANGLE_FULL_SCALE:+.0f}deg ")
                + f"axis {np.degrees(prediction['grasp_axis_rad']):+.0f}deg "
                f"k{prediction['axis_concentration']:.1f} "
                f"lat {self.ob.gripper_client.video_latency(float('nan')):.02f}s")

    async def _grasp(self, attempts):
        """Try to pick up whatever the model sees below, up to `attempts` times."""
        tried, held, evidence = 0, False, 'no attempt made'
        while not held and tried < attempts and self.ob.run_command_loop:
            tried += 1
            logger.debug(f'Open fingers to {OPEN} to clear camera')
            asyncio.create_task(
                self.ob.gripper_client.send_commands({'set_finger_angle': OPEN}))

            reason, held, evidence, asked_to_close = await self._approach()
            self.ob.slow_stop_all_spools()
            logger.info(f'Visual servo approach ended: {reason}')

            if held:
                logger.info(f'Grip loaded during the approach: {evidence}')

            elif reason in ('nothing seen', 'rangefinder reading went stale'):
                continue # nothing worth closing on; spend another attempt looking

            elif reason == 'stuck' and not asked_to_close:
                # Being stuck is not itself a reason to close - the gripper could be hung
                # up anywhere. But it is often what arriving feels like when the object
                # stops the descent before the rangefinder reads through it, so the model
                # gets the casting vote: if it is asking to close, close.
                logger.info('Stuck and the model is not asking to close; not grasping')
                continue

            else:
                held, evidence = await self._close_until_held()

            if not held:
                await self._recover(evidence)
                continue

            await self._lift()
            return True

        logger.info(f'Gave up on visual servo grasp after {tried} attempt(s): {evidence}')
        return False

    async def _approach(self):
        """Fly to the object and close on it, returning why the approach ended.

        Returns (reason, held, evidence, asked_to_close). `held` is the grasp completing
        mid-flight, which is the normal way a good attempt ends: the model has the
        fingers, so the grip can be made while this loop is still steering toward it - and
        success switches off the very signal the loop would otherwise wait for, since an
        operator stops commanding squeeze once the grip is made. So the goal state is
        tested every pass rather than only after the approach ends.
        """
        self.filter.reset()
        self.reset_close()
        nothing_seen_countdown = NOTHING_SEEN_FRAMES
        close_countdown = CLOSE_CONFIRM_FRAMES
        approach_timeout = time.time() + APPROACH_TIMEOUT_S
        reason, evidence, asked_to_close = 'approach timed out', 'no frames', False
        still_since = time.time()
        still_at = np.array(self.ob.pe.gant_pos, dtype=float)

        while time.time() < approach_timeout and self.ob.run_command_loop:
            # Stuck: the gantry has not gone anywhere for a while. Whatever the reason - a
            # line snagged, the gripper resting on something, a descent that has run out
            # of rangefinder to close - continuing to command a velocity into it achieves
            # nothing, and the approach timeout is a long time to spend finding that out.
            if float(np.linalg.norm(self.ob.pe.gant_pos - still_at)) > STUCK_DISTANCE_M:
                still_at = np.array(self.ob.pe.gant_pos, dtype=float)
                still_since = time.time()
            elif time.time() - still_since > STUCK_TIMEOUT_S:
                return 'stuck', False, evidence, asked_to_close

            range_ts, range_to_target = self.ob.datastore.range_record.getLast()
            if time.time() - range_ts > RANGE_MAX_AGE_S:
                # the range is both an input to the model and the descent's only sense of
                # how far there is left to go; flying on a stale one is how the fingers
                # end up driven into the floor
                return 'rangefinder reading went stale', False, evidence, asked_to_close
            if range_to_target < RANGE_ITEM:
                gripper_height = self.ob.pe.grip_pose[1][2]
                return (f'reached target at height {gripper_height:.3f}m '
                        f'range {range_to_target:.3f}m'), False, evidence, asked_to_close

            prediction = await self.predict()
            if prediction is None:
                logger.debug('No frame available from gripper')
                await asyncio.sleep(LOOP_DELAY)
                continue

            if prediction['present'] < PRESENT_THRESHOLD and range_to_target > COMMIT_RANGE_M:
                if nothing_seen_countdown == NOTHING_SEEN_FRAMES:
                    # High up and unsure anything graspable is down there. Hold still
                    # rather than chase whatever the position head picked out of an empty
                    # floor - every other head is conditional on this one. Stopping once,
                    # not every pass, since it stays stopped.
                    self.ob.slow_stop_all_spools()
                nothing_seen_countdown -= 1
                if nothing_seen_countdown <= 0:
                    return 'nothing seen', False, evidence, asked_to_close
                await asyncio.sleep(LOOP_DELAY)
                continue
            nothing_seen_countdown = NOTHING_SEEN_FRAMES

            # horizontal part of the offset from the lens to the target, in the room
            # frame: exactly the error that must go to zero for the jaws to be over it,
            # filtered over a swing period so the descent steers at the object rather
            # than at the pendulum
            error_xy = self.filter.error(prediction)
            error = float(np.linalg.norm(error_xy))
            tolerance = max(CENTER_TOL_MIN_M, CENTER_TOL_FRACTION * range_to_target)
            centered = error < tolerance

            lateral = error_xy * LATERAL_GAIN
            lateral_speed = float(np.linalg.norm(lateral))
            if lateral_speed > LATERAL_SPEED_MAX:
                lateral = lateral * (LATERAL_SPEED_MAX / lateral_speed)
                lateral_speed = LATERAL_SPEED_MAX
            descent = self.descent_speed(range_to_target, lateral_speed) if centered else 0.0
            await self.ob.move_direction_speed([lateral[0], lateral[1], -descent])

            if range_to_target > WRIST_LOCK_RANGE_M:
                await self.servo_wrist(prediction)

            # The pre-open is only worth spending on a settled approach with room left
            # to spend it in: still steering means the angle would be predicted from a
            # picture that is about to change, and inside the commit range there is no
            # longer time for the jaws to travel before the close.
            may_preopen = (range_to_target > COMMIT_RANGE_M
                           and self.approach_is_settled(prediction, lateral_speed))
            last_finger = await self.drive_fingers(prediction, may_preopen)
            asked_to_close = self.wants_to_close(prediction)

            held, evidence = self.holding_now(prediction)
            if held:
                return 'grip loaded', True, evidence, asked_to_close

            # The close threshold's remaining job is to stop the gantry: once the model is
            # closing, driving on is how the object gets pushed out of the jaws.
            if asked_to_close:
                close_countdown -= 1
                if close_countdown <= 0:
                    return 'model asked to close', False, evidence, asked_to_close
            else:
                close_countdown = CLOSE_CONFIRM_FRAMES

            await asyncio.sleep(LOOP_DELAY)

        return reason, False, evidence, asked_to_close

    async def _close_until_held(self):
        """Hold the gantry still, let the model work the fingers, and wait for a grip.

        Reached when the approach ended without the grip already being made - the model
        stopped short of it, or the descent ran out of range first. The gantry is stopped
        by the caller, so the only thing moving here is the fingers, and the only thing
        moving them is the model: these frames, jaws around the object and filling the
        view, are exactly the ones its finger label was mined from.
        """
        logger.info('Close gripper')
        end_time = time.time() + PRESSURE_SENSE_WAIT
        held, evidence = False, 'no frames'
        angle = self.ob.datastore.finger.getLast()[1]
        while time.time() < end_time and angle < CLOSED:
            prediction = await self.predict()
            if prediction is not None:
                await self.drive_fingers(prediction)
            held, evidence = self.holding_now(prediction)
            if held:
                break
            await asyncio.sleep(LOOP_DELAY)
            angle = self.ob.datastore.finger.getLast()[1]
        # finger speed commands expire after 200ms, so this is only tidiness - but it
        # keeps a failed close from leaving one more command in flight behind it
        await self.ob.gripper_client.send_commands({'set_finger_speed': 0})
        logger.info(f'Close ended {"held" if held else "empty"}: {evidence}, '
                    f'finger angle {angle:.1f}')
        return held, evidence

    async def _lift(self):
        """Carry the object up and away from the floor."""
        logger.info('Successful grasp')
        # slowly at first, until the fingers are clear of the floor and the pole is
        # vertical, which keeps unwanted swinging to a minimum
        await self.ob.move_direction_speed(np.array([0, 0, 0.05]))
        await asyncio.sleep(1.0)
        # and then all at once
        await self.ob.move_direction_speed(np.array([0, 0, 0.15]))
        await asyncio.sleep(2.0)
        logger.info('Stop moving')
        self.ob.slow_stop_all_spools()

    async def _recover(self, evidence):
        """Back off after a failed attempt, into a position to try again from.

        The sideways part of the hop is deliberately random: the next attempt looks at
        whatever is below, so coming back to exactly the same place is how a routine
        retries a view that has already failed once.
        """
        logger.info(f'No hold ({evidence}); opening and going back up high enough to get '
                    f'a view of the object')
        await self.ob.move_direction_speed([0, 0, 0.06])
        await asyncio.sleep(1.0)
        asyncio.create_task(self.ob.gripper_client.send_commands({'set_finger_angle': OPEN}))
        direction = np.concatenate([np.random.uniform(-0.025, 0.025, (2)), [0.14]])
        await self.ob.move_direction_speed(direction)
        await asyncio.sleep(2.2)
        self.ob.slow_stop_all_spools()

    # -- scoring a checkpoint ----------------------------------------------

    async def score(self, settle_s=2.0, radius_m=0.4):
        """Grasp whatever is below, drop it, repeat, and keep score until cancelled.

        Each pass is one call allowed a single attempt, so a failure here is one grab that
        missed rather than the routine exhausting its retries. That is the number worth
        reporting: a hit rate over attempts, not over calls that each got three goes.

        Nothing re-targets between passes - the routine takes whatever is under the
        gripper - so this measures the servo loop and not the room-level targeting. An
        object flung out of view ends the useful part of a run, which is worth watching
        for rather than reading the tally afterwards and wondering.

        After each drop the gantry moves to a random point within `radius_m` of where the
        run began and the wrist to a random angle anywhere in its range, so the object is
        approached from a different direction, distance and orientation every time.
        Without it the loop would measure one approach geometry repeatedly and report the
        result as a hit rate.

        The drop is where the lift ended, a third of a metre or so up, which suits socks
        and towels and is worth thinking about before running it on anything fragile.
        """
        origin = np.array(self.ob.pe.gant_pos, dtype=float)
        logger.info(f'servoloop starting from {np.round(origin, 3)}; repositioning within '
                    f'{radius_m}m of it after every drop')
        attempts = successes = 0
        success_time, failure_time = [], []
        started_at = time.time()

        def tally():
            rate = successes / attempts if attempts else 0.0
            line = (f'servoloop {successes}/{attempts} ({rate:.0%}) in '
                    f'{(time.time() - started_at) / 60:.1f} min')
            if success_time:
                line += f' | success {np.mean(success_time):.1f}s avg'
            if failure_time:
                line += f' | failure {np.mean(failure_time):.1f}s avg'
            return line

        try:
            while self.ob.run_command_loop:
                began = time.time()
                success = await self.run(mode=SERVO_MODE_GRASP, attempts=1)
                elapsed = time.time() - began

                attempts += 1
                if success:
                    successes += 1
                    success_time.append(elapsed)
                else:
                    failure_time.append(elapsed)
                logger.info(f'servoloop attempt {attempts}: '
                            f'{"SUCCESS" if success else "failure"} in {elapsed:.1f}s | '
                            f'{tally()}')

                if success:
                    if not await self._release_payload():
                        # Refusing to keep going beats logging failures that are really
                        # this one condition repeating: every later attempt would return
                        # False without ever moving.
                        logger.error('servoloop stopping: could not release the payload')
                        break
                    await self._reposition_near(origin, radius_m)
                    # A random wrist angle too, drawn across the whole usable range rather
                    # than around neutral. Two things get exercised by that: the axis head,
                    # which sees the object at a new orientation every attempt, and the
                    # turnaround in choose_wrist_setpoint, which only comes into play when
                    # an approach starts near a limit.
                    wrist_target = float(np.random.uniform(
                        WRIST_RANGE_DEG[0] + WRIST_LIMIT_MARGIN_DEG,
                        WRIST_RANGE_DEG[1] - WRIST_LIMIT_MARGIN_DEG))
                    logger.info(f'servoloop wrist to {wrist_target:.0f} deg')
                    await self.ob.settle_wrist(wrist_target)
                # after the hop, not before it: the settle is for the swing the hop leaves
                await asyncio.sleep(settle_s)
        except asyncio.CancelledError:
            raise
        finally:
            logger.info(f'servoloop final: {tally()}')
            self.ob.slow_stop_all_spools()

    async def _release_payload(self, timeout=6.0):
        """Open the fingers and wait for the payload to be gone. True if it went.

        pe.holding is driven by finger pressure with hysteresis, so opening the hand is
        what clears it. This routine decides "held" its own way (see holding_now), but for
        letting go the shared flag is the right one to watch: it is pressure going away,
        which is exactly what a release is.
        """
        await self.ob.gripper_client.send_commands({'set_finger_angle': OPEN})
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.ob.pe.holding:
                return True
            await asyncio.sleep(0.1)
        logger.warning(f'Payload still registering as held {timeout}s after opening the '
                       f'fingers; the next attempt will refuse to start')
        return False

    async def _reposition_near(self, origin, radius, max_drop=0.1, timeout=30.0):
        """Fly the gantry to a random point within `radius` metres of `origin`.

        Uniform over the ball rather than over the coordinates, so the middle does not get
        visited more often than the edge - a benchmark that mostly repeats the same
        approach is not sampling approaches.

        Two constraints on the draw. The work area is the same test move_direction_speed
        applies before it will move at all, so a point outside it would simply not be
        flown to. `max_drop` is the vertical one: a gantry z below the start puts the
        fingers nearer the floor than the operator's parking judgement allowed for, and
        40cm of that finds the floor, the furniture, or the object just dropped. Upward is
        left alone - the descent covers height variety anyway, from above.

        Returns the point actually flown to, or None if the draw kept landing outside the
        work area, in which case nothing moves.
        """
        origin = np.asarray(origin, dtype=float)
        for _ in range(REPOSITION_DRAW_TRIES):
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-9
            # cube root, so the samples fill the volume evenly instead of crowding the centre
            offset = direction * radius * np.random.random() ** (1 / 3)
            target = origin + offset
            target[2] = max(target[2], origin[2] - max_drop)
            if self.ob.pe.point_inside_work_area(target):
                break
        else:
            logger.warning('Could not draw a reposition inside the work area; staying put')
            return None

        logger.info(f'servoloop repositioning to {np.round(target, 3)} '
                    f'({np.linalg.norm(target - origin):.2f}m from the start point)')
        try:
            # auto_altitude off: it climbs to a traversal height and back down, which is
            # the right move across a room and absurd for a hop of tens of centimetres
            await asyncio.wait_for(self.ob.seek_goal(target, auto_altitude=False), timeout)
        except asyncio.TimeoutError:
            logger.warning(f'Reposition did not arrive within {timeout}s; carrying on')
        finally:
            await self.ob.clear_goal()
        return target
