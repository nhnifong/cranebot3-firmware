from __future__ import annotations

import signal
import sys
import shutil
import faulthandler
import threading
import time
import socket
import asyncio
import argparse
import logging
import importlib.metadata
from zeroconf import IPVersion, ServiceStateChange, Zeroconf
from zeroconf.asyncio import (
    AsyncServiceBrowser,
    AsyncServiceInfo,
    AsyncZeroconf,
    AsyncZeroconfServiceTypes,
    InterfaceChoice,
)
from multiprocessing import Pool, Process
import numpy as np
import scipy.optimize as optimize
from scipy.spatial.transform import Rotation
from random import random
import traceback
import cv2
import pickle
import inspect
import itertools
from contextlib import asynccontextmanager, contextmanager
from collections import deque, defaultdict
from dataclasses import replace
import uuid
from functools import partial, wraps
from pathlib import Path
import json
import re
import subprocess
import zipfile
from packaging.version import parse as parse_version, InvalidVersion

from nf_robot.common.model_revisions import pinned_revision
from nf_robot.common.pose_functions import compose_poses, invert_pose
from nf_robot.common.cv_common import *
from nf_robot.common.config_loader import *
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common
import nf_robot.generated.nf.config as nf_config
from nf_robot.common.image_motion import image_shift, heading_error, mean_heading_error, wrap_angle
from nf_robot.host.data_store import DataStore
from nf_robot.host.stats import StatCounter
from nf_robot.host.eyelet_calibration import (optimize_arp_anchors, analyze_diamond_data,
                                             refinement_is_plausible, estimate_cam_tilts,
                                             DIAMOND_SIZE)
from nf_robot.host.component_client import max_origin_detections, parse_config_var
from nf_robot.host.derail_detector import DerailDetector
from nf_robot.host.arp_gripper_client import (ArpeggioGripperClient, rotate_vector,
                                              ROUTE_TAG_MAX_AGE_S, CAPTURE_RESOLUTION_SIZE,
                                              OPEN, CLOSED, RANGE_MAX_AGE_S)
from nf_robot.host import swing
from nf_robot.host.visual_servo import VisualServo, SERVO_MODE_GRASP, SERVO_MODE_OBSERVE, SERVO_MODE_CENTER
from nf_robot.host.arp_anchor_client import ArpeggioAnchorClient
from nf_robot.host.position_estimator import Positioner2
from nf_robot.host.telemetry_manager import TelemetryManager, LOCAL, normalize_control_plane_host
from nf_robot.host.webui_server import WebUiServer
from nf_robot.host.maneuver import (SafetyPolicy, OverTension, RESERVED_KEY_PREFIXES, ItemImage,
                                    DROP_POSITION_NAME, PREDICTED_DROP_NAME, ROUTE_POINT_TAG_NAMES)
from nf_robot.host.maneuvers import BUILTIN_MANEUVERS

logger = logging.getLogger(__name__)

# Define the service names for network discovery
arp_gripper_service_name = 'cranebot-gripper-arpeggio-service'
arp_anchor_service_name = 'cranebot-anchor-arpeggio-service'

N_ANCHORS = 2
N_LINES = 4
DEFAULT_MAX_SAFE_TENSION = 16.0  # newtons, when config.max_safe_tension says nothing
INPUT_VELOCITY_TTL_S = 2.0 # a commanded velocity keyed by a source expires this long after its last update
INFO_REQUEST_TIMEOUT_MS = 3000 # milliseconds
# visual centering nudges. The move is open loop (commanded speed for a computed duration), so
# the speed trades travel time against how much overshoot and swing each step leaves behind.
NUDGE_SPEED_MPS = 0.12
NUDGE_SETTLE_S = 0.3
NUDGE_STEP_S = 0.05   # how often the eased velocity is re-issued while a nudge runs
# (seconds) how long a nudge takes to reach its speed, and to come off it again. Comparable
# to a quarter of the pole's swing period, which is what makes the move start and stop
# without setting the gripper swinging under the gantry.
NUDGE_RAMP_S = 0.4
# The same easing for the wrist, which swings the gripper under the gantry when it starts
# or stops abruptly. Degrees per second at the top of the move, and the time spent getting
# there; a turn smaller than the minimum is left to the servo, since it cannot build a swing.
WRIST_EASE_DPS = 90.0
WRIST_RAMP_S = 0.4
WRIST_STEP_S = 0.05
WRIST_EASE_MIN_DEG = 5.0
# Velocity source keys of the observer's own. Inbound moves may not use the 'ob:' prefix,
# so none of these can be driven or zeroed from outside.
DEFAULT_VELOCITY_KEY = 'ob:default'
SWING_VELOCITY_KEY = 'ob:swingc'
NUDGE_VELOCITY_KEY = 'ob:centering'
SPIN_VELOCITY_KEY = 'ob:spin'
# The step names set_startup_sequence starts with, which is what an --auto_start robot did
# before the sequence was configurable.
DEFAULT_STARTUP_SEQUENCE = ('unpark', 'pick_and_place', 'park')
# Commands, ControlItem fields and Debug verbs AsyncObserver handles itself. A maneuver
# claiming one would never be reached, so add_maneuver refuses it.
BUILTIN_COMMANDS = frozenset({
    control.Command.STOP_ALL, control.Command.TIGHTEN_LINES, control.Command.HALF_CAL,
    control.Command.FULL_CAL, control.Command.SHUTDOWN, control.Command.RECORD_DROP,
    control.Command.GRASP, control.Command.UPDATE_FIRMWARE,
    control.Command.DISABLE_TORQUE, control.Command.ENABLE_TORQUE,
    control.Command.DEBUG_LOG_OVER_T, control.Command.ENABLE_TENSION_REG,
    control.Command.DISABLE_TENSION_REG, control.Command.SAFE_COMPONENT_SHUTDOWN,
})
BUILTIN_CONTROL_FIELDS = frozenset({
    'command', 'move', 'gantry_goal_pos', 'jog_spool', 'debug', 'set_swing_cancellation',
    'single_component_action', 'set_point', 'popup_ack', 'add_relay_creds',
})
BUILTIN_VERBS = frozenset({
    'spincal', 'spin-recover', 'fingercal', 'eyelets', 'gripcards', 'stow', 'upright', 'swinglatency',
    'swinglatencycal', 'polecal', 'reset_wrist', 'spind', 'sync_timezone', 'pull_logs',
    'untwist', 'setvar', 'savevar', 'spoollog', 'holdtension', 'tensionreg', 'findorigin', 'centerorigin',
    'servograsp', 'servowatch', 'servocenter', 'servoloop',
})
# (seconds) how far a wrist record may be from a frame's capture time and still describe
# where the wrist was when that frame was taken. Grip sensors arrive with the gripper's
# heartbeat, so in normal running the nearest record is milliseconds away; this only fires
# across a telemetry gap, where the honest move is to skip the correction.
TRIM_SPEED_MPS = 0.08 # altitude trim moves slower than a lateral nudge; it is closing centimeters
# (seconds) typical delay between a camera capturing a frame and its detections landing here.
# Frames carry their capture time, so this is not an offset to correct for: it is how long a
# step that wants a view of what it just did has to wait before the first such frame can arrive,
# and the margin by which a cutoff is pushed into the future so clock skew between the host and
# a component cannot let a frame from before the step slip past the filter.
VIDEO_LATENCY_S = 0.25
# monitor_gantry_visibility's cadence, and how long every camera may go without a new sighting
# of the gantry marker before that is called a fault rather than a gap.
VISIBILITY_POLL_S = 1.0
UNSEEN_LIMIT_S = 40.0
# (s) of spool records monitor_spools keeps, to write out with a derailment
SPOOL_HISTORY_S = 60.0

USER_TARGETS_DIR = "user_targets_data"
METADATA_PATH = os.path.join(USER_TARGETS_DIR, "metadata.jsonl")

# threshold of non slack tension in newtons for arp anchors
TENSION_THRESH = 1.38


# distance from the tip of the pole (self.pole[2] below the gantry) down to the bottom of
# the arp gripper fingers when they hang straight. gantry -> fingertip is self.pole[2] + this.
GRIPPER_FINGER_LEN_M = 0.18

# Laser range to an item under the gripper at which last_clear_item_image keeps a frame of it:
# near enough that the item fills the frame, far enough that the fingers are not across it.
# The drop point model was trained on frames from this band (placer
# mine_teleop.SNAPSHOT_RANGE_M).
CLEAR_ITEM_RANGE_M = (0.12, 0.25)
CLEAR_ITEM_INTERVAL_S = 0.25

# feature key -> minimum nf_robot version every connected component must run to use it
VERSION_GATES = {
    "speed_0.45": "4.1.0",
    "gripper_card_survey": "4.2.0",
}

# Which commit the published models are downloaded at lives in
# common/model_revisions.json, written by nf_robot.ml.pin_latest_model. Same kind of
# declaration as VERSION_GATES above - what this build expects to run against - kept as
# data because bumping a pin should read as a data change. --local_models ignores it and
# reads models/ instead, which is how a checkpoint gets flown before it is pinned.

def host_nf_robot_version():
    """This host's installed nf_robot version, or None when it runs from a source tree that was
    never installed and so has no package metadata to read."""
    try:
        return importlib.metadata.version('nf_robot')
    except importlib.metadata.PackageNotFoundError:
        return None

def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def _robust_spread(points):
    """Median distance from the median position: how far apart a handful of position samples
    are, unmoved by the one bad detection a standard deviation would let dominate."""
    P = np.asarray(points, dtype=float)
    return float(np.median(np.linalg.norm(P - np.median(P, axis=0), axis=1)))


def _widest_gap(points):
    """The largest distance between any two of these positions."""
    P = np.asarray(points, dtype=float)
    return float(max(np.linalg.norm(a - b) for a, b in itertools.combinations(P, 2)))


def _spiral_waypoints(step, max_r):
    """(x, y) offsets along an Archimedean spiral out to max_r, starting one step from the
    center. Points sit about step apart along the path and the spiral gains step per turn, so
    a downward camera whose footprint is step wide sweeps the whole disc as it follows them."""
    points = []
    theta = 0.0
    r = step
    while r <= max_r:
        points.append((r * np.cos(theta), r * np.sin(theta)))
        dtheta = step / r
        theta += dtheta
        r += step * dtheta / (2 * np.pi)
    return points


def eased_speed(elapsed, total, ramp):
    """Speed as a fraction of the peak, for a move that eases in and out over `ramp` seconds.

    Raised cosine at both ends, flat in between. A move that starts and stops abruptly kicks
    the pole, and the swing that leaves behind outlasts the move by a long way - the gripper
    camera hangs off that pole, so it is the difference between a sharp photograph and a
    smeared one. Distance works out to peak * (total - ramp), which is what the callers size
    `total` from.
    """
    if elapsed <= 0.0 or elapsed >= total:
        return 0.0
    if elapsed < ramp:
        return 0.5 * (1 - np.cos(np.pi * elapsed / ramp))
    if elapsed > total - ramp:
        return 0.5 * (1 - np.cos(np.pi * (total - elapsed) / ramp))
    return 1.0


def eased_move_time(dist, peak, ramp):
    """(total duration, ramp duration) for an eased move of dist at peak speed.

    A move too short to reach the peak gets a shorter ramp and never does, which keeps the
    distance right instead of overshooting it.
    """
    ramp = min(ramp, dist / peak)
    return dist / peak + ramp, ramp


def with_swing_cancellation_preferred(func):
    """Decorate an AsyncObserver coroutine method to run under prefer_swing_cancellation.

    AsyncObserver.prefer_swing_cancellation is the primitive and says what the preference
    means; this is that primitive applied to a whole method, for the long tasks that want it
    from their first move to their last and would otherwise have to indent their entire body
    to say so. Keeps __name__, which is what invoke_motion_task names the motion task by.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        async with self.prefer_swing_cancellation():
            return await func(self, *args, **kwargs)
    return wrapper


class TelemetryLogHandler(logging.Handler):
    """Forwards log records to the telemetry stream via send_ui."""

    def __init__(self, observer):
        super().__init__()
        self._observer = observer

    def emit(self, record):
        try:
            line = self.format(record)
            self._observer.send_ui(logs=telemetry.Logs(line=[line]))
        except Exception:
            self.handleError(record)


class AsyncObserver:
    """
    Manager of multiple tasks running clients connected to each robot component
    The job of this class in a nutshell is to discover four anchors and a gripper on the network,
    connect to them, and forward data between them and the position estimator, shape tracker, and UI.

    It reads from the config file to find any components it already knows about.
    It starts zeroconf to discover any components it doesn't know about and add them to the config.
    it starts keep_robot_connected to continually reconnect to all known components.
    It starts position_estimator to continually run kalman filters on the observed variables.
    It starts run_perception to continually run inference on the camera feeds.
    It hands every telemetry item to the TelemetryManager, which owns the local websocket
    server and the cloud relay link and hands inbound control messages back here.

    It reads from the config file to find any components it already knows about.
    It starts zeroconf to discover any components it doesn't know about and add them to the config.
    As soon as a component in the config has a known address, it starts keep_robot_connected to continually reconnect to all known components.
    As soon as the first component websocket is connected, It starts position_estimator to continually run kalman filters on the observed variables.
    As soon as a feed from the first preferred camera is up, It starts run_perception to continually run inference on the camera feeds.

    Since this class serves as the coordination center of all the robot compnents, it also contains methods to perform
    various actions like calibration and the pick and place routine.

    Other high-level behaviors are maneuvers (see nf_robot.host.maneuver), registered with
    add_maneuver. A maneuver may call any method here without a leading underscore, and
    reads sensors and positions through the getters (gantry_position, laser_range,
    gripper_frame, ...) rather than through pe, datastore, gripper_client or anchors, which
    together with calibration and the robot geometry are the observer's own.
    """
    def __init__(self, terminate_with_ui, config_path, telemetry_env=None, run_ortho=True, auto_start=False, local_models=False, port=4245, debug=False, bind_address="127.0.0.1", rec_diagnostics=False, serve_ui=True, ui_port=8090, diamond_size=DIAMOND_SIZE, lerobot_grasp=False) -> None:
        self.port = port
        # (half height, half width, floor clearance) of the calibration diamond, in meters.
        # Overridable with --diamond_size. Consumed both by the physical diamond motion in
        # collect_arp_anchor_eyelet_experiment_data and by optimize_arp_anchors.
        self.diamond_size = diamond_size
        # Interface the local telemetry websocket and all local mjpeg video streams bind to.
        # Defaults to loopback (single-machine use). Set to a LAN IP or 0.0.0.0 to let a
        # record/eval client on another machine connect. See src/nf_robot/ml/README.md.
        self.bind_address = bind_address
        self.serve_ui = serve_ui
        self.ui_port = ui_port
        self.terminate_with_ui = terminate_with_ui
        self.position_update_task = None
        self.aiobrowser: AsyncServiceBrowser | None = None
        self.aiozc: AsyncZeroconf | None = None
        self.run_command_loop = True
        self.datastore = DataStore()
        self.pool = None
        # all clients by server name
        self.bot_clients = {}
        # all connected anchors keyed by anchor num
        self.anchors = {}
        # per line, the last (time, value, 'aim'|'jog'|'stop') sent to its spool, for the spool log
        self.line_speed_cmds = [None] * N_LINES
        # the last SPOOL_HISTORY_S of what monitor_spools read, written out if it sees a derailment
        self.spool_history = deque(maxlen=int(SPOOL_HISTORY_S / 0.1))
        # the open file start_spool_log is writing everything to, if any
        self.spool_log = None
        self.spool_monitor_task = None
        # convenience reference to gripper client
        self.gripper_client = None
        # TODO allow a command line argument to override the config file path
        self.config_path = config_path
        self.config = load_config(config_path)
        self.apply_pole_geometry()
        self.telemetry_env = telemetry_env
        self.debug = debug
        self.loop_monitor = None  # only created in main() when --debug is passed
        # when set, full_auto_calibration pickles the args of every optimize_arp_anchors call
        # (Arpeggio hardware only) to calibration_diagnostics.pkl for offline analysis.
        self.rec_diagnostics = rec_diagnostics
        self._calibration_diagnostics = []
        # (percent_complete, current_action) of the last calibration step reported to the UI,
        # captured in send_ui so that every step counts, including the ones sent from the
        # helpers calibration calls rather than from full_auto_calibration itself. Read only
        # when a run aborts, to record what it was doing at the time.
        self._calibration_step = (0.0, None)
        self.stat = StatCounter(self)
        self.enable_shape_tracking = False
        self.shape_tracker = None
        # Position Estimator. this used to be a seperate process so it's still somewhat independent.
        self.pe = Positioner2(self.datastore, self)
        self.locate_anchor_task = None
        # only one motion task can be active at a time
        self.motion_task = None
        # where seek_goal is steering, and the flight doing it. Shared by every caller of
        # seek_goal, so a second call re-aims the flight instead of starting another.
        self._goal_pos = None
        self._seek_task = None
        # the maneuver the running motion task belongs to (None for the observer's own), and
        # the policy the safety monitors apply to it
        self._motion_owner = None
        self._motion_safety = SafetyPolicy()
        # set by passive_safety when line tension exceeds the safe limit during a running
        # motion task. Swing latency cal runs under OverTension.NOTIFY and polls it to back
        # off and retry the current trial.
        self.tension_over_limit = False
        # onboard tension regulation (floor + soft mute) on/off state, mirrored to the UI
        # via tension_regulation_state whenever it changes. Both this and torque below
        # start on because that is what a spool comes up in (see spool_dm); nothing
        # commands either at startup, so anything else would be a wrong initial guess.
        self.tension_reg_enabled = True
        # motor torque on/off state, mirrored to the UI via torque_state whenever it
        # changes. Sourced from what the anchors report rather than what was commanded.
        self.torque_enabled = True
        # set while passive_safety cycles torque to shed an over-tension. That is a safety
        # action, not an operator one, so it is kept out of the reported torque state.
        self._torque_reports_suppressed = False
        # set by monitor_gantry_visibility to the phrase describing why it aborted a running
        # calibration, so the calibration's own cancel handler can report the real reason.
        self.gantry_marker_fault = None
        # set while full calibration has not yet stood the pole upright. A leaning pole can
        # face the marker away from every camera, so it going unseen until then is expected.
        self._marker_may_be_hidden = False
        # which gantry marker faults have already been reported, so a standing fault is not a
        # repeating popup. A key is dropped when its condition clears, and the whole set is
        # cleared when a calibration starts, so a re-run of a calibration that was aborted by
        # a fault says so again instead of proceeding quietly.
        self._gantry_marker_warned = set()
        # keys whose popup the operator has already seen. Never cleared, so a fault that keeps
        # coming back still aborts calibration but stops interrupting with the same message.
        self._gantry_marker_popped = set()
        # only used for integration test only to allow some code to run right after sending the gantry to a goal point
        self.test_gantry_goal_callback = None
        # event used to notify tasks that gripper is connected.
        self.gripper_client_connected = asyncio.Event()
        self.last_user_move_time = time.time()
        # last known positions of named tags/objects live in self.config.named_positions
        # (the single source of truth). It's written to disk on shutdown, in async_close.
        # Grasps with the visual servoing model, which is how grasping works unless
        # --lerobot_grasp hands it to a policy instead. Holds the checkpoint, loaded on
        # first use.
        self.servo = VisualServo(self)
        self.perception_task = None
        self.webui_server = None
        # owns every telemetry destination: the local websocket server and the cloud relay
        # link. Constructed here rather than in main() so send_ui works before the sockets
        # are up. Both transports also carry inbound control, hence the callbacks.
        self.telemetry = TelemetryManager(
            config=self.config,
            telemetry_env=telemetry_env,
            bind_address=bind_address,
            port=port,
            on_control_message=self.handle_command,
            on_peer_connected=self._on_telemetry_peer_connected,
            on_peer_disconnected=self._on_telemetry_peer_disconnected,
        )
        self.startup_complete = asyncio.Event()
        self.any_anchor_connected = asyncio.Event() # fires as soon as first anchor connects, starting pe
        self.gip_task = None
        self.passive_safety_task = None
        self.gantry_visibility_task = None
        # last attempt to connect, keyed by service name
        self.connection_tasks: dict[str, asyncio.Task] = {}
        self.time_last_grip_sensors_retain_key = 0
        # {key: (velocity, monotonic_timestamp)} last velocities commanded by different subsystems. all keys in active_set are summed.
        # Entries expire INPUT_VELOCITY_TTL_S after their last update; expiration is lazy (pruned at read time in move_direction_speed),
        # so a source key that stops sending moves stops contributing without needing any timer or background task.
        self.input_velocities = {DEFAULT_VELOCITY_KEY: (np.zeros(3), time.monotonic())}
        self.active_set = {DEFAULT_VELOCITY_KEY}
        self.run_ortho = run_ortho
        self.auto_start = auto_start
        self._device = None
        self._telem_log_handler: TelemetryLogHandler | None = None
        self.swing_cancellation_task = None
        self.local_models = local_models
        # ortho projection state - written by _ortho_worker thread, read by run_perception AI task
        self.ortho_event = threading.Event()
        # rgb24, the order the anchor clients decode to; only converted to BGR for the streamer
        self.last_ortho_rgb = None
        # see last_clear_item_image
        self._clear_item_image = None
        self._clear_item_task = None
        # list of (NfVideoStreamer, feed_number) for ortho feeds, so send_setup_telemetry can replay them
        self.ortho_streamers: list = []
        # futures awaiting a PopupAck, keyed by the Popup.id they were sent with
        self.pending_popup_acks: dict[int, asyncio.Future] = {}
        self._next_popup_id = 1
        # source and destination for pick and place. self.config is the source of truth;
        # these are kept in sync with self.config.last_route_source/last_route_destination.
        self.pnp_src = self.config.last_route_source
        self.pnp_dst = self.config.last_route_destination
        # where maneuvers write files of their own (Maneuver.output_dir)
        self.output_root = Path('.')
        # registered maneuvers by name, and what each one answers to
        self.maneuvers = {}
        self._maneuver_commands = {}    # control.Command -> (handler, motion, safety, owner)
        self._maneuver_controls = {}    # ControlItem field name -> handler
        self._maneuver_verbs = {}       # first word of a Debug action -> (handler, motion, safety, owner)
        self._startup_steps = {}        # step name -> (coroutine function, safety, owner)
        self.startup_sequence_names = list(DEFAULT_STARTUP_SEQUENCE)
        builtin_options = {'lerobot': {'use_for_grasp': lerobot_grasp}}
        for maneuver_class in BUILTIN_MANEUVERS:
            self.add_maneuver(maneuver_class, **builtin_options.get(maneuver_class.name, {}))

    def add_maneuver(self, maneuver_class, **options):
        """Construct a maneuver, route everything it declared to it, and return it.

        Refuses a name that is taken, and a command, control field, verb or startup step that
        the observer or another maneuver already answers to, since only one could ever run.
        Call before main().
        """
        maneuver = maneuver_class(self, **options)
        if not maneuver.name:
            raise ValueError(f'{maneuver_class.__name__} has no name')
        if maneuver.name in self.maneuvers:
            raise ValueError(f'A maneuver named {maneuver.name!r} is already registered')

        commands, controls, verbs, steps = {}, {}, {}, {}
        for attr in dir(type(maneuver)):
            entries = getattr(getattr(type(maneuver), attr, None), '_maneuver_entries', ())
            for kind, key, motion, safety in entries:
                handler = getattr(maneuver, attr)
                safety = safety or maneuver.safety
                if kind == 'command':
                    if key in BUILTIN_COMMANDS or key in self._maneuver_commands or key in commands:
                        raise ValueError(f'{maneuver.name}: command {key!r} is already handled')
                    commands[key] = (handler, motion, safety, maneuver)
                elif kind == 'control_item':
                    if key in BUILTIN_CONTROL_FIELDS or key in self._maneuver_controls or key in controls:
                        raise ValueError(f'{maneuver.name}: control item {key!r} is already handled')
                    controls[key] = handler
                elif kind == 'verb':
                    if key in BUILTIN_VERBS or key in self._maneuver_verbs or key in verbs:
                        raise ValueError(f'{maneuver.name}: debug verb {key!r} is already handled')
                    verbs[key] = (handler, motion, safety, maneuver)
                elif kind == 'startup_step':
                    if key in self._startup_steps or key in steps:
                        raise ValueError(f'{maneuver.name}: startup step {key!r} already exists')
                    steps[key] = (handler, safety, maneuver)

        self.maneuvers[maneuver.name] = maneuver
        self._maneuver_commands.update(commands)
        self._maneuver_controls.update(controls)
        self._maneuver_verbs.update(verbs)
        self._startup_steps.update(steps)
        return maneuver

    def maneuver(self, name):
        """The registered maneuver of that name."""
        return self.maneuvers[name]

    def set_startup_sequence(self, names):
        """The startup steps an --auto_start robot runs, in order, once every component is
        connected. Names are checked when main() starts, so maneuvers can be added after."""
        self.startup_sequence_names = list(names)

    def _check_startup_sequence(self):
        unknown = [n for n in self.startup_sequence_names if n not in self._startup_steps]
        if unknown:
            raise ValueError(f'Unknown startup steps {unknown}; known steps are '
                             f'{sorted(self._startup_steps)}')

    async def _run_maneuver_handler(self, handler, motion, safety, owner, *args):
        if motion:
            return await self.invoke_motion_task(handler(*args), owner=owner, safety=safety)
        result = handler(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    @contextmanager
    def override_safety(self, **changes):
        """Change the running motion task's SafetyPolicy until the block exits."""
        previous = self._motion_safety
        self._motion_safety = replace(previous, **changes)
        try:
            yield
        finally:
            self._motion_safety = previous

    @contextmanager
    def _running_as(self, owner, safety):
        """Attribute the running motion task to owner under safety until the block exits,
        for a step that runs inside a motion task rather than as one."""
        previous = self._motion_owner, self._motion_safety
        self._motion_owner, self._motion_safety = owner, safety
        if owner is not None:
            owner.abort_reason = None
        try:
            yield
        finally:
            self._motion_owner, self._motion_safety = previous

    async def _start_maneuvers(self):
        for maneuver in self.maneuvers.values():
            try:
                await maneuver.start()
            except Exception:
                logger.exception(f'Maneuver {maneuver.name} failed to start')

    async def _stop_maneuvers(self):
        for maneuver in self.maneuvers.values():
            try:
                await maneuver._cancel_spawned()
                await maneuver.stop()
            except Exception:
                logger.exception(f'Maneuver {maneuver.name} failed to stop cleanly')

    def apply_pole_geometry(self):
        """Re-derive everything the configured pole decides.

        What the pole affects on this robot: how far the gripper hangs below the gantry,
        which marker the gantry has, and the pendulum it swings as. All four are cached
        rather than looked up per use, so swapping the pole at runtime has to come back
        through here or the robot keeps flying the old geometry.
        """
        self.pole_geometry = model_constants.pole_geometry(self.config)
        self.pole = np.array([0, 0, self.pole_geometry.gantry_to_gripper])
        self.gantry_april_inv = invert_pose(self.pole_geometry.gantry_april)
        self.pendulum = swing.pendulum_for(self.config)

    def send_anchor_poses(self):
        """Push the stored poses and the setup values that ride with them.

        Only arpeggio anchors have eyelets and tilt adapters to report. The pole goes in
        either way: every robot hangs from one. So does the host's own version, which is how
        a UI tells which setup steps this host still needs it to ask about.
        """
        host_version = host_nf_robot_version()
        if self.config.anchor_type == common.AnchorType.ARPEGGIO:
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                poses=[a.pose for a in self.config.anchors],
                eyelets=[a.indirect_line.eyelet_pos for a in self.config.anchors],
                tilt=[a.indirect_line.cam_tilt for a in self.config.anchors],
                swing_latency=self.config.swing_latency,
                calibrated=self.config.calibrated_status,
                pole_type=self.config.gripper.pole_type,
                host_version=host_version,
            ))
        else:
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                poses=[a.pose for a in self.config.anchors],
                calibrated=self.config.calibrated_status,
                pole_type=self.config.gripper.pole_type,
                host_version=host_version,
            ))

    async def send_setup_telemetry(self):
        logger.debug('Sending setup telemetry')
        self.send_anchor_poses()
        for maneuver in self.maneuvers.values():
            try:
                maneuver.send_setup_telemetry()
            except Exception:
                logger.exception(f'Maneuver {maneuver.name} failed to send setup telemetry')
        for name, position in self.config.named_positions.items():
            self.send_ui(named_position=telemetry.NamedObjectPosition(
                name = name,
                position = position
            ))
        for client in self.bot_clients.values():
            client.send_conn_status()
            if (client.local_video_uri is not None or client.remote_stream_path is not None) and client.anchor_num in [None, *self.config.preferred_cameras]:
                self.send_ui(video_ready=telemetry.VideoReady(
                    is_gripper=client.anchor_num is None,
                    anchor_num=client.anchor_num,
                    local_uri=client.local_video_uri,
                    feed_number=client.feed_number,
                    stream_path=client.remote_stream_path,
                ))
        for vs, feed_number in self.ortho_streamers:
            if vs._ready_sent:
                self.send_ui(video_ready=telemetry.VideoReady(
                    is_gripper=None,
                    anchor_num=None,
                    local_uri=vs.local_uri,
                    stream_path=vs.stream_path,
                    feed_number=feed_number,
                ))
        self.send_ui(task_status=telemetry.TaskStatus(
            route_source=self.pnp_src, route_destination=self.pnp_dst,
        ))
        self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=(SWING_VELOCITY_KEY in self.active_set), present='.'))
        self.send_ui(tension_regulation_state=telemetry.TensionRegulationState(enabled=self.tension_reg_enabled, present=True))
        self.send_ui(torque_state=telemetry.TorqueState(enabled=self.torque_enabled, present=True))
        r = await self.flush_tele_buffer()

    async def _on_telemetry_peer_connected(self, peer):
        """A local UI/lerobot session or the cloud relay just connected. Bring it up to date
        before it starts issuing commands."""
        r = await self.send_setup_telemetry()

    async def _on_telemetry_peer_disconnected(self, peer, local_remaining):
        self.zero_input_velocities()
        if peer == LOCAL and local_remaining == 0 and self.terminate_with_ui:
            # The only local UI has disconnected and we were asked to shutdown when it disconnects
            self.run_command_loop = False

    def zero_input_velocities(self):
        """ Reset all commanded velocities to zero.

        Called when a websocket connection (local UI or control plane) is lost so
        that the last velocity commanded from a now-disconnected source key does
        not keep driving the robot indefinitely. Since source keys are arbitrary
        and not tracked per-connection, we clear them all; subsystems like swing
        cancellation recompute their entry on the next tick.
        """
        self.input_velocities = {DEFAULT_VELOCITY_KEY: (np.zeros(3), time.monotonic())}

    def _prune_input_velocities(self):
        """ Lazily drop commanded velocities older than INPUT_VELOCITY_TTL_S.

        Called at read time (from move_direction_speed) rather than on a timer, so
        stale source keys are cleaned up as a side effect of the next combined move.
        The common case where nothing has expired is a cheap scan with no deletions.
        """
        now = time.monotonic()
        expired = [k for k, (_, ts) in self.input_velocities.items() if now - ts > INPUT_VELOCITY_TTL_S]
        for k in expired:
            del self.input_velocities[k]

    async def handle_command(self, message: bytes):
        """ Decodes a binary batch of commands """
        # betterproto .parse() returns a standard python dataclass
        batch = control.ControlBatchUpdate().parse(message)
        for update in batch.updates:
            r = await self._dispatch_update(update)

    async def _dispatch_update(self, item: control.ControlItem):
        # In betterproto2, 'oneof' fields appear as attributes. 
        # Only one will be non-None.
        # not that checking if the field is truthy is insufficient, as a default instance of the proto is false
        # and default instances can carry meaningful information such as zeroing out a value.
        
        # Standard Commands (Stop, Calibrate, Zero)
        if item.command is not None:
            r = await self._handle_common_command(item.command.name)

        # Movement Vector (Gamepad/AI Policy)
        elif item.move is not None:
            r = await self._handle_movement(item.move)

        # Setting gantry goal
        elif item.gantry_goal_pos is not None:
            r = await self._handle_gantry_goal_pos(tonp(item.gantry_goal_pos.pos))

        # Manual Spool Control
        elif item.jog_spool is not None:
            r = await self._handle_jog_spool(item.jog_spool)

        elif item.debug is not None:
            r = await self._handle_debug_command(item.debug)

        elif item.set_swing_cancellation is not None:
            r = await self._handle_set_swing_cancellation(item.set_swing_cancellation)

        elif item.single_component_action is not None:
            r = await self._handle_single_component_action(item.single_component_action)

        elif item.set_point is not None:
            asyncio.create_task(self._handle_set_point(item.set_point))

        elif item.popup_ack is not None:
            self._handle_popup_ack(item.popup_ack)

        elif item.add_relay_creds is not None:
            self._handle_add_relay_creds(item.add_relay_creds)

        else:
            for field, handler in self._maneuver_controls.items():
                payload = getattr(item, field, None)
                if payload is not None:
                    r = await self._run_maneuver_handler(handler, False, None, None, payload)
                    break

    async def _handle_set_point(self, item: control.SetPoint):
        """Set either the route source or destination (the To: and From: fields in the UI)"""
        logger.debug(f'_handle_set_point {item}')
        self.set_route(source=item.route_source or None, destination=item.route_destination or None)
        r = await self.flush_tele_buffer()

    def route(self):
        """The (source, destination) RoutePoints things are carried between."""
        return self.pnp_src, self.pnp_dst

    def set_route(self, source=None, destination=None):
        """Change either end of the route, save it, and show it in the UI's To: and From:."""
        if source is not None:
            self.pnp_src = source
            self.config.last_route_source = source
        if destination is not None:
            self.pnp_dst = destination
            self.config.last_route_destination = destination
        save_config(self.config, self.config_path)
        self.send_ui(task_status=telemetry.TaskStatus(
            route_source=self.pnp_src, route_destination=self.pnp_dst,
        ))

    def route_point_position(self, route_point):
        """Floor position of a route point, or None if it has none.

        Quiet about failures: this is consulted every targeting round, and NA (drop where
        each target says) genuinely has no single position.
        """
        if route_point == common.RoutePoint.ORIGIN:
            return np.zeros(3)
        name = ROUTE_POINT_TAG_NAMES.get(route_point)
        return self.named_position(name) if name is not None else None

    def named_position(self, name):
        """Last known room position of a named place, or None if it has never been seen."""
        if name not in self.config.named_positions:
            return None
        return tonp(self.config.named_positions[name])

    def set_named_position(self, name, position, save=True):
        """Remember a named place, show it in the UI, and write it out unless save is False."""
        self.config.named_positions[name] = fromnp(np.asarray(position, dtype=float))
        if save:
            save_config(self.config, self.config_path)
        self.send_ui(named_position=telemetry.NamedObjectPosition(
            position=fromnp(np.asarray(position, dtype=float)), name=name))

    async def _handle_single_component_action(self, item: control.SingleComponentAction):
        """Issue a special command to a single component"""
        client = None
        if item.is_gripper:
            client = self.gripper_client
        else:
            client = self.anchors.get(item.anchor_num, None)
        if client is not None:
            if item.action == control.ComponentAction.REBOOT:
                r = await client.send_commands({'reboot': None})
            elif item.action == control.ComponentAction.IDENTIFY:
                r = await client.send_commands({'identify': None})
            elif item.action == control.ComponentAction.TIGHTEN:
                r = await client.send_commands({'tighten': None})
            elif item.action == control.ComponentAction.RELAX:
                r = await client.send_commands({'relax': None})
            elif item.action == control.ComponentAction.SET_CAM_ANGLE and self.config.anchor_type == common.AnchorType.ARPEGGIO:
                self.config.anchors[item.anchor_num].indirect_line.cam_tilt = item.cam_angle
                save_config(self.config, self.config_path)
                self.anchors[item.anchor_num].updatePoseAndEye()
                self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                    poses=[a.pose for a in self.config.anchors],
                    eyelets=[a.indirect_line.eyelet_pos for a in self.config.anchors],
                    tilt=[a.indirect_line.cam_tilt for a in self.config.anchors],
                    swing_latency=self.config.swing_latency,
                    pole_type=self.config.gripper.pole_type,
                ))
            elif item.action == control.ComponentAction.SET_POLE_TYPE and item.pole_type is not None:
                await self.set_pole_type(item.pole_type)
            elif item.action == control.ComponentAction.SHUTDOWN:
                name = 'the gripper' if item.is_gripper else f'anchor {item.anchor_num}'
                await self.shutdown_one_bot(client, name)

    async def set_pole_type(self, pole_type: common.PoleType):
        """Record which pole is installed and re-derive everything hanging off it.

        Saved rather than held for the session: the pole is a property of the robot, and
        the calibration this precedes is stored against the geometry chosen here.

        The gripper fits its own swing model, so it needs the new length too - it is told
        once on connect and would otherwise keep the old frequency until it reboots.
        """
        if self.config.gripper.pole_type == pole_type:
            return
        self.config.gripper.pole_type = pole_type
        save_config(self.config, self.config_path)
        self.apply_pole_geometry()

        gc = self.gripper_client
        if gc is not None:
            gc.pendulum = swing.pendulum_for(self.config)
            await gc.send_config()
        logger.info(f'Pole type set to {pole_type.name}, '
                    f'swing length now {self.pole_geometry.swing_length:.3f}m')
        self.send_anchor_poses()

    def set_swing_cancellation(self, enabled: bool) -> bool:
        """Start or stop the swing cancellation task, idempotently.

        Enabling when it is already running (or disabling when it is already stopped) is a
        no-op, so callers can just declare the state they want. Returns whether the task was
        running before this call, which lets a caller decide if it needs to restart it later.
        """
        was_running = self.swing_cancellation_task is not None and not self.swing_cancellation_task.done()
        if enabled and not was_running:
            self.swing_cancellation_task = asyncio.create_task(self.run_swing_cancellation())
        elif not enabled and was_running:
            self.swing_cancellation_task.cancel()
        return was_running

    @asynccontextmanager
    async def prefer_swing_cancellation(self):
        """Run a block with swing cancellation on, for a robot that has one which works.

        Only a calibration run that watched a test swing actually damp sets
        swing_cancellation_verified, so a robot whose check failed, or one calibrated before
        the check existed, is left alone instead. That is the conservative half of the trade:
        cancellation drives the spools from the gripper's IMU, and where the geometry or the
        latency is off it pumps the swing rather than damping it, which is worse than the
        swing it was asked to remove.

        The state on the way in is restored on the way out, so this states a preference for
        the length of the block without overriding the operator's switch."""
        if not self.config.swing_cancellation_verified:
            logger.debug('Swing cancellation not verified on this robot; leaving it as it is')
            yield
            return
        if self.gripper_client is None:
            # it reads the gripper's IMU, so without one it has nothing to cancel from
            logger.debug('No gripper connected; leaving swing cancellation as it is')
            yield
            return
        was_running = self.set_swing_cancellation(True)
        try:
            yield
        finally:
            self.set_swing_cancellation(was_running)

    async def _handle_set_swing_cancellation(self, item: control.SetSwingCancellation):
        logger.info(f'Swing cancellation set {item.enabled}')
        if item.enabled:
            if self.gripper_client is None:
                self.send_ui(pop_message=telemetry.Popup(
                    message=f'Swing cancellation requires a connected gripper'
                ))
                return
        self.set_swing_cancellation(item.enabled)

    async def run_swing_cancellation(self):
        """ Task which adds swing cancellation inputs. """

        # config.swing_latency is the round trip time between an IMU measurement on the
        # gripper and our input moving the spools. Tune it with calibrate_swing_latency
        # (the 'swinglatencycal' debug command). It varies by host machine.
        # If cancellation seems wonky, the gripper may have a different timezone than the
        # host; run the sync_timezone debug command to fix.
        try:
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=True, present='.'))
            r = await self.flush_tele_buffer()
            self.active_set.add(SWING_VELOCITY_KEY)
            while self.run_command_loop:
                if self.gripper_client is None:
                    await asyncio.sleep(1)
                    continue
                vel2 = self.gripper_client.compute_swing_correction(time.time() + self.config.swing_latency)
                if vel2 is not None:
                    await self.move_direction_speed(np.array([vel2[0], vel2[1], 0]), key=SWING_VELOCITY_KEY, downward_bias=0)
                await asyncio.sleep(1/100)
        except asyncio.CancelledError:
            pass
        finally:
            self.active_set.remove(SWING_VELOCITY_KEY)
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=False, present='.'))
            r = await self.flush_tele_buffer()
            self.slow_stop_all_spools()

    async def _induce_swing(self, direction=np.array([1.0, 0.0, 0.0]), cycles=2, speed=0.05):
        """Pump the gripper into a swing by driving the gantry back and forth at
        the pendulum's resonant frequency.
        """
        direction = np.asarray(direction, dtype=float)
        try:
            for _ in range(cycles):
                await self.move_direction_speed(direction, speed, downward_bias=0)
                await asyncio.sleep(self.pendulum.half_period)
                await self.move_direction_speed(-direction, speed, downward_bias=0)
                await asyncio.sleep(self.pendulum.half_period)
        finally:
            self.slow_stop_all_spools()

    async def measure_pendulum_length(self, decay_s=None):
        """Measure what the gripper actually swings as, and report it against the config.

        The pole a robot has is recorded in config.gripper.pole_type, but the number behind
        it is an effective pendulum length: the gripper is a body with its own moment of
        inertia, not a weight on a string, so the length that sets the swing frequency is
        not something to measure with a tape. Swinging it and reading the frequency back
        off the gyro is. Use this after changing a pole, a marker, or anything that moves
        the gripper's mass, and put the answer in definitions.POLE_GEOMETRY.

        The gripper's published swing model is no use here: it is fitted assuming the
        configured frequency, so it would only ever confirm what it was told. This records
        the raw gyro instead, over a free decay with nothing cancelling or driving it.

        This is a motion task. Run it hanging clear, with room to swing.
        """
        # long enough for the spectrum to resolve the peak, short enough that the swing has
        # not decayed into the noise by the end of it
        decay_s = decay_s or 20.0

        gc = self.gripper_client
        if gc is None:
            logger.warning('Measuring the pendulum requires a connected gripper')
            return None

        def report(action):
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=100.0, name="Measure Pendulum", current_action=action))

        # anything still driving the gantry would show up in the gyro as a second frequency
        was_cancelling = self.set_swing_cancellation(False)
        try:
            await gc.record_raw_gyro(True)
            logger.info('Inducing a swing to measure the pendulum')
            report('Inducing a swing...')
            await self._induce_swing()
            logger.info(f'Letting it swing freely for {decay_s:.0f}s')
            report(f'Recording a free swing for {decay_s:.0f}s...')
            await asyncio.sleep(decay_s)
        finally:
            await gc.record_raw_gyro(False)
            self.slow_stop_all_spools()
            if was_cancelling:
                self.set_swing_cancellation(True)
        # the last samples are still in flight when recording stops
        await asyncio.sleep(0.5)

        samples = gc.collect_raw_gyro()
        freq, length = swing.measure_pendulum(samples)
        if freq is None:
            message = (f'No swing found in {len(samples)} gyro samples. Is the IMU '
                       f'installed, and did the gripper have room to swing?')
            logger.warning(message)
            report(message)
            return None

        configured = self.pendulum.length
        logger.info(f'Measured swing {freq:.4f} Hz ({1 / freq:.3f}s period) from '
                    f'{len(samples)} gyro samples over {samples[-1, 0] - samples[0, 0]:.1f}s')
        logger.info(f'Effective pendulum length {length:.4f} m; configured '
                    f'{configured:.4f} m ({(length - configured) * 1000:+.0f} mm)')
        report(f'{freq:.4f} Hz, effective length {length:.4f} m '
               f'(configured {configured:.4f} m, {(length - configured) * 1000:+.0f} mm off)')
        return length

    async def _identify_pole_type(self, record_s=None):
        """Set config.gripper.pole_type from how the gripper actually swings.

        The pole sets the frequency every swing-latency trial is phased against, so it is
        worth reading rather than trusting: tuned against the wrong pole, the correction
        lands a fraction of a period out and nothing damps well. The gripper's published
        swing model is no use for this, being fitted at the configured frequency, so this
        times a free swing off the raw gyro the way measure_pendulum_length does.

        Only the CARBON270 has to come out right - see swing.nearest_pole_type. Returns the
        pole it settled on, or None when the swing could not be read, leaving the configured
        pole standing. This is a motion task and induces a swing.
        """
        # long enough for measure_swing_frequency to have the swings it wants to time,
        # short enough not to add much to a calibration that already induces one per trial
        record_s = record_s or 12.0

        gc = self.gripper_client
        if gc is None:
            return None

        # anything still driving the gantry would put a second frequency in the gyro
        was_cancelling = self.set_swing_cancellation(False)
        try:
            # Recording starts after the induction, not before it. _induce_swing pumps at
            # the configured half period, so on a misconfigured pole it drives off
            # resonance, and leaving that in the record would offer the spectrum a peak at
            # the very frequency this is trying not to take on faith. The free swing that
            # follows is at the pole's own frequency whatever pumped it.
            await self._induce_swing()
            await gc.record_raw_gyro(True)
            await asyncio.sleep(record_s)
        finally:
            await gc.record_raw_gyro(False)
            self.slow_stop_all_spools()
            if was_cancelling:
                self.set_swing_cancellation(True)
        # the last samples are still in flight when recording stops
        await asyncio.sleep(0.5)

        samples = gc.collect_raw_gyro()
        freq, length = swing.measure_pendulum(samples)
        if freq is None:
            logger.warning(f'Pole identification: no swing in {len(samples)} gyro samples; '
                           f'keeping the configured pole')
            return None

        pole_type, mismatch = swing.nearest_pole_type(length)
        if pole_type is None:
            logger.warning(f'Pole identification: {freq:.3f} Hz is an effective length of '
                           f'{length:.3f} m, {mismatch * 1000:.0f} mm from the nearest pole; '
                           f'keeping the configured pole')
            return None

        was = self.config.gripper.pole_type
        logger.info(f'Pole identification: {freq:.3f} Hz, effective length {length:.3f} m, '
                    f'nearest pole {pole_type.name} ({mismatch * 1000:.0f} mm off)')
        if pole_type != was:
            logger.warning(f'Pole identification: measured {pole_type.name} where '
                           f'{was.name} was configured. Whatever this run fitted before now '
                           f'used the old pole, so rerun calibration after it finishes.')
        await self.set_pole_type(pole_type)
        return pole_type

    def _broadcast_swing_latency(self, latency):
        """Set config.swing_latency (in memory) and tell the UI. Does not persist;
        callers save_config only once a value is committed."""
        self.config.swing_latency = float(latency)
        # calibrated rides along because an omitted enum arrives at the UI as
        # CALIBRATEDSTATUS_UNSET, indistinguishable from a robot that lost its calibration.
        self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
            swing_latency=self.config.swing_latency,
            calibrated=self.config.calibrated_status))

    async def _recenter_gantry(self, center_pos):
        """Drive the gantry back to center_pos and stop."""
        await self.seek_goal(np.array(center_pos, dtype=float), head_turn=False, auto_altitude=False)
        self.slow_stop_all_spools()

    async def _recenter_gantry_if_drifted(self, center_pos, drift_limit_m):
        """Recenter only if the gantry has wandered past drift_limit_m. Running swing
        cancellation slowly pushes the gantry off-center (and, because it hangs from
        four lines, upward), so we pull it back between trials to keep them comparable
        and stay in the workspace."""
        drift = np.linalg.norm(self.pe.gant_pos - center_pos)
        if drift <= drift_limit_m:
            return
        logger.info(f'Gantry drifted {drift:.2f} m; recentering')
        await self._recenter_gantry(center_pos)

    async def _verify_swing_cancellation(self, log_context, popup_context):
        """Re-test swing cancellation against the geometry as it now stands, record the
        verdict, and switch it on only if it passed. True if it did.

        _measure_swing_residual induces a swing, runs cancellation, and reports the leftover
        swing (or the safety cap / no reading if it pumped or drifted). Anything that isn't a
        clearly-damped low residual leaves it OFF. The verdict outlives the process:
        prefer_swing_cancellation reads it to decide whether it may turn cancellation on by
        itself. log_context and popup_context say what changed, for the log and the popup.
        """
        center_pos = self.gantry_position()
        residual, aborted = await self._measure_swing_residual(self.config.swing_latency, center_pos)
        self.config.swing_cancellation_verified = (
            residual is not None and residual < swing.VERIFIED_RESIDUAL_RAD)
        save_config(self.config, self.config_path)
        if self.config.swing_cancellation_verified:
            logger.info(f'Swing cancellation damps {log_context} (residual {np.degrees(residual):.1f} deg); enabling.')
            self.set_swing_cancellation(True)
        else:
            detail = aborted or (f'{np.degrees(residual):.1f} deg residual' if residual is not None else 'no reading')
            logger.warning(f'Swing cancellation did not damp {log_context} ({detail}); leaving it OFF.')
            self.send_ui(pop_message=telemetry.Popup(
                message=f'Swing cancellation did not damp {popup_context} and was left OFF. Re-check the calibration before running.'))
        return self.config.swing_cancellation_verified

    async def _measure_swing_residual(self, latency, center_pos):
        """Run swing cancellation at `latency` and return how much the swing still
        settles to (the residual), plus an abort reason or None.

        A good latency drives the swing to nothing; a bad one leaves a steady
        residual swing. So we induce a fresh swing, run cancellation for a while,
        and report the average swing over the last few periods. Lower is better.

        Returns (residual, abort_reason); see Pendulum.trial_residual for what each abort
        scores.
        """
        RUN_PERIODS = 6.4          # how many pendulum periods to run cancellation per trial (main time cost)
        SETTLE_S = 0.5             # pause after inducing, before turning cancellation on
        DRIFT_LIMIT_M = 0.6        # stop early if the gantry wanders this far
        LOOP_S = 1 / 100

        gc = self.gripper_client

        # A fresh, modest swing so every candidate starts comparably. Cancellation
        # is off during the settle pause, so it cannot pump.
        await self._induce_swing()
        await asyncio.sleep(SETTLE_S)

        gc._swing_position_offset = np.zeros(2)
        gc._last_future_time = 0
        self._broadcast_swing_latency(latency)

        ts, amps = [], []
        self.active_set.add(SWING_VELOCITY_KEY)
        self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=True, present='.'))
        start = time.time()
        aborted = None
        try:
            while (t := time.time() - start) < RUN_PERIODS * self.pendulum.period:
                now = time.time()
                v = gc.compute_swing_correction(now + latency)
                vx, vy = (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0)
                vz = swing.altitude_hold_velocity(center_pos[2] - self.pe.gant_pos[2])
                await self.move_direction_speed(np.array([vx, vy, vz]), key=SWING_VELOCITY_KEY, downward_bias=0)
                # passive_safety raised a tension trip; bail out so the caller can back off and retry.
                if self.tension_over_limit:
                    aborted = 'tension'
                    logger.warning(f'latency {latency:.3f}s tripped the tension limit; stopping to recover')
                    break
                amp = gc.get_swing_amplitude()
                if amp is not None:
                    ts.append(t)
                    amps.append(amp)
                    if amp > swing.SAFETY_AMP_RAD:
                        aborted = 'amp_cap'
                        logger.warning(f'latency {latency:.3f}s pumped past cap; stopping (counts as bad)')
                        break
                if np.linalg.norm(self.pe.gant_pos - center_pos) > DRIFT_LIMIT_M:
                    aborted = 'drift'
                    logger.warning(f'latency {latency:.3f}s drifted too far; stopping')
                    break
                await asyncio.sleep(LOOP_S)
        finally:
            self.input_velocities[SWING_VELOCITY_KEY] = (np.zeros(3), time.monotonic())
            self.active_set.discard(SWING_VELOCITY_KEY)
            self.slow_stop_all_spools()
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=False, present='.'))

        return self.pendulum.trial_residual(ts, amps, aborted), aborted

    async def calibrate_swing_latency(self, progress_range=None, progress_name='Calibration'):
        """Tune config.swing_latency by finding the value that damps the swing best.

        A good latency drives the swing to nothing; a bad one leaves a steady
        residual swing. So we try a range of latencies, measure the leftover swing at
        each, and keep the one that leaves the least. Every candidate stays close
        enough to the ideal that it damps (rather than pumps), so nothing gets
        thrown around.

        The coarse pass spreads its candidates wide (0.3, 0, 0.6) rather than sweeping a
        narrow range: the ideal latency depends on host event-loop contention and can land
        as high as ~0.6s. A spread this wide means the outer candidates can pump hard
        rather than damp, but the safety amplitude cap stops those early, and whichever
        candidate is nearest the ideal still yields a clean, low residual to lock onto.

        0.3 is tried first because it is usually the answer, and a coarse trial under
        COARSE_GOOD_ENOUGH_RAD is close enough to refine around directly. Stopping there skips
        the two candidates most likely to pump, and the fine pass that follows supplies the
        trials MIN_TRIALS wants.

        Either pass ends outright on a trial under EXCELLENT_RESIDUAL_RAD: that is as well as
        this can damp, so the remaining trials could only tie it, and that latency is taken
        without waiting for MIN_TRIALS.
        """
        DRIFT_LIMIT_M = 0.6          # recenter between trials once drift exceeds this
        MIN_TRIALS = 3               # need at least this many good trials to choose
        TENSION_BACKOFF_S = 1.1      # wait this long after a tension trip before retrying a trial
        MAX_TENSION_RETRIES = 3      # give up (and abort) if a single trial keeps tripping tension

        def report_progress(pct, action):
            """Progress for the enclosing operation, or nothing if the caller wants none.

            The end of progress_range doubles as this routine's completion: for a standalone
            run that is 100, which is what clears the UI's progress bar, and for a run nested
            in full calibration it is just the next step's starting percentage.
            """
            if progress_range is None:
                return
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=pct, name=progress_name, current_action=action))

        def finish(action):
            report_progress(progress_range[1] if progress_range else 0.0, action)

        if self.gripper_client is None:
            logger.warning('Swing latency calibration requires a connected gripper')
            finish('Swing latency tuning needs a connected gripper')
            return None

        # Before anything is tuned: the pole decides the frequency the trials below are
        # phased against, and the operator picking it from a list is the weakest link in
        # that chain. Ahead of center_pos so it is taken where the trials will actually run.
        report_progress(progress_range[0] if progress_range else 0.0,
                        'Identifying the gripper pole')
        await self._identify_pole_type()

        original_latency = self.config.swing_latency
        center_pos = np.array(self.pe.gant_pos, dtype=float)
        all_results = []      # (latency, residual) from every reliable trial
        excellent = None      # latency of a trial good enough to end the calibration outright

        async def sweep(cands, stop_below=None):
            nonlocal excellent
            out = []
            for idx,lat in enumerate(cands):
                if progress_range is not None:
                    start_pct, end_pct = progress_range
                    pct = start_pct + (end_pct - start_pct) * (idx + 1) / (len(cands) + 1)
                    report_progress(pct, f"Tuning swing cancellation {idx + 1}/{len(cands)} ({lat})")

                lat = float(lat)
                # A tension trip during a trial is recoverable: wait for the back-off, move
                # back to the swing cal starting position, and retry this same latency. Only a
                # trial that keeps tripping gives up and aborts the whole calibration.
                attempts = 0
                while True:
                    await self._recenter_gantry_if_drifted(center_pos, DRIFT_LIMIT_M)
                    residual, aborted = await self._measure_swing_residual(lat, center_pos)
                    if aborted != 'tension':
                        break
                    attempts += 1
                    if attempts > MAX_TENSION_RETRIES:
                        logger.warning(f'Tension kept exceeding the limit at latency {lat:.3f}s; aborting calibration')
                        # leave tension_over_limit set so the abort reports the real reason
                        self.motion_task.cancel()
                        await asyncio.sleep(0)  # let the cancellation take effect
                        return out
                    logger.warning(f'Tension over limit during latency {lat:.3f}s trial (attempt {attempts}); waiting {TENSION_BACKOFF_S}s and returning to start')
                    await asyncio.sleep(TENSION_BACKOFF_S)
                    self.tension_over_limit = False  # cleared after the back-off so the retry starts fresh
                    await self._recenter_gantry(center_pos)
                tag = f' [{aborted}]' if aborted else ''
                if residual is not None:
                    out.append((lat, residual))
                    all_results.append((lat, residual))
                    logger.info(f'swing_latency {lat:.3f}s -> residual {residual*1000:.0f} mrad ({np.degrees(residual):.1f} deg){tag}')
                    if residual < swing.EXCELLENT_RESIDUAL_RAD:
                        logger.info(f'swing_latency {lat:.3f}s damps to under '
                                    f'{swing.EXCELLENT_RESIDUAL_RAD*1000:.0f} mrad; taking it and '
                                    f'skipping the remaining trials')
                        excellent = lat
                        return out
                    if stop_below is not None and residual < stop_below:
                        logger.info(f'swing_latency {lat:.3f}s already damps below {stop_below*1000:.0f} mrad; '
                                    f'skipping the remaining coarse candidates and refining around it')
                        return out
                else:
                    logger.info(f'swing_latency {lat:.3f}s -> unreliable, excluded{tag}')
                await asyncio.sleep(0.3)
            return out

        self.tension_over_limit = False  # clear any stale trip so the first trial isn't cut short
        try:
            # let passive_safety recover (not abort) on a tension trip here
            with self.override_safety(on_over_tension=OverTension.NOTIFY):
                coarse = await sweep(swing.COARSE_CANDS, stop_below=swing.COARSE_GOOD_ENOUGH_RAD)
                if coarse and excellent is None:
                    best_coarse = min(coarse, key=lambda r: r[1])[0]
                    # Recenter before the fine pass so the trials we care about start with
                    # full drift headroom and don't get cut short.
                    await self._recenter_gantry(center_pos)
                    # skip latencies the coarse pass already measured; the fine spread is
                    # centered on the winner, so it always lands on it again
                    await sweep(swing.fine_candidates(best_coarse, [lat for lat, _ in all_results]))
        except asyncio.CancelledError:
            # Reported here rather than left to the caller: a standalone run has no outer
            # handler to clear the progress bar for it.
            finish('Aborted: line tension exceeded the safe limit' if self.tension_over_limit
                   else 'Swing latency tuning cancelled')
            raise
        finally:
            # Do not clear tension_over_limit here: on a max-retry abort it must survive to the
            # calibration's CancelledError handler so it can report the tension reason.
            self.input_velocities[SWING_VELOCITY_KEY] = (np.zeros(3), time.monotonic())
            self.active_set.discard(SWING_VELOCITY_KEY)
            self.slow_stop_all_spools()
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=False, present='.'))
            await self._recenter_gantry(center_pos)

        if excellent is not None:
            # A single trial this good outranks a range picked from mediocre ones, so it
            # stands on its own without the MIN_TRIALS worth of context they need.
            best = excellent
            self._broadcast_swing_latency(best)
            self.config.swing_cancellation_verified = True
            save_config(self.config, self.config_path)
            logger.info(f'Calibrated swing_latency = {best:.3f}s (residual '
                        f'{min(r for _, r in all_results)*1000:.0f} mrad, damps)')
            finish(f'Swing latency set to {best:.3f}s')
            return best

        if len(all_results) < MIN_TRIALS:
            # too little to choose a latency from, and equally too little to say anything
            # about whether cancellation works here, so the stored verdict is left alone
            logger.warning(f'Swing latency calibration got only {len(all_results)} usable trials; keeping existing value')
            self._broadcast_swing_latency(original_latency)
            finish(f'Only {len(all_results)} usable trials; keeping swing latency {original_latency:.3f}s')
            return None

        best = swing.select_min_residual(all_results)
        self._broadcast_swing_latency(best)
        # The sweep just watched cancellation damp (or fail to) at a spread of latencies, which
        # is the same measurement the end of full calibration makes. Judge it the same way and
        # store it, so a robot tuned by the swinglatencycal debug command alone is as much
        # entitled to prefer_swing_cancellation as one that ran the whole calibration.
        best_residual = min(r for _, r in all_results)
        self.config.swing_cancellation_verified = best_residual < swing.VERIFIED_RESIDUAL_RAD
        save_config(self.config, self.config_path)
        logger.info(f'Calibrated swing_latency = {best:.3f}s (best residual '
                    f'{np.degrees(best_residual):.1f} deg, '
                    f'{"damps" if self.config.swing_cancellation_verified else "does NOT damp"})')
        finish(f'Swing latency set to {best:.3f}s '
               f'({"damps" if self.config.swing_cancellation_verified else "does NOT damp"})')
        return best

    async def _handle_debug_command(self, item: control.Debug):
        logger.debug(f'Debug action "{item.action}"')
        words = item.action.split()
        if words and words[0] in self._maneuver_verbs:
            return await self._run_maneuver_handler(*self._maneuver_verbs[words[0]], *words[1:])
        if item.action == "spincal":
            # as a motion task, so the stop button reaches it
            r = await self.invoke_motion_task(self.calibrate_spin())
        if item.action == 'spin-recover':
            # re-measures the spin with no origin card, by flying a slow circle
            r = await self.invoke_motion_task(self.recover_spin())
        if item.action == 'fingercal':
            asyncio.create_task(self.calibrate_finger_servo())
        if item.action == 'eyelets':
            # use the currently calibrated anchor poses from the config
            anchor_poses = [poseProtoToTuple(a.pose) for a in self.config.anchors]
            # top of work area, from the anchor-side pull points only, as full_auto_calibration does
            upper_z = float(np.mean(self.pe.anchor_points[[0, 2], 2]))
            r = await self.invoke_motion_task(self.collect_arp_anchor_eyelet_experiment_data(anchor_poses, upper_z))
        if item.action == 'gripcards':
            # run the gripper card survey standalone and pickle the result for offline
            # experimentation with the optimizer. cards must still be in place.
            async def survey_and_save():
                gripper_obs = await self.collect_gripper_card_observations()
                with open('gripper_card_obs.pkl', 'wb') as f:
                    pickle.dump(gripper_obs, f)
                logger.info(f'Saved gripper card survey to gripper_card_obs.pkl: {list(gripper_obs.keys())}')
            r = await self.invoke_motion_task(survey_and_save())
        if item.action == 'stow':
            r = await self.stow_lines()
        if item.action == 'upright':
            r = await self.invoke_motion_task(self.ensure_pole_upright())
        if item.action.startswith('swinglatency '):
            parts = item.action.split(' ')
            self.config.swing_latency = float(parts[1])
            save_config(self.config, self.config_path)
        if item.action == 'swinglatencycal':
            # Its own operation name, not "Calibration": this run owns the whole progress bar
            # and its completion must not read as the full calibration having finished.
            r = await self.invoke_motion_task(self.calibrate_swing_latency(
                progress_range=(0.0, 100.0), progress_name='Swing Latency'))
        if item.action.startswith('polecal'):
            # 'polecal [seconds]' - how long to record the free decay for
            parts = item.action.split()
            decay_s = float(parts[1]) if len(parts) == 2 else None
            r = await self.invoke_motion_task(self.measure_pendulum_length(decay_s))
        if item.action == 'reset_wrist':
             r = await self.gripper_client.send_commands({'reset_wrist': None})
        if item.action == 'spind':
            print(self.gripper_client.get_spin(True))
        if item.action == 'sync_timezone':
            await self.sync_timezone_to_bots()
        if item.action == 'pull_logs':
            asyncio.create_task(self.pull_logs_to_zip())
        if item.action.startswith('untwist'):
            parts = item.action.split()
            if len(parts)==2 and parts[0]=='untwist':
                r = await self.gripper_client.send_commands({'untwist': int(parts[1])})
        if item.action.startswith(('setvar ', 'savevar ')):
            # 'setvar KEY VALUE' broadcasts a live config override to every component.
            # used for bench tuning of onboard loop constants without restarting firmware.
            # 'savevar KEY VALUE' does the same and keeps it in the robot config, so every
            # component is sent it again each time it connects.
            parts = item.action.split()
            if len(parts) == 3:
                verb, key, text = parts
                value = parse_config_var(text)
                if verb == 'savevar':
                    self.config.component_vars[key] = text
                    save_config(self.config, self.config_path)
                logger.info(f'Broadcasting set_config_vars {key}={value} to all components')
                await asyncio.gather(*[
                    client.send_commands({'set_config_vars': {key: value}})
                    for client in self.bot_clients.values()
                ])
            else:
                logger.warning(f'invalid {parts[0]} command, expected "{parts[0]} KEY VALUE": {item.action}')
        if item.action.startswith('spoollog'):
            # 'spoollog [off]' starts or stops the spool diagnostic log.
            if item.action.split()[-1] == 'off':
                self.stop_spool_log()
            else:
                self.start_spool_log()
        if item.action.startswith('holdtension '):
            # 'holdtension LINE VALUE|off' engages onboard two-sided tension hold on one
            # arpeggio line, or clears it with 'off'. for bench testing hold mode.
            parts = item.action.split()
            if len(parts) == 3:
                line_no = int(parts[1])
                value = None if parts[2] == 'off' else float(parts[2])
                await self.send_line_speed(line_no, 0)
                await self.set_line_tension_target(line_no, value)
                logger.info(f'set tension target on line {line_no} to {value}')
            else:
                logger.warning(f'invalid holdtension command, expected "holdtension LINE VALUE|off": {item.action}')
        if item.action.startswith('tensionreg'):
            parts = item.action.split()
            if len(parts) == 2:
                offon = parts[1]
                if offon == 'on':
                    r = await self.set_tension_reg(True)
                else:
                    r = await self.set_tension_reg(False)
        if item.action == 'findorigin':
            # The calibration step on its own, so the search can be tuned in the room that breaks
            # it - low ceiling, origin card up on a bed - without running a whole calibration.
            r = await self.invoke_motion_task(self.find_origin_card())
        if item.action == 'centerorigin':
            r = await self.invoke_motion_task(self._center_card_in_view('origin'))
        if item.action == 'servograsp':
            # One visual servoing grasp from wherever the gripper is now, without the
            # pick and place loop. Park it over the object first, roughly - closing the
            # rest is the model's job.
            async def servo_grasp_once():
                if not await self.servo.ensure_model():
                    return
                async with self.prefer_swing_cancellation():
                    success = await self.servo.run(mode=SERVO_MODE_GRASP)
                logger.info(f'servograsp succeeded={success}')
            r = await self.invoke_motion_task(servo_grasp_once())
        if item.action == 'servowatch':
            # The same model on the same frames, reported to the gripper overlay and
            # nothing else, until another motion task or a stop cancels it. Park the
            # gripper over an object and watch where the arrow points before trusting it
            # with the gantry.
            async def servo_watch():
                if not await self.servo.ensure_model():
                    return
                await self.servo.run(mode=SERVO_MODE_OBSERVE)
            r = await self.invoke_motion_task(servo_watch())
        if item.action == 'servocenter':
            # Steers, but only sideways: the lateral servo and the wrist, running until
            # cancelled, with no descent and nothing done to the fingers. Park the gripper
            # somewhere safe above an object and watch whether it settles over it.
            async def servo_center():
                if not await self.servo.ensure_model():
                    return
                await self.servo.run(mode=SERVO_MODE_CENTER)
            r = await self.invoke_motion_task(servo_center())
        if item.action == 'servoloop':
            # Grasp, drop, repeat, keeping score until cancelled. One checkpoint against
            # the next is a question about hit rate on real objects, and a hit rate needs
            # more attempts than anyone will sit through by hand.
            async def servo_score_loop():
                if not await self.servo.ensure_model():
                    return
                async with self.prefer_swing_cancellation():
                    await self.servo.score()
            r = await self.invoke_motion_task(servo_score_loop())

    async def set_tension_reg(self, enabled: bool):
        """Enable or disable onboard tension regulation (the floor + soft mute) on both
        spools of every anchor."""
        logger.info(f'setting tension reg {"on" if enabled else "off"} for all anchors')
        await asyncio.gather(*[
            anchor.send_commands({'set_tension_reg': (enabled, spool_no)})
            for anchor in self.anchors.values()
            for spool_no in (0, 1)
        ])
        if enabled != self.tension_reg_enabled:
            self.tension_reg_enabled = enabled
            self.send_ui(tension_regulation_state=telemetry.TensionRegulationState(enabled=enabled, present=True))

    async def sync_timezone_to_bots(self):
        tz = self._get_local_timezone_name()
        if not tz:
            logger.warning("Could not determine local timezone; skipping timezone sync to bots")
            return
        await asyncio.gather(*[
            client.send_commands({'set_timezone': tz})
            for client in self.bot_clients.values()
        ])

    async def shutdown_all_bots(self):
        """Ask every connected component to power its Pi off cleanly."""
        clients = list(self.bot_clients.items())
        if not clients:
            logger.warning('no components connected; nothing to shut down')
            self.send_ui(pop_message=telemetry.Popup(message='No components are connected.'))
            return
        # nothing should be commanding motion into a component that is about to halt.
        await self.stop_all()
        logger.info(f'requesting poweroff of {len(clients)} components: '
                    f'{", ".join(name for name, _ in clients)}')
        results = await asyncio.gather(*[
            client.send_commands({'shutdown_pi': True})
            for _, client in clients
        ] + [asyncio.sleep(10)], return_exceptions=True) # ten seconds to ensure green led is off. user cant see them.
        for (name, _), result in zip(clients, results):
            if isinstance(result, Exception):
                logger.warning(f'{name} may not have received the poweroff request: {result!r}')
        logger.info('poweroff requested; wait 10 seconds before cutting power')
        self.send_ui(pop_message=telemetry.Popup(message='Shutdown complete.'))

    async def shutdown_one_bot(self, client, name):
        """Ask one component to power its Pi off cleanly.

        Motion stops across the whole robot first, not just at this component: the lines
        are coupled, so a spool that goes dead while the others are still pulling is the
        case that leaves the gantry hanging off the remainder.

        The component latches its own poweroff, so asking twice is harmless.
        """
        # nothing should be commanding motion into a component that is about to halt.
        await self.stop_all()
        logger.info(f'requesting poweroff of {name}')
        try:
            # ten seconds to ensure green led is off. user cant see them.
            await asyncio.gather(client.send_commands({'shutdown_pi': True}), asyncio.sleep(10))
        except Exception as e:
            logger.warning(f'{name} may not have received the poweroff request: {e!r}')
            self.send_ui(pop_message=telemetry.Popup(
                message=f'{name.capitalize()} may not have received the shutdown request. '
                        f'Check that it is off before cutting power.'))
            return
        logger.info(f'poweroff of {name} requested; safe to cut its power')
        self.send_ui(pop_message=telemetry.Popup(
            message=f'{name.capitalize()} has shut down. Its power can now be cut safely. '
                    f'It will not come back until it is power cycled.'))

    async def pull_logs_to_zip(self):
        """Pull recent log lines from every connected component and bundle them into a
        local zip file, entries named after each component's IP address. Includes the
        thermal watchdog log when the component sends one."""
        clients = list(self.bot_clients.values())
        logs = await asyncio.gather(*[client.pull_logs() for client in clients])
        zip_path = f'pulled_logs_{int(time.time())}.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for client, (log_text, thermal_text) in zip(clients, logs):
                if log_text is None:
                    logger.warning(f'No logs received from {client.address}; skipping')
                    continue
                zf.writestr(f'{client.address}.log', log_text)
                if thermal_text:
                    zf.writestr(f'{client.address}_thermal.log', thermal_text)
        logger.info(f'Saved logs from {len(clients)} component(s) to {zip_path}')

    @staticmethod
    def _get_local_timezone_name():
        """Return the host's IANA timezone name (e.g. 'America/New_York').

        The bots run Linux and expect an IANA name. On Linux `timedatectl` is
        authoritative, but it doesn't exist on Windows (and Windows names zones
        differently), so fall back to tzlocal, which maps to IANA on every platform.
        """
        if sys.platform != 'win32':
            try:
                tz = subprocess.check_output(
                    ['timedatectl', 'show', '--property=Timezone', '--value']
                ).decode().strip()
                if tz:
                    return tz
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass
        try:
            from tzlocal import get_localzone_name
            return get_localzone_name()
        except Exception:
            logger.exception("Failed to determine local timezone name")
            return None

    async def calibrate_finger_servo(self):
        self.gripper_client.finger_contact_calibration_complete.clear()
        await asyncio.create_task(self.gripper_client.send_commands({'measure_finger_contact': None}))
        await asyncio.wait_for(self.gripper_client.finger_contact_calibration_complete.wait(), 20)

    async def _handle_common_command(self, cmd: control.Command):
        # betterproto Enums are IntEnums, comparable directly
        match cmd:
            case control.Command.STOP_ALL:
                r = await self.stop_all()
            case control.Command.TIGHTEN_LINES:
                r = await self.tension_lines()
            case control.Command.HALF_CAL:
                r = await self.invoke_motion_task(self.half_auto_calibration())
            case control.Command.FULL_CAL:
                # it turns sightings of the gantry marker into stored geometry, so a marker
                # fault would be fitted rather than merely degrade it
                r = await self.invoke_motion_task(self.full_auto_calibration(),
                                                  safety=SafetyPolicy(needs_gantry_marker=True))
            case control.Command.SHUTDOWN:
                self.run_command_loop = False
            case control.Command.RECORD_DROP:
                self.record_drop_position()
            case control.Command.GRASP:
                r = await self.invoke_motion_task(self.grasp())
            case control.Command.UPDATE_FIRMWARE:
                r = await self._handle_update_firmware()
            case control.Command.DISABLE_TORQUE:
                await self.set_torque(False)
            case control.Command.ENABLE_TORQUE:
                await self.set_torque(True)
            case control.Command.DEBUG_LOG_OVER_T:
                self._enable_debug_log_over_telemetry()
            case control.Command.ENABLE_TENSION_REG:
                r = await self.set_tension_reg(True)
            case control.Command.DISABLE_TENSION_REG:
                r = await self.set_tension_reg(False)
            case control.Command.SAFE_COMPONENT_SHUTDOWN:
                r = await self.shutdown_all_bots()
            case _:
                entry = self._maneuver_commands.get(cmd)
                if entry is None:
                    logger.warning(f'No handler for command {cmd!r}')
                else:
                    r = await self._run_maneuver_handler(*entry)

    def _enable_debug_log_over_telemetry(self):
        if self._telem_log_handler is not None:
            return
        nf_logger = logging.getLogger('nf_robot')
        nf_logger.setLevel(logging.DEBUG)
        handler = TelemetryLogHandler(self)
        handler.setFormatter(logging.Formatter('%(levelname)s %(name)s %(message)s'))
        nf_logger.addHandler(handler)
        self._telem_log_handler = handler
        logger.info('Debug logging over telemetry enabled')

    async def _handle_update_firmware(self):
        r = await self.stop_all()
        async def update_bar_task():
            for i in range(100):
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=float(i),
                    name="Update Component Firmware",
                    current_action="updating...",
                ))
                if not self.run_command_loop:
                    break
                await asyncio.sleep(0.5)
        bar = asyncio.create_task(update_bar_task())
        await self.sync_timezone_to_bots()
        await asyncio.sleep(0.3)
        tasks = []
        # capture each client's address now, while it still exists in bot_clients. a
        # successful update restarts the component, which removes it from the dict before
        # we build the results table below, so we can't look it up again afterward.
        addresses = []
        for name, client in self.bot_clients.items():
            tasks.append(client.firmware_update())
            addresses.append(client.address)
        results = await asyncio.gather(*tasks)
        bar.cancel()
        lines = []
        for i, r in enumerate(results):
            a = "Not supported"
            if r == True:
                a = "Success"
            elif r == False:
                a = "Failed"
            lines.append(f"({addresses[i]}) {a}")
        table = '\n'.join(lines)
        if any(x is False for x in results):
            message = f"Failed on one or more components \n\n{table}"
        elif all(results):
            message = "Updated successfully. Components are now rebooting. Please wait 10 to 20 seconds."
        else:
            message = f"Successful on some components, others require manual updating \n\n{table}"
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=float(100),
            name="Update Component Firmware",
            current_action=message,
        ))

    async def set_torque(self, enabled: bool):
        """Enable or disable position-holding torque on every anchor's motors.

        Counterpart to set_tension_reg. The resulting state is not recorded here: the
        anchors echo the commanded torque state back and publish_torque_state reports it.
        """
        if self.config.anchor_type != common.AnchorType.ARPEGGIO:
            return
        logger.info(f'setting torque {"on" if enabled else "off"} for all anchors')
        command = 'enable_torque' if enabled else 'disable_torque'
        await asyncio.gather(*[
            client.send_commands({command: None})
            for client in self.anchors.values()
        ])

    def publish_torque_state(self):
        """Push the anchors' aggregate torque state to the UI when it changes.

        Called by every anchor client that reports a torque state. Torque counts as on
        only when every connected anchor says it is on. Automatic torque cycling done
        for safety is suppressed, so the UI only ever shows operator-driven state.
        """
        if self._torque_reports_suppressed:
            return
        states = [client.conn_status.motor_enabled for client in self.anchors.values()]
        enabled = bool(states) and all(s == telemetry.MotorTorque.ENABLED for s in states)
        if enabled != self.torque_enabled:
            self.torque_enabled = enabled
            self.send_ui(torque_state=telemetry.TorqueState(enabled=enabled, present=True))

    async def _handle_jog_spool(self, jog: control.JogSpool):
        """Handles manually jogging a spool motor."""
        # identify the client we need to send the command to
        client = None
        if jog.is_gripper:
            if jog.speed is not None:
                r = await self.gripper_client.send_commands({'aim_speed': jog.speed})
            elif jog.offset is not None:
                r = await self.gripper_client.send_commands({'jog': jog.offset})
        else:
            if jog.speed is not None:
                await self.send_line_speed(jog.anchor_num, jog.speed)
            elif jog.offset is not None:
                await self.send_line_speed(jog.anchor_num, jog.offset, jog=True)

    async def _handle_gantry_goal_pos(self, goal_pos: np.ndarray):
        """Handles moving the marker box to a specific goal position."""
        await self.invoke_motion_task(self.seek_goal(goal_pos))

    async def _handle_slow_stop_one(self, stop_data: dict):
        """Handles stopping a single spool motor."""
        if stop_data.get('id') == 'gripper' and self.gripper_client:
            r = await self.gripper_client.slow_stop_spool()
        else:
            for client in self.anchors.values():
                if client.anchor_num == stop_data.get('id'):
                    r = await client.slow_stop_spool()

    async def _handle_movement(self, move: control.CombinedMove):
        if move.source_key and move.source_key.startswith(RESERVED_KEY_PREFIXES):
            logger.warning(f'Ignoring a move under the reserved source key {move.source_key!r}')
            return
        winch = None
        wrist = None
        if self.gripper_client is not None:
            # if we have to clip these values to legal limits, save what they were clipped to
            if move.finger_speed is not None or move.wrist_speed is not None:
                winch, finger, wrist = await self.send_gripper_move(move.winch, move.finger_speed, move.wrist_speed)
            else:
                # this type of message may be sent from older UIs. probably safe to removed by end of Feb.
                winch, finger, wrist = await self.send_gripper_move_legacy(move.winch, move.finger, move.wrist)

        direction = np.zeros(3)
        if move.direction:
            direction = tonp(move.direction)

            if self.gripper_client is not None:
                if move.direction_is_in_gripper_frame:
                    if move.speed is not None:
                        velocity = direction * move.speed # make sure the network receives information on speed as well
                    else:
                        velocity = direction
                    self.send_ui(raw_commanded_vel=telemetry.CommandedVelocity(velocity=fromnp(velocity)))
                    # rotate later component of direction into room frame
                    direction[:2] = rotate_vector(direction[:2], -self.gripper_client.get_spin())
                else:
                    # direction is already in room frame, and we can use it, but we still want to send the lerobot record script a direction in gripper frame
                    gf_direction = direction.copy()
                    gf_direction[:2] = rotate_vector(gf_direction[:2], self.gripper_client.get_spin())
                    if move.speed is not None:
                        velocity = gf_direction * move.speed # make sure the network receives information on speed as well
                    else:
                        velocity = gf_direction
                    self.send_ui(raw_commanded_vel=telemetry.CommandedVelocity(velocity=fromnp(velocity)))

        # Allow source keys to be used to distinguish the input
        commanded_vel = await self.move_direction_speed(direction, move.speed, key=move.source_key)

        self.last_user_move_time = time.time()

    @property
    def max_safe_tension(self):
        """Line tension passive_safety sheds torque (and aborts a motion task) over."""
        if self.config.max_safe_tension is not None:
            return self.config.max_safe_tension
        return DEFAULT_MAX_SAFE_TENSION

    async def measure_free_tension(self, samples=5, interval_s=0.1):
        """Per-line tension while the gantry hangs free, as the (4,) median of a short burst.

        A reading under TENSION_THRESH is not a light load being measured, it is a line not
        reporting one, so each line is floored there: taken at face value it would leave any
        threshold derived from this trivially trippable. Sample it somewhere the gantry is
        hanging from all four lines and nothing else.
        """
        burst = []
        for _ in range(samples):
            burst.append(np.asarray(self.pe.tension, dtype=float))
            await asyncio.sleep(interval_s)
        return np.maximum(np.median(burst, axis=0), TENSION_THRESH)

    def _tension_near_limit(self, frac):
        """True once the tightest line is within frac of the tension passive_safety trips at.
        Reading it lets a move back off before the trip; after it, the motion task is gone."""
        tension = self.pe.tension
        return tension is not None and float(np.max(tension)) > frac * self.max_safe_tension

    def _respond_to_over_tension(self, tensions):
        """Deal with the running motion task as its SafetyPolicy says, once a line is over."""
        if self.motion_task is None or self.motion_task.done():
            return
        self.tension_over_limit = True
        name = self.motion_task.get_name()
        owner = self._motion_owner
        match self._motion_safety.on_over_tension:
            case OverTension.ABORT:
                logger.warning(f'Tension overload during motion task "{name}" - aborting it')
                if owner is not None:
                    owner.abort_reason = 'tension'
                self.motion_task.cancel()
            case OverTension.NOTIFY:
                logger.warning(f'Tension overload during motion task "{name}" - letting it back off')
                if owner is not None:
                    try:
                        owner.on_over_tension(tensions)
                    except Exception:
                        logger.exception(f'{owner.name}.on_over_tension failed')
            case OverTension.IGNORE:
                logger.warning(f'Tension overload during motion task "{name}" - it runs on')

    async def passive_safety(self):
        """If any line becomes too tight, switch all motors to damped movement for one second.
        That happens whatever is running. What becomes of a motion task that was running at
        the time is its SafetyPolicy's call: by default it is aborted, since backing off
        mid-motion corrupts whatever it was doing."""
        max_safe_tension = self.max_safe_tension

        ema = np.zeros(4)
        while self.run_command_loop and self.pe.tension is not None:
            ema = ema * 0.9 + self.pe.tension * 0.1
            if np.any(ema > max_safe_tension):
                logger.warning(f'Tension limit reached! backing off. limit={max_safe_tension} actual={ema}')
                self._respond_to_over_tension(ema.copy())
                # Shedding tension by cycling torque is a safety action, not an
                # operator one, so it must not move the UI's torque toggle.
                self._torque_reports_suppressed = True
                try:
                    await self.set_torque(False)
                    await asyncio.sleep(1)
                    await self.set_torque(True)
                    await asyncio.sleep(1)
                finally:
                    self._torque_reports_suppressed = False
            await asyncio.sleep(0.2)

    def _report_gantry_marker_fault(self, key, phrase, message, detail, once_per_session=False):
        """Show the operator a gantry marker fault once, and abort a running calibration.

        Calibration is what cannot survive one of these: it reads a batch of sightings as
        repeated looks at one point and writes the result out as room geometry, so a marker
        seen in two places, or last seen minutes ago, is fitted rather than rejected and the
        run finishes 'successfully' on a room that does not exist. Everything else that uses
        the marker is a live estimate that recovers on the next good frame."""
        if key in self._gantry_marker_warned:
            return
        self._gantry_marker_warned.add(key)
        logger.warning(f'Gantry marker fault: {detail}')
        if not (once_per_session and key in self._gantry_marker_popped):
            self.send_ui(pop_message=telemetry.Popup(message=message))
        self._gantry_marker_popped.add(key)
        if (self.motion_task is not None and not self.motion_task.done()
                and self._motion_safety.needs_gantry_marker):
            logger.warning(f'Gantry marker fault during motion task '
                           f'"{self.motion_task.get_name()}" - aborting it')
            self.gantry_marker_fault = phrase
            if self._motion_owner is not None:
                self._motion_owner.abort_reason = 'marker'
            self.motion_task.cancel()

    async def monitor_gantry_visibility(self):
        """Watch how the anchor cameras see the gantry marker for the two faults that produce
        confident, wrong observations rather than an obvious absence of them:

        1. one camera seeing the marker in several places at once, which means there is a
           second robot tag in the room or a mirror showing it this one, and
        2. no camera having seen it for a while. Losing it from one camera is normal,
           losing it from both means that its in a blind spot or it's mounted wrong.

        Cheap enough to leave running: once a second it reads the position buffer the detection
        callback already fills, and the only measurement it takes is between sightings that
        share a capture time, of which there is normally at most one per frame.
        """
        # Every detection from one frame is filed under that frame's capture time, so sightings
        # sharing a timestamp are one camera's account of one instant. Two of them this far
        # apart is two tags: the gantry cannot be in both places, and unlike a spread measured
        # across time this says nothing about how fast the real one was moving.
        SPLIT_LIMIT_M = 0.75
        # Detection runs on crops around the last known position of each tag, with a full frame
        # scan once a second, so a duplicate is not seen every frame. Keep enough history for
        # several of those scans, and require more than one to have split before calling it.
        HISTORY_S = 10.0
        MIN_SPLIT_FRAMES = 2

        await self.any_anchor_connected.wait()
        history = {}                # anchor num -> its recent rows, older than the buffer holds
        newest_per_anchor = {}      # anchor num -> newest capture time already taken from it
        # host clock, so a component whose clock is skewed cannot fake staleness
        last_advance = time.time()

        # Warmup time. Seconds until this task becomes active
        await asyncio.sleep(30)

        while self.run_command_loop:
            await asyncio.sleep(VISIBILITY_POLL_S)
            advanced = False        # did any camera deliver a sighting it had not before
            # The live array, not deepCopy: nothing here depends on row order, and an insert
            # racing this read can only replace one row of the several being weighed.
            rows = self.datastore.gantry_pos.asNpa()
            rows = rows[rows[:, 0] > 0]  # rows never written hold zeros

            # ---- 1. the marker in more than one place at once, per camera
            for anchor_num in {int(n) for n in rows[:, 1]}:
                mine = rows[rows[:, 1] == anchor_num]
                newest = float(mine[:, 0].max())
                # Nothing new from this camera. Its old sightings are still in the buffer and
                # would otherwise be re-judged, and re-reported, forever.
                if newest <= newest_per_anchor.get(anchor_num, 0.0):
                    continue
                advanced = True
                # The datastore's buffer is only a second or two deep and is shared between the
                # anchors, which is too little to catch a duplicate that shows up on the once-a-
                # second full scans, so keep our own window of what it has held.
                fresh = mine[mine[:, 0] > newest_per_anchor.get(anchor_num, 0.0)]
                kept = history.get(anchor_num)
                window = fresh if kept is None else np.concatenate([kept, fresh])
                window = window[window[:, 0] > newest - HISTORY_S]
                history[anchor_num] = window
                newest_per_anchor[anchor_num] = newest

                times, counts = np.unique(window[:, 0], return_counts=True)
                gaps = [_widest_gap(window[window[:, 0] == t][:, 2:]) for t in times[counts > 1]]
                split = [g for g in gaps if g > SPLIT_LIMIT_M]
                if len(split) >= MIN_SPLIT_FRAMES:
                    self._report_gantry_marker_fault(
                        ('duplicate', anchor_num),
                        f'the gripper marker was seen in more than one place by anchor {anchor_num}',
                        f"The gripper's marker tag appears to anchor {anchor_num} in multiple "
                        f"places. Please check the room for other robot tags, or mirrors which "
                        f"are visible to this anchor, and cover them.",
                        f'anchor {anchor_num} saw the gantry marker in two places at once in '
                        f'{len(split)} of the last {len(times)} frames, up to {max(split):.2f} m '
                        f'apart',
                    )
                else:
                    self._gantry_marker_warned.discard(('duplicate', anchor_num))

            # ---- 2. no camera seeing the marker at all. Judged on whether any one camera's
            # own newest sighting moved on, so two components whose clocks disagree cannot
            # leave the slower one's fresh sightings looking older than the faster one's.
            if advanced:
                last_advance = time.time()
                self._gantry_marker_warned.discard('unseen')
            elif not self.anchors or self._marker_may_be_hidden:
                # nothing is looking, which is a connection problem and reported as one; or
                # calibration has yet to stand the pole upright, which is what brings the
                # marker into view, and the clock starts from there
                last_advance = time.time()
            elif time.time() - last_advance > UNSEEN_LIMIT_S:
                self._report_gantry_marker_fault(
                    'unseen',
                    f'the gripper marker has not been seen for {UNSEEN_LIMIT_S:.0f} seconds',
                    f"The gripper's marker tag hasn't been detected in {UNSEEN_LIMIT_S:.0f} "
                    f"seconds. Please confirm the carabiners are attached such that the markers "
                    f"face the anchor cameras and that nothing is obscuring it",
                    f'no anchor camera has seen the gantry marker in '
                    f'{time.time() - last_advance:.0f}s',
                    # a marker that keeps dropping out would otherwise pop this every time it
                    # came back and went away again
                    once_per_session=True,
                )

    def update_avg_named_pos(self, key: str, position: np.ndarray):
        """Update the running average of the named position, keeping self.config.named_positions
        as the single source of truth so the last known position survives a restart."""
        if key in self.config.named_positions:
            # exponential moving average
            position = tonp(self.config.named_positions[key]) * 0.75 + position * 0.25
        self.config.named_positions[key] = fromnp(position)
        self.send_ui(named_position=telemetry.NamedObjectPosition(
            position=fromnp(position),
            name=key,
        ))

    async def invoke_motion_task(self, coro, owner=None, safety=None):
        """
        Cancel whatever else is happening and start a new long running motion task
        Any task that can be called this way is known in this file as a "motion task"
        The defining feature of a motion task is that it could send a second motion command to any client after any amount of sleeping
        every motion task must have the follwing structure

        try:
            # do something
        except asyncio.CancelledError:
            raise
        finally:
            # perform any clean up work

        Do not call invoke_motion_task from within a motion task or it will cancel itself.
        It is ok to call a motion task from within another, just don't start it with invoke_motion_task
        Do not call stop_all from within a motion task. use slow_stop_all_spools instead

        owner is the maneuver the task belongs to, if any, and safety the SafetyPolicy the
        safety monitors hold it to (the default one if not given).
        """
        if self.motion_task is not None and not self.motion_task.done():
            logger.debug(f'current motion task {self.motion_task} done={self.motion_task.done()}')
            logger.info(f"Cancelling previous motion task: {self.motion_task.get_name()}")
            self.motion_task.cancel()
            try:
                # Wait briefly for the old task's cleanup to complete.
                result = await self.motion_task
            except asyncio.CancelledError:
                pass # Expected behavior
        # a seek outlives a caller that stopped waiting on it, so it is ended here as well
        await self._end_seek()

        self._motion_owner = owner
        self._motion_safety = safety or SafetyPolicy()
        if owner is not None:
            owner.abort_reason = None
        self.motion_task = asyncio.create_task(coro)
        self.motion_task.set_name(coro.__name__)

    async def tension_lines(self):
        """Request all anchors to reel in all lines until tight."""
        sends = []
        for client in self.anchors.values():
            sends.append(client.send_commands({'tighten': 0}))
            sends.append(client.send_commands({'tighten': 1}))
        # Awaiting only delivers the command; it does not wait for confirmation that every
        # anchor has finished tightening, as that would just hold up the processing of the ob_q.
        # this is similar to sending a manual move command. it can be overridden by any subsequent command.
        # thus, it should be done while paused.
        await asyncio.gather(*sends)

    async def stow_lines(self):
        """Request all anchors to reel in all lines until tight and then disable motors"""
        await self.set_tension_reg(False)
        sends = []
        for client in self.anchors.values():
            sends.append(client.send_commands({'stow': 0}))
            sends.append(client.send_commands({'stow': 1}))
        await asyncio.gather(*sends)

    async def wait_for_tension(self):
        """this function returns only once all anchors are reporting tight lines in their regular line record"""
        POLL_INTERVAL_S = 0.1 # seconds
        SPEED_SUM_THRESHOLD = 0.01 # m/s
        threshold = 0.5
        if self.config.anchor_type == common.AnchorType.ARPEGGIO:
            threshold = TENSION_THRESH
        
        complete = False
        timeout = time.time() + 10
        while not complete and time.time() < timeout:
            await asyncio.sleep(POLL_INTERVAL_S)
            records = np.array([alr.getLast() for alr in self.datastore.anchor_line_record])
            speeds = np.array(records[:,2])
            tension = np.array(records[:,3])
            complete = np.all(tension > threshold) and abs(np.sum(speeds)) < SPEED_SUM_THRESHOLD
        logger.debug(f'tension on lines = {tension}')
        return True

    async def tension_and_wait(self):
        """Send tightening command and wait until lines appear tight. This is not a motion task"""
        logger.info('Tightening all lines')
        await self.tension_lines()
        await self.wait_for_tension()

    async def sendReferenceLengths(self, lengths):
        if len(lengths) != N_LINES:
            logger.warning(f'Cannot send {len(lengths)} ref lengths to anchors')
            return
        for client in self.anchors.values():
            # which two lines is this anchor responsible for?
            asyncio.create_task(client.send_commands({
                'two_reference_lengths': (lengths[client.anchor_num*2], lengths[client.anchor_num*2+1])
            }))

        # reset biases on kalman filter
        data = self.datastore.gantry_pos.deepCopy()
        position = np.mean(data[:,2:], axis=0)
        logger.debug(f'Resetting filter biases with assumed position of {position}')
        self.pe.kf.reset_biases(position)

    async def stop_all(self):
        # stop swing cancellation so it does not keep commanding moves
        self.set_swing_cancellation(False)

        # zero input velocities from all sources
        self.zero_input_velocities()

        for maneuver in self.maneuvers.values():
            try:
                maneuver.on_stop_all()
            except Exception:
                logger.exception(f'Maneuver {maneuver.name} failed in on_stop_all')

        # Cancel any active motion task
        if self.motion_task is not None:
            # Store the handle and clear the class attribute immediately.
            # This prevents race conditions if another command comes in.
            task_to_stop = self.motion_task
            self.motion_task = None

            # Only cancel the task if it's actually still running.
            if not task_to_stop.done():
                logger.info(f"Cancelling motion task: {task_to_stop.get_name()}")
                task_to_stop.cancel()

            # await the task's completion.
            try:
                # Awaiting a task will re-raise any exception it had, or raise CancelledError if we just cancelled it.
                await task_to_stop
            except asyncio.CancelledError:
                # This is the expected, non-error outcome of a clean cancellation.
                logger.debug(f"Task '{task_to_stop.get_name()}' was successfully stopped.")
            except Exception:
                # If any other exception occurred, log it with traceback so it reaches every handler, not just stdout.
                logger.exception(f"An unhandled exception occurred in motion task '{task_to_stop.get_name()}'")

        await self._end_seek()
        self.slow_stop_all_spools()

    def slow_stop_all_spools(self):
        now = time.time()
        self.line_speed_cmds = [(now, 0.0, 'stop')] * N_LINES
        for name, client in self.bot_clients.items():
            # Slow stop all spools. gripper too
            asyncio.create_task(client.slow_stop_spool())
        self.pe.record_commanded_vel(np.zeros(3))
        # this stops the spools directly, bypassing move_direction_speed, so the stale
        # 'default' velocity must be cleared here too or it'll get summed back in the
        # next time anything (e.g. swing cancellation) triggers a combined move.
        self.input_velocities[DEFAULT_VELOCITY_KEY] = (np.zeros(3), time.monotonic())

    def snapshot_tag_observations(self, gantry_since=None):
        """Recent origin detections and cal_assist marker detections

        returns a dict of raw observations of various markers
        the shape of a pose is (2,3) with rotation coming first
        the first dimension is anchor number, the next is observation
        # for the arp anchor, the shape would be (2,12,2,3)

        'marker_name': array(n_anchors, n_observations, 2, 3)

        gantry_since keeps only gantry sightings captured at or after that time.time() timestamp.
        The consistency residual reads a marker's batch as repeated looks at one static point, so
        restrict it to a window in which the gantry was standing still.
        """
        markers = ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3', 'gantry']
        raw_obs = defaultdict(lambda: [[]]*N_ANCHORS)
        for client in self.anchors.values():
            # copy each list of detections, but leave them in the camera's reference frame.
            for marker in markers:
                if marker == 'gantry':
                    raw_obs[marker][client.anchor_num] = [
                        pose for ts, pose in list(client.raw_gant_poses)
                        if gantry_since is None or ts >= gantry_since
                    ]
                else:
                    raw_obs[marker][client.anchor_num] = list(client.origin_poses[marker])
                # print(f'anchor {client.anchor_num} has {len(raw_obs[marker][client.anchor_num])} observations of {marker}')
        return dict(raw_obs)

    async def await_still_gantry_window(self, min_dets=6, max_spread_m=0.04, timeout_s=12.0,
                                        what='measurement'):
        """Wait until the anchor cameras have delivered a batch of gantry sightings, all captured
        after this call, in which the gantry was standing still. Returns the capture time the
        batch starts at, or None if no settled batch turned up within timeout_s.

        This is what waiting for the machine to settle should cost: no longer than it takes the
        evidence to arrive. A fixed sleep has to be as long as the worst case swing and is still
        only a guess, whereas the sightings say directly both that the frames are new enough
        (captured after the cutoff, so they show the machine after whatever it was just asked to
        do) and that the gantry was holding still while they were taken.

        A settled batch reads about 1 cm by _robust_spread, so the default max_spread_m leaves 4x
        headroom and still rejects drift above roughly 0.3 m/s over the ~0.4s a six-frame window
        spans. A failed window is discarded and a new one opened, starting that much later.

        min_dets is required from one anchor rather than all: the gantry tag faces one way, so a
        camera seeing none of it is a normal geometry, not a reason to wait."""
        deadline = time.time() + timeout_s
        # Capture times come from the bot's clock, so start the window slightly in the future to
        # keep skew from letting a frame taken before this call through.
        window_start = time.time() + VIDEO_LATENCY_S

        while True:
            batches = {
                client.anchor_num: [pose for ts, pose in list(client.raw_gant_poses) if ts >= window_start]
                for client in self.anchors.values()
            }
            counts = {num: len(b) for num, b in batches.items()}
            # Only an anchor with a full batch can be judged. A partial one needs no check of its
            # own: its sightings fall inside the same window.
            spreads = {num: _robust_spread([p[1] for p in b])
                       for num, b in batches.items() if len(b) >= min_dets}
            if spreads:
                if max(spreads.values()) <= max_spread_m:
                    logger.info(
                        f'Settled gantry window for {what} after '
                        f'{timeout_s - (deadline - time.time()):.1f}s: {counts} sightings per '
                        f'anchor, camera-frame spread '
                        f'{({n: round(s * 100, 1) for n, s in spreads.items()})} cm'
                    )
                    return window_start
                logger.info(
                    f'Gantry still moving (camera-frame spread '
                    f'{({n: round(s * 100, 1) for n, s in spreads.items()})} cm '
                    f'> {max_spread_m * 100:.0f} cm); discarding this window, '
                    f'{deadline - time.time():.0f}s left before giving up.'
                )
                window_start = time.time() + VIDEO_LATENCY_S

            if time.time() >= deadline:
                logger.warning(
                    f'No settled gantry window for {what} within {timeout_s:.0f}s (last counts '
                    f'{counts}, wanted {min_dets} from at least one anchor under '
                    f'{max_spread_m * 100:.0f} cm spread).'
                )
                return None
            await asyncio.sleep(0.05)

    async def snapshot_tag_observations_still(self, min_dets=6, max_spread_m=0.04, timeout_s=12.0):
        """snapshot_tag_observations with the gantry sightings taken from a window in which the
        gantry was standing still.

        The gantry batch is the only one the consistency residual can be badly wrong about: the
        cards do not move, but the gantry buffer keeps filling while the machine flies around, and
        a batch spanning a move scatters far enough to dominate the whole cost function.

        Stops the spools, then waits out the settle on the sightings themselves. Falls back to the
        unfiltered buffer on timeout, since a noisy estimate beats none."""
        self.slow_stop_all_spools()
        window_start = await self.await_still_gantry_window(
            min_dets=min_dets, max_spread_m=max_spread_m, timeout_s=timeout_s,
            what='tag observation snapshot')
        if window_start is None:
            logger.warning('Falling back to the unfiltered gantry buffer.')
            return self.snapshot_tag_observations()
        return self.snapshot_tag_observations(gantry_since=window_start)

    def save_poses_arp(self, anchor_poses, eyelet_positions):
        # Use the optimization output to update anchor poses and spool params
        for anum, client in self.anchors.items():
            self.config.anchors[anum].pose = poseTupleToProto(anchor_poses[anum])
            self.config.anchors[anum].indirect_line.eyelet_pos = fromnp(eyelet_positions[anum])
            client.updatePoseAndEye(anchor_poses[anum], eyelet_positions[anum])
        save_config(self.config, self.config_path)
        # inform UI
        self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
            poses=[poseTupleToProto(p) for p in anchor_poses],
            eyelets=[fromnp(e) for e in eyelet_positions],
            calibrated=self.config.calibrated_status,
        ))
        # inform position estimator
        anchor_points = np.array([
            compose_poses([anchor_poses[0], model_constants.arp_anchor_right_eyelet])[1],
            eyelet_positions[0],
            compose_poses([anchor_poses[1], model_constants.arp_anchor_right_eyelet])[1],
            eyelet_positions[1],
        ])
        self.pe.set_anchor_points(anchor_points)

    async def touch_floor(self):
        await self.gripper_client.send_commands({'set_finger_angle': -30})
        laser_range = self.datastore.range_record.getLast()[1]
        logger.info(f'Touch the floor. current range: {laser_range}')
        try:
            await self.move_direction_speed(np.array([0, 0, -0.1]))
            timeout = time.time()+20
            while laser_range > 0.12 and time.time() < timeout:
                await asyncio.sleep(0.1)
                laser_range = self.datastore.range_record.getLast()[1]
                logger.debug(f'Laser range: {laser_range}')
        finally:
            self.slow_stop_all_spools()


    async def collect_arp_anchor_eyelet_experiment_data(self, anchor_poses, upper_z):
        """
        Perform experiments in which only the eyelet lines are tight and a diamond pattern is observed

        upper_z is the height (top of the work area, i.e. mean anchor z) in the room frame whose
        floor is at z=0. The diamond's vertical extent is sized automatically from it so that the
        top point leaves TOP_MARGIN_M of headroom below the work area while the bottom point (the
        gantry's current settled height) keeps the gripper fingers off the floor.
        """
        # target tension in newtons to hold the direct (anchor) lines at during the diamond
        DIAMOND_DIRECT_TENSION_N = 0.65
        # Stop a diamond move once an eyelet line is pulling this hard. The commanded jog comes
        # from eyelet positions guessed from origin-card views alone, so it can ask for a length
        # the geometry cannot reach and pull the lines up taut at the top of the work area. Well
        # under config.max_safe_tension, so this stops the move instead of passive_safety
        # aborting the whole procedure.
        DIAMOND_MAX_EYELET_TENSION_N = 20.0
        TENSION_RISE_N = 1.0   # a move must add at least this much to count as pulling taut
        SPOOL_SPIN_UP_S = 2.0  # ignore the stopped test until the spools have had time to start

        tilts = (self.config.anchors[0].indirect_line.cam_tilt, self.config.anchors[1].indirect_line.cam_tilt)

        try:
            for a in self.anchors.values():
                a.save_raw = True

            # touch the floor using the rangefinder
            # await self.touch_floor()

            self.slow_stop_all_spools()

            logger.info('Relax the direct lines, tighten the indirect line')

            # half_h (the diamond's vertical half-extent, as an eyelet line-length delta) is sized
            # automatically once the gantry has settled at the bottom point; see below. half_w (the
            # horizontal half-extent) keeps its configured value.
            _, half_w, _ = self.diamond_size
            # how far below the top of the work area (upper_z) the gantry's top point should stay.
            TOP_MARGIN_M = 1.15

            results = {}
            line_deltas = {}

            def get_eyelet_lengths():
                l1 = self.datastore.anchor_line_record[1].getLast()[1]
                l3 = self.datastore.anchor_line_record[3].getLast()[1]
                return l1, l3

            def eyelet_tension():
                """Highest tension on either eyelet (indirect) line, in newtons."""
                return max(float(self.pe.tension[1]), float(self.pe.tension[3]))

            async def wait_for_lines_to_stop(deadband=0.05, timeout=30, tension_limit=None):
                """Wait for every line to stop moving. Returns 'settled', or 'tension' as soon as
                an eyelet line passes tension_limit, or 'timeout'."""
                start = asyncio.get_event_loop().time()
                deadline = start + timeout
                while asyncio.get_event_loop().time() < deadline:
                    if tension_limit is not None and eyelet_tension() > tension_limit:
                        return 'tension'
                    # the spools take a moment to get going, so don't test for stopped until they have
                    if asyncio.get_event_loop().time() - start > SPOOL_SPIN_UP_S:
                        speeds = [abs(self.datastore.anchor_line_record[i].getLast()[2]) for i in range(N_LINES)]
                        if all(s < deadband for s in speeds):
                            await asyncio.sleep(2)
                            return 'settled'
                    await asyncio.sleep(1/30)
                logger.warning('wait_for_lines_to_stop timed out; proceeding with current line lengths')
                return 'timeout'

            async def move_to_diamond_point(jog1=0.0, jog3=0.0):
                """Reposition the gantry to a diamond point by jogging the two eyelet
                (indirect) lines. The two anchor (direct) lines are held at
                DIAMOND_DIRECT_TENSION_N by the onboard tension loop (set up below), so we
                only have to wait until every line has stopped moving before measuring.

                Stops short if an eyelet line goes taut. Wherever it stops is still a usable
                corner: each leg's length delta is measured after the move rather than assumed
                from the jog, and the corner's position comes from the anchor cameras. Only a
                rise counts, so a leg that pays line back out can start from an already-taut
                corner without being cut short immediately.

                A descending leg (the jogs lengthening on average) gets no tension check at all:
                descending is the only way out of a corner reached at the limit, and any check
                would trip on the tension already there. passive_safety still holds
                config.max_safe_tension throughout."""
                descending = (jog1 + jog3) > 0
                limit = None if descending else max(DIAMOND_MAX_EYELET_TENSION_N,
                                                    eyelet_tension() + TENSION_RISE_N)
                if descending:
                    logger.info('Diamond move descends; no tension check on this leg')
                if jog1:
                    await self.send_line_speed(1, jog1, jog=True)
                if jog3:
                    await self.send_line_speed(3, jog3, jog=True)
                reason = await wait_for_lines_to_stop(tension_limit=limit)
                await self.send_line_speed(1, 0)
                await self.send_line_speed(3, 0)
                if reason == 'tension':
                    logger.warning(
                        f'Diamond move stopped early: eyelet line reached {eyelet_tension():.1f}N '
                        f'(limit {limit:.1f}N). Measuring the position it got to.'
                    )
                return reason

            async def observe_corner(label):
                """Record this corner's gantry sightings once the anchor cameras have shown it
                standing still there. The corner is only reached when the lines stop, and the
                buffer still holds sightings from the move, so the batch has to be cut to frames
                captured after the move ended. await_still_gantry_window both makes that cut and
                waits for the swing to die down, in whatever time that actually takes."""
                # A corner is one point in the fit, so it wants a batch behind it rather than the
                # bare minimum that proves stillness; raw_gant_poses holds 24. The timeout is
                # short because the old fixed wait was 5s: a corner that will not settle should
                # cost about what it used to, not four times more.
                reached = time.time() + VIDEO_LATENCY_S
                since = await self.await_still_gantry_window(
                    min_dets=12, timeout_s=6.0, what=f'diamond {label}')
                if since is None:
                    # Nothing settled in time, but frames captured since the corner was reached
                    # are still the right ones, just fewer or more scattered than wanted.
                    since = reached
                    logger.warning(f'Diamond {label}: no settled window; measuring on whatever '
                                   f'arrived since the move ended.')
                batch = self.snapshot_tag_observations(gantry_since=since)['gantry']
                if not any(len(b) for b in batch):
                    # An empty corner would take a point out of the fit entirely, so a batch that
                    # spans the move is still the better of the two bad options.
                    logger.warning(f'Diamond {label}: no gantry sightings at all since the move '
                                   f'ended; falling back to the unfiltered buffer.')
                    batch = self.snapshot_tag_observations()['gantry']
                results[label] = batch

            # hand the direct lines to the onboard tension loop to hold at the target.
            # this runs at the component's loop rate with no wifi round trip, replacing the
            # host-side regulator that suffered from latency.
            await self.send_line_speed(0, 0)
            await self.send_line_speed(2, 0)
            await self.set_line_tension_target(0, DIAMOND_DIRECT_TENSION_N)
            await self.set_line_tension_target(2, DIAMOND_DIRECT_TENSION_N)

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=3.0,
                name="Calibration",
                current_action="Observe diamond bottom",
            ))
            logger.info('This position is the bottom of the diamond. Observe the gantry here')
            # regulate the anchor lines to the target tension and wait for everything to settle before measuring
            await move_to_diamond_point()
            await observe_corner('bottom')

            # Now that the gantry has settled at the bottom point, size the diamond's vertical
            # extent. Bottom is fixed (the gantry is here, with the fingers held off the floor by
            # the pre-diamond seek); the top point should sit TOP_MARGIN_M below the work area, so
            # the vertical travel we need is:
            gantry_pos = np.array(self.pe.gant_pos, dtype=float)
            target_span = (upper_z - TOP_MARGIN_M) - gantry_pos[2]
            # Convert that metric rise into an eyelet line-length delta. Raising the gantry straight
            # up by dz shortens each eyelet line by dz*cos(theta), where theta is that line's angle
            # from vertical. Over bottom->top each eyelet line shortens by 2*half_h, so
            # half_h = 0.5 * mean(cos theta) * span, using the current eyelet estimate.
            cosines = []
            for anchor in self.config.anchors:
                to_eyelet = tonp(anchor.indirect_line.eyelet_pos) - gantry_pos
                line_len = np.linalg.norm(to_eyelet)
                if line_len > 1e-6:
                    cosines.append((to_eyelet[2]) / line_len)
            cos_mean = float(np.mean(cosines)) if cosines else 1.0
            half_h = 0.5 * cos_mean * target_span
            # guard against a non-positive/degenerate span collapsing or inverting the diamond
            half_h = max(half_h, 0.05)
            logger.info(
                f'Sized diamond: bottom gantry z={gantry_pos[2]:.3f}, upper_z={upper_z:.3f}, '
                f'target vertical span={target_span:.3f} m, mean cos(theta)={cos_mean:.3f} '
                f'-> half_h={half_h:.3f} m (half_w={half_w:.3f} m)'
            )

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=6.0,
                name="Calibration",
                current_action="Observe diamond right",
            ))
            # RIGHT:
            logger.info('Move to RIGHT')
            l1_before, l3_before = get_eyelet_lengths()
            await move_to_diamond_point(jog1=-half_w-half_h, jog3=half_w-half_h)
            l1_after, l3_after = get_eyelet_lengths()
            line_deltas['bot_to_rig'] = (l1_after - l1_before, l3_after - l3_before)
            logger.info(f'bot_to_rig actual deltas: line1={line_deltas["bot_to_rig"][0]:.4f}, line3={line_deltas["bot_to_rig"][1]:.4f}')
            await observe_corner('right') # it is to the right from the perspective of camera 0

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=12.0,
                name="Calibration",
                current_action="Observe diamond top",
            ))
            # TOP:
            logger.info('Move to TOP')
            l1_before, l3_before = get_eyelet_lengths()
            await move_to_diamond_point(jog1=half_w-half_h, jog3=-half_w-half_h)
            l1_after, l3_after = get_eyelet_lengths()
            line_deltas['rig_to_top'] = (l1_after - l1_before, l3_after - l3_before)
            logger.info(f'rig_to_top actual deltas: line1={line_deltas["rig_to_top"][0]:.4f}, line3={line_deltas["rig_to_top"][1]:.4f}')
            await observe_corner('top')

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=17.0,
                name="Calibration",
                current_action="Observe diamond left",
            ))
            # LEFT:
            logger.info('Move to LEFT')
            l1_before, l3_before = get_eyelet_lengths()
            await move_to_diamond_point(jog1=half_w+half_h, jog3=-half_w+half_h)
            l1_after, l3_after = get_eyelet_lengths()
            line_deltas['top_to_lef'] = (l1_after - l1_before, l3_after - l3_before)
            logger.info(f'top_to_lef actual deltas: line1={line_deltas["top_to_lef"][0]:.4f}, line3={line_deltas["top_to_lef"][1]:.4f}')
            await observe_corner('left')

            # release the direct lines back to the normal tension floor
            await self.set_line_tension_target(0, None)
            await self.set_line_tension_target(2, None)

            logger.info('Return result')
            for a in self.anchors.values():
                a.save_raw = False

            analyze_diamond_data(results, anchor_poses, tilts, gantry_marker_inv=self.gantry_april_inv)

            return results, line_deltas

        except asyncio.CancelledError:
            raise
        finally:
            # always release the direct lines from hold mode, even on cancel, so they
            # don't stay regulating to the diamond target after the experiment ends.
            await self.set_line_tension_target(0, None)
            await self.set_line_tension_target(2, None)
            self.slow_stop_all_spools()

    def card_room_positions(self):
        """Best current estimate of each calibration card's room position, from the anchor
        cameras. Projects every stored anchor-camera sighting of each CAL marker into the room
        using the anchors' calibrated camera poses and averages them. Returns a dict keyed by
        marker name; markers never seen by any anchor are absent. Used to know where to fly the
        gripper for the close-range card survey, and to anchor those measurements in the room."""
        positions = {}
        for name in CAL_MARKERS:
            pts = []
            for client in self.anchors.values():
                for pose_cam in list(client.origin_poses.get(name, [])):
                    pts.append(compose_poses([client.camera_pose, pose_cam])[1])
            if pts:
                positions[name] = np.mean(pts, axis=0)
        return positions

    async def collect_gripper_card_observations(self, progress_range=None):
        """Fly the gripper over each calibration card in turn and record, from the gripper
        camera's close-range view, the room vector from the card to the gantry together with the
        four line lengths at that moment. This is a motion task.

        Returns a dict keyed by card name, each value a list of per-height samples; each sample is
        a dict with 'gantry_minus_card' (room vector) and 'line_lengths' (length-4 array), for
        passing to optimize_arp_anchors as gripper_obs. Each card is visited at several altitudes so
        the samples span a vertical baseline. Cards (or individual heights) the gripper never sees
        are skipped. Hover altitudes are taken relative to each card's own height, so cards may sit
        on the floor or raised.

        If progress_range=(start_pct, end_pct) is given, a Calibration operation_progress message
        is sent as each card is surveyed, spread across that percent range."""
        HOVER_CAMERA_HEIGHTS_M = [1.1, 0.7, 0.45]  # camera heights over each card to sample. Visiting a
                                                  # card from several altitudes gives the length-delta
                                                  # constraints a vertical baseline, which is what lets
                                                  # them begin to observe the far external eyelets (a
                                                  # single-height cluster leaves the eyelet radial
                                                  # direction free). The spread is deliberately wide -
                                                  # a wider baseline recovers more of a bad pass-2 - but
                                                  # each height is clamped under the work-area ceiling.
        SETTLE_S = 4.0                # let swing cancellation settle the gripper before measuring
        # How many views of the card to average the measurement over. A count, not a duration:
        # the averaging wants frames, and waiting on a clock only buys frames indirectly, at
        # whatever rate the gripper stream happens to be running.
        MEASURE_MIN_SAMPLES = 20
        MEASURE_TIMEOUT_S = 6.0       # give up on a card if the gripper never sees it
        SEEK_TIMEOUT_S = 20.0         # cap the move to each hover altitude
        TOP_MARGIN = 0.5              # meters under the top of the work area to keep gantry below

        if self.gripper_client is None:
            logger.warning('collect_gripper_card_observations requires a connected gripper')
            return {}

        card_positions = self.card_room_positions()
        if not card_positions:
            logger.warning('No calibration cards visible to the anchor cameras; cannot run gripper card survey')
            return {}

        # keep the gripper vertical and the lines taut throughout the survey. Put back as it
        # was when the survey finishes; a stop turns it off, as a stop always does.
        was_running = self.set_swing_cancellation(True)

        # don't fly higher than just under the top of the work area
        upper_z = np.mean(self.pe.anchor_points[:, 2]) - TOP_MARGIN

        survey_names = [n for n in ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3'] if n in card_positions]

        async def measure_hover(name, target_range_m):
            """Center on the card, settle onto the requested height, and average the card-to-gantry
            offset and line lengths over a short window. Returns a sample dict, or None if the
            gripper never sees the card here."""
            # center the card in view so the measurement is taken on the camera's axis, and so the
            # rangefinder is looking at the card rather than past it
            await self._center_card_in_view(name)
            # the seek only gets the altitude approximately right; the rangefinder gets it exact
            camera_height = await self.trim_altitude_to_range(target_range_m, ceiling_z=upper_z)
            # Let sightings accumulate in the gripper client's buffer, then take the whole window
            # at once. The cutoff sits a video latency ahead of now so nothing captured during the
            # trim can be counted; from there it is only a question of how long the stream takes
            # to deliver MEASURE_MIN_SAMPLES frames, which is as short as this step can honestly be.
            start = time.time() + VIDEO_LATENCY_S
            deadline = start + MEASURE_TIMEOUT_S
            while True:
                samples = self.gripper_client.get_route_tag_samples(name, since=start)
                if len(samples) >= MEASURE_MIN_SAMPLES:
                    break
                if time.time() >= deadline:
                    logger.info(f'Card survey: only {len(samples)} views of {name} in '
                                f'{MEASURE_TIMEOUT_S:.0f}s, wanted {MEASURE_MIN_SAMPLES}; '
                                f'measuring on what arrived')
                    break
                await asyncio.sleep(0.02)

            if not samples:
                return None
            # every quantity is evaluated at the frame's capture time, so the card pose,
            # the body orientation it is rotated by, and the line lengths it is paired
            # with all describe the same instant.
            gantry_offsets = [
                self.gripper_client.measure_gantry_minus_card(pose, timestamp=ts)
                for ts, pose in samples
            ]
            line_samples = [
                [self.datastore.anchor_line_record[i].getClosest(ts)[1] for i in range(N_LINES)]
                for ts, _ in samples
            ]
            return {
                'gantry_minus_card': np.mean(gantry_offsets, axis=0),
                'line_lengths': np.mean(line_samples, axis=0),
                'n': len(gantry_offsets),
                # measured, not requested: what the rangefinder read once the trim finished.
                # Recorded for diagnostics; the optimizer reads only the two arrays above.
                'camera_height': camera_height,
            }

        gripper_obs = {}
        try:
            for idx, name in enumerate(survey_names):
                if progress_range is not None:
                    start_pct, end_pct = progress_range
                    pct = start_pct + (end_pct - start_pct) * (idx + 1) / (len(survey_names) + 1)
                    self.send_ui(operation_progress=telemetry.OperationProgress(
                        percent_complete=pct,
                        name="Calibration",
                        current_action=f"Refining geometry: surveying card {idx + 1}/{len(survey_names)} ({name})",
                    ))
                cpos = card_positions[name]
                # gantry altitudes to sample this card from, clamped under the top of the work area,
                # deduplicated (a low ceiling can collapse several requests onto the same height), and
                # ordered highest-first so we approach high and descend through the samples.
                gant_zs = sorted({min(upper_z - 0.1, cpos[2] + self.pole[2] + h) for h in HOVER_CAMERA_HEIGHTS_M}, reverse=True)

                # Fly toward the anchor-camera estimate at the highest sampled height (widest view, so
                # the best chance to catch and center the card), but stop the moment the gripper camera
                # sees the card: its true spot can differ from the estimate, and continuing can carry it
                # back out of the narrow gripper FOV.
                approach_z = gant_zs[0]
                approach_goal = np.array([cpos[0], cpos[1], approach_z])
                logger.info(f'Gripper card survey: flying over {name} at goal {np.round(approach_goal, 3)} (card at {np.round(cpos, 3)})')
                seek_task = asyncio.create_task(self.seek_goal(approach_goal, head_turn=False))
                try:
                    while not seek_task.done():
                        if self.gripper_client.get_route_tag_pose(name) is not None:
                            logger.info(f'Gripper card survey: sighted {name} during approach; stopping to hold it in view')
                            break
                        await asyncio.sleep(0.03)
                finally:
                    if not seek_task.done():
                        seek_task.cancel()
                    try:
                        await seek_task
                    except asyncio.CancelledError:
                        pass
                self.slow_stop_all_spools()

                # Measure the card from each altitude in turn. The spread in height is the whole point:
                # it gives the length-delta constraints a vertical baseline to triangulate the eyelets.
                samples = []
                for i, gz in enumerate(gant_zs):
                    # the camera hangs self.pole below the gantry, so this is the height the camera (and
                    # the rangefinder beside it) should end up at. Derived from the clamped gz rather
                    # than from h, so a height the ceiling cut short trims to what it can reach.
                    target_range = gz - self.pole[2] - cpos[2]
                    if i == 0:
                        # hold the exact target altitude (auto_altitude would cruise at a fixed height and
                        # defeat the point of sampling several).
                        seek_task = asyncio.create_task(self.seek_goal(
                            np.array([cpos[0], cpos[1], gz]), head_turn=False, auto_altitude=False))
                        try:
                            await asyncio.wait_for(seek_task, timeout=SEEK_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            logger.warning(f'Gripper card survey: did not reach z={gz:.2f} over {name} within {SEEK_TIMEOUT_S:.0f}s; measuring anyway')
                    else:
                        # The rest of the heights sit directly under the first one and centering has
                        # already put the gripper over the card, so just drop to them. Another seek
                        # would re-run its whole xy approach only to finish within GOAL_PROXIMITY_M;
                        # the rangefinder trim in measure_hover is what actually lands the height.
                        delta_z = gz - gant_zs[i - 1]
                        logger.info(f'Gripper card survey: dropping {delta_z:+.2f}m to z={gz:.2f} over {name}')
                        await self.nudge_gantry(np.array([0.0, 0.0, delta_z]), max_step=1.0)
                    self.slow_stop_all_spools()
                    await asyncio.sleep(SETTLE_S)

                    sample = await measure_hover(name, target_range)
                    if sample is None:
                        logger.warning(f'Gripper card survey: never saw {name} at gantry z={gz:.2f}; skipping this height')
                        continue
                    samples.append(sample)
                    measured = sample["camera_height"]
                    logger.info(
                        f'Gripper card survey: {name} z={gz:.2f} n={sample["n"]} '
                        f'camera height wanted {target_range:.2f}m got '
                        f'{"unknown" if measured is None else f"{measured:.2f}m"} '
                        f'gantry_minus_card={np.round(sample["gantry_minus_card"], 3)} '
                        f'lines={np.round(sample["line_lengths"], 3)}'
                    )

                if samples:
                    gripper_obs[name] = samples
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
        self.set_swing_cancellation(was_running)

        total = sum(len(s) for s in gripper_obs.values())
        logger.info(f'Gripper card survey collected {total} hover samples across {len(gripper_obs)} cards: '
                    f'{ {k: len(v) for k, v in gripper_obs.items()} }')
        return gripper_obs

    async def _nudge_gantry_xy(self, delta_xy, speed=NUDGE_SPEED_MPS):
        """Move the gantry a small horizontal step of (approximately) delta_xy meters, then stop."""
        return await self.nudge_gantry(np.array([delta_xy[0], delta_xy[1], 0.0]), speed=speed)

    async def nudge_gantry(self, delta, speed=NUDGE_SPEED_MPS, max_step=0.35):
        """Move the gantry a small step of (approximately) delta meters, then stop.
        Commands a velocity rather than a speed along a direction, so the step follows delta
        apart from move_direction_speed's own downward bias. max_step caps how far one call
        will travel; raise it for a deliberate transit rather than a correction.
        Returns the time the spools were stopped, which bounds when a settled view can appear."""
        dist = float(np.linalg.norm(delta))
        if dist < 0.005:
            return time.time()
        dist = min(dist, max_step)  # cap a single nudge for safety
        uvec = np.asarray(delta, dtype=float)
        uvec = uvec / (np.linalg.norm(uvec) + 1e-9)
        # Hold the velocity on our own source key rather than 'default': a UI sending idle
        # zero-velocity moves owns 'default' and would overwrite the nudge the instant it
        # arrived. Sources sum, so an idle 'default' adds nothing to ours. Re-issued every
        # step, both to follow the ease profile and to recompute the line speeds from where
        # the gantry has actually got to.
        total, ramp = eased_move_time(dist, speed, NUDGE_RAMP_S)
        started = time.monotonic()
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= total:
                break
            await self.move_direction_speed(uvec * speed * eased_speed(elapsed, total, ramp),
                                            None, self.pe.gant_pos, key=NUDGE_VELOCITY_KEY)
            await asyncio.sleep(NUDGE_STEP_S)
        await self.move_direction_speed(np.zeros(3), 0, key=NUDGE_VELOCITY_KEY)
        self.slow_stop_all_spools()
        await asyncio.sleep(NUDGE_SETTLE_S)
        return time.time()

    def laser_range(self):
        """The gripper's rangefinder reading in metres - the distance to whatever is under it -
        or None if it is stale or absent."""
        ts, distance = self.datastore.range_record.getLast()
        if time.time() - ts > RANGE_MAX_AGE_S or distance <= 0:
            return None
        return float(distance)

    # -- getters for maneuvers. Each returns a copy, so nothing a maneuver does to the
    # result reaches the estimator or a component client.

    def gantry_position(self):
        """Where the position estimate puts the gantry, as a room-frame (3,) array."""
        return np.array(self.pe.gant_pos, dtype=float)

    def visual_gantry_position(self):
        """The anchor cameras' running average of where the gantry marker is."""
        return np.array(self.pe.visual_pos, dtype=float)

    def anchor_points(self):
        """The (4, 3) room positions the four lines leave from."""
        return np.array(self.pe.anchor_points, dtype=float)

    def line_tensions(self):
        """Per-line tension in newtons, as a (4,) array, or None before any has been reported."""
        if self.pe.tension is None:
            return None
        return np.array(self.pe.tension, dtype=float)

    def gripper_connected(self):
        return self.gripper_client is not None

    def wrist_angle(self):
        """The wrist angle the gripper last reported, in degrees (0 to 1080)."""
        return float(self.datastore.winch_line_record.getLast()[1])

    def pole_tilt(self, max_age=1.0):
        """How far the pole leans off vertical, in degrees, or None when the gripper has not
        said for max_age seconds."""
        gripper = self.gripper_client
        if gripper is None or gripper.last_angle_from_vertical is None:
            return None
        if time.time() - gripper.angle_from_vertical_ts > max_age:
            return None
        return float(gripper.last_angle_from_vertical)

    def wrist_angle_for_heading(self, heading):
        """The wrist angle that points the gripper's nose along a room-frame heading (radians)."""
        return self.gripper_client.wrist_angle_for_spin(heading)

    def gripper_camera_intrinsics(self):
        """The gripper camera's 3x3 intrinsic matrix."""
        return np.array(self.config.camera_cal_wide.intrinsic_matrix).reshape(3, 3)

    def gripper_camera_to_room(self, vec, timestamp=None):
        """A vector in the gripper camera's optical frame (x right, y down, z along the axis),
        rotated into the room frame, with the gripper as it was at timestamp (now, by default).

        Goes through the same chain as measure_gantry_minus_card, which is the one place that
        knows how the camera is mounted and which way the pole is leaning; guessing it from
        the nose heading instead leaves a 180 degree ambiguity.
        """
        in_gripper = Rotation.from_rotvec(model_constants.gripper_camera[0]).apply(vec)
        in_body = Rotation.from_euler('x', 90, degrees=True).apply(in_gripper)
        return self.gripper_client.gripper_body_room_rotation(timestamp=timestamp).apply(in_body)

    async def gripper_frame(self, after=None, timeout=3.0):
        """An RGB frame from the gripper camera captured after `after` (now, by default), or
        None if none arrives within timeout. after=0 takes the newest frame without waiting
        for a fresher one."""
        if self.gripper_client is None:
            return None
        _, frame = await self.gripper_client.capture_raw_frame(
            time.time() if after is None else after, timeout=timeout)
        return frame

    async def set_finger_angle(self, angle):
        """Command the fingers to angle (degrees, -90 open to 90 closed) without waiting."""
        if self.gripper_client is not None:
            await self.gripper_client.send_commands({'set_finger_angle': angle})

    async def set_wrist_angle(self, angle):
        """Command the wrist to angle (degrees, 0 to 1080) without waiting."""
        if self.gripper_client is not None:
            await self.gripper_client.send_commands({'set_wrist_angle': float(angle)})

    def start_spool_log(self, path=None):
        """Also write everything monitor_spools sees to a JSON Lines file until stop_spool_log:
        every line record, where the gantry is, and what each line was told to do. Turns on the
        anchors' SPOOL_DIAG while it runs, which adds the raw torque and the soft mute's
        decisions from anchors whose firmware has it."""
        if self.spool_log is not None:
            logger.info('spool log already running')
            return
        if path is None:
            path = f'spool_diag_{int(time.time())}.jsonl'
        self.spool_log = open(path, 'w')
        self.spool_log.write(json.dumps({
            'header': True,
            'start': time.time(),
            'anchor_points': np.asarray(self.pe.anchor_points, dtype=float).tolist(),
            'anchors_connected': sorted(self.anchors),
            'component_vars': dict(self.config.component_vars),
            'line_row': ['t', 'length_m', 'speed_mps', 'tension_n'],
            'diag_row': ['t', 'torque_nm', 'hold_nm_motor_frame', 'motor_vel_rad_s',
                         'meters_per_rev', 'aim_mps', 'mute', 'wanted_mps', 'tension_reg',
                         'mute_tension_n'],
        }) + '\n')
        asyncio.create_task(self._set_spool_diag(True))
        logger.info(f'Spool log writing to {path}')

    def stop_spool_log(self):
        if self.spool_log is None:
            return
        logger.info(f'Spool log stopped, {self.spool_log.name}')
        self.spool_log.close()
        self.spool_log = None
        asyncio.create_task(self._set_spool_diag(False))

    def _report_derailed_spool(self, line_no, moved, resisted):
        """Stop whatever is moving the gantry, tell the operator which spool has derailed, and
        write out the spool history that led up to it.
        Carrying on pays out more line from the other spools against one that cannot follow."""
        spool = 'upper (direct)' if line_no % 2 == 0 else 'lower (indirect)'
        logger.warning(f'Line {line_no} looks derailed: paid out {moved:.0%} of what it was told '
                       f'to, {resisted:.0%} of its payout records resisted. Stopping.')
        # not awaited, so the monitor keeps recording while the motion task winds down
        asyncio.create_task(self.stop_all())
        self.send_ui(pop_message=telemetry.Popup(message=(
            f'The {spool} spool on anchor {line_no // 2} appears to have lost its line: it cannot '
            f'pay out. All motion has been stopped. Check that the line is seated on the spool.')))
        now = time.time()
        header = {
            'header': True,
            't': now,
            'line': line_no,
            'moved_fraction': moved,
            'resisted_fraction': resisted,
            'line_row': ['t', 'speed_mps', 'tension_n'],
            'cmd': 'aim speed in m/s the line was last told, null if unknown or a jog',
        }
        asyncio.create_task(asyncio.to_thread(
            self._write_derail_event, f'derail_event_{int(now * 1000)}.jsonl', header,
            list(self.spool_history)))

    @staticmethod
    def _write_derail_event(path, header, history):
        with open(path, 'w') as f:
            f.write(json.dumps(header) + '\n')
            for tick in history:
                f.write(json.dumps(tick) + '\n')
        logger.info(f'Wrote derailment event to {path}')

    async def _set_spool_diag(self, on):
        await asyncio.gather(*[
            client.send_commands({'set_config_vars': {'SPOOL_DIAG': 1 if on else 0}})
            for client in self.anchors.values()
        ], return_exceptions=True)

    async def monitor_spools(self, period_s=0.1):
        """Watch every spool for having lost its line, for as long as the observer runs.

        Keeps the last SPOOL_HISTORY_S of what the detector reads, trimmed to what it uses, so
        a detection can be written out with what led up to it. With a spool log open, also
        writes everything to that.
        """
        detector = DerailDetector(N_LINES)
        while True:
            await asyncio.sleep(period_s)
            now = time.time()
            tick = {'t': now, 'lines': []}
            trips = []
            verbose = [] if self.spool_log is not None else None
            for line_no in range(N_LINES):
                client = self.anchors.get(line_no // 2)
                rows, diag = [], []
                if client is not None:
                    q_rows = client.spool_log_rows[line_no % 2]
                    q_diag = client.spool_log_diag[line_no % 2]
                    while q_rows:
                        rows.append(q_rows.popleft())
                    while q_diag:
                        diag.append(q_diag.popleft())
                cmd = self.line_speed_cmds[line_no]
                cmd_speed = None if cmd is None or cmd[2] == 'jog' else cmd[1]
                detector.add(line_no, rows, cmd_speed)
                derailed, moved, resisted = detector.verdict(line_no)
                tick['lines'].append({
                    'cmd': cmd_speed,
                    'rows': [(r[0], r[2], r[3]) for r in rows],
                    'derailed': derailed,
                })
                if verbose is not None:
                    verbose.append({'cmd': cmd, 'rows': rows, 'diag': diag, 'derailed': derailed})
                if derailed:
                    detector.reset(line_no)
                    trips.append((line_no, moved, resisted))
            self.spool_history.append(tick)
            for trip in trips:
                self._report_derailed_spool(*trip)
            if verbose is not None:
                self._write_spool_log_tick(now, verbose)

    def _write_spool_log_tick(self, now, lines):
        def arr(v):
            return None if v is None else np.asarray(v, dtype=float).tolist()
        gant_pos = np.asarray(self.pe.gant_pos, dtype=float)
        geom = np.linalg.norm(np.asarray(self.pe.anchor_points) - gant_pos, axis=1)
        for line_no, line in enumerate(lines):
            line['geom_len'] = float(geom[line_no])
        self.spool_log.write(json.dumps({
            't': now,
            'gant_pos': arr(gant_pos),
            'gant_vel': arr(self.pe.gant_vel),
            'hang_pos': arr(self.pe.hang_pos),
            'visual_pos': arr(self.pe.visual_pos),
            'slack_lines': [bool(x) for x in self.pe.slack_lines],
            'holding': bool(self.pe.holding),
            'lines': lines,
        }) + '\n')
        self.spool_log.flush()

    def save_config(self):
        """Write the robot config out. A maneuver writes only its own field before calling it."""
        save_config(self.config, self.config_path)

    def pole_offset(self):
        """The (3,) offset from the gantry down to where the gripper hangs on its pole. Add
        it to a gripper goal to get the gantry goal seek_goal wants."""
        return np.array(self.pole, dtype=float)

    def gripper_position(self):
        """Where the position estimate puts the gripper, as a room-frame (3,) array."""
        return np.array(self.pe.grip_pose[1], dtype=float)

    def is_holding(self):
        """Whether the fingers are closed on something, as the estimator judges it."""
        return bool(self.pe.holding)

    def inside_work_area_2d(self, point):
        return self.pe.point_inside_work_area_2d(np.asarray(point, dtype=float)[:2])

    def finger_angle(self):
        """The finger angle the gripper last reported, in degrees (-90 open to 90 closed)."""
        return float(self.datastore.finger.getLast()[1])

    def finger_pad_voltage(self):
        """The finger pressure pad reading the gripper last reported."""
        return float(self.datastore.finger.getLast()[2])

    def reset_finger_pressure_rising(self):
        """Forget any finger pressure rise seen so far, so finger_pressure_rose asks afresh."""
        self.pe.finger_pressure_rising.clear()

    def finger_pressure_rose(self):
        """Whether the finger pressure has risen since reset_finger_pressure_rising."""
        return self.pe.finger_pressure_rising.is_set()

    def robot_id(self):
        """This robot's id with the control plane it is running against, or None."""
        return self.telemetry.cloud_robot_id

    def ortho_enabled(self):
        return self.run_ortho

    def latest_ortho(self):
        """The newest orthographic floor view as RGB, or None. Replaced, never written into,
        so holding it is safe; copy it before keeping it past the next frame."""
        return self.last_ortho_rgb

    def anchor_pixel_to_floor(self, anchor_num, norm_xy):
        """Where a point in an anchor camera's image (normalized 0-1 coordinates) lands on the
        floor, or None if that anchor is not connected or the ray misses."""
        if anchor_num not in self.anchors:
            return None
        points = project_pixels_to_floor([list(norm_xy)], self.anchors[anchor_num].camera_pose,
                                         self.config.camera_cal)
        return points[0] if len(points) == 1 else None

    def route_tag_samples(self, name, since):
        """The gripper camera's (timestamp, pose) sightings of a route tag since a time."""
        if self.gripper_client is None:
            return []
        return self.gripper_client.get_route_tag_samples(name, since=since)

    def gantry_minus_card(self, pose, timestamp):
        """Room-frame gantry position relative to a card, from one gripper camera sighting."""
        return self.gripper_client.measure_gantry_minus_card(pose, timestamp=timestamp)

    async def gripper_capture(self, after=None, timeout=3.0, expect_size=None):
        """(timestamp, RGB frame) from the gripper camera captured after `after` (now, by
        default), or (None, None) if none arrives within timeout. expect_size holds out for
        frames of that (width, height)."""
        if self.gripper_client is None:
            return None, None
        return await self.gripper_client.capture_raw_frame(
            time.time() if after is None else after, timeout=timeout, expect_size=expect_size)

    async def use_gripper_capture_stream(self):
        """Switch the gripper camera to its full capture resolution. It stays there for the
        rest of the session: switching back costs a stream restart."""
        await self.gripper_client.use_capture_stream()

    def start_gripper_recording(self, path):
        """Record the gripper camera's compressed stream to path, as it arrives."""
        self.gripper_client.recording_path = path

    def gripper_recorded_packets(self):
        return self.gripper_client.recorded_packets

    def stop_gripper_recording(self):
        """Stop recording and return (packets recorded, stream start timestamp). The file is
        closed when the next packet arrives."""
        client = self.gripper_client
        packets, stream_start_ts = client.recorded_packets, client.recording_stream_start_ts
        client.recording_path = None
        return packets, stream_start_ts

    async def set_wrist_speed(self, dps):
        """Turn the wrist at dps degrees per second. The gripper zeroes it after a short
        timeout, so it has to be repeated to keep turning."""
        if self.gripper_client is not None:
            await self.gripper_client.send_commands({'set_wrist_speed': float(dps)})

    def torch_device(self):
        """The torch device every model on this host shares. The first call imports torch,
        so make it from a worker thread."""
        if self._device is None:
            import torch
            self._device = ("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available() else "cpu")
        return self._device

    def last_clear_item_image(self):
        """The newest ItemImage: a gripper frame taken while the rangefinder read the distance
        at which an item under the gripper fills the view before the fingers close over it.
        None until there has been one."""
        return self._clear_item_image

    async def _watch_for_clear_item(self, interval_s=CLEAR_ITEM_INTERVAL_S):
        """Keep the newest gripper frame taken with the laser in CLEAR_ITEM_RANGE_M."""
        low, high = CLEAR_ITEM_RANGE_M
        while self.run_command_loop:
            await asyncio.sleep(interval_s)
            client = self.gripper_client
            if client is None or client.last_output_frame is None:
                continue
            laser = self.laser_range()
            if laser is None or not (low <= laser <= high):
                continue
            taken = client.last_frame_cap_time
            last = self._clear_item_image
            if last is not None and taken is not None and taken <= last.timestamp:
                continue
            # decoded frames arrive BGR; the getters all hand out RGB
            self._clear_item_image = ItemImage(
                image_rgb=cv2.cvtColor(client.last_output_frame, cv2.COLOR_BGR2RGB),
                timestamp=taken or time.time(),
                laser_range=laser,
                gantry_position=self.gantry_position(),
            )

    async def trim_altitude_to_range(self, target_range_m, tol_m=0.02, max_steps=4,
                                      ceiling_z=None, max_travel_m=None):
        """Close the gantry's altitude onto the height where the downward rangefinder reads
        target_range_m, and report the range finally measured (None if it never got a reading).

        The rangefinder is coplanar with the gripper camera, so its reading is the camera's
        height above whatever is beneath it - the card, once centering has put the gripper over
        it. Seeking to a computed gantry z only lands within GOAL_PROXIMITY_M and inherits any
        bias in the position estimate, so the sampled hover heights are otherwise approximate.
        Measuring the height directly makes them what was asked for.

        Call this with the card already centered, or the beam may be reading the floor beside a
        raised card rather than the card itself.

        max_travel_m bounds the total vertical distance this may cover, for a caller that
        knows roughly how wrong the altitude can be and would rather stop than keep hunting
        on a beam that has found something other than what it was aimed at."""
        laser_range = None
        travelled = 0.0
        for step in range(max_steps):
            ts, laser_range = self.datastore.range_record.getLast()
            age = time.time() - ts
            if age > RANGE_MAX_AGE_S:
                logger.warning(f'Altitude trim: rangefinder reading is {age:.1f}s old; leaving altitude as-is')
                return None
            error = target_range_m - laser_range  # positive means we are too low and must rise
            if abs(error) < tol_m:
                logger.info(f'Altitude trim: range {laser_range:.3f}m within {tol_m*100:.0f}cm '
                            f'of target {target_range_m:.3f}m after {step} steps')
                return laser_range
            delta_z = clamp(error, -0.35, 0.35)
            if ceiling_z is not None:
                delta_z = min(delta_z, ceiling_z - self.pe.gant_pos[2])
            if max_travel_m is not None:
                budget = max_travel_m - travelled
                if budget <= 0:
                    logger.info(f'Altitude trim: used the whole {max_travel_m:.2f}m of travel '
                                f'at range {laser_range:.3f}m (target {target_range_m:.3f}m)')
                    return laser_range
                delta_z = clamp(delta_z, -budget, budget)
            travelled += abs(delta_z)
            logger.info(f'Altitude trim: step {step} range {laser_range:.3f}m vs target '
                        f'{target_range_m:.3f}m, moving z by {delta_z:+.3f}m')
            await self.nudge_gantry(np.array([0.0, 0.0, delta_z]), speed=TRIM_SPEED_MPS)
        logger.info(f'Altitude trim: reached max steps at range {laser_range:.3f}m '
                    f'(target {target_range_m:.3f}m)')
        return laser_range

    async def _await_card_pose(self, name, after_ts, timeout=1.5):
        """Newest sighting of the named card captured after after_ts, or None if none arrives
        within timeout. Waiting on capture time rather than a fixed sleep means the next step
        uses a view taken after the previous nudge finished, however long the stream lags."""
        deadline = time.time() + timeout
        while True:
            samples = self.gripper_client.get_route_tag_samples(name, since=after_ts)
            if samples:
                return samples[-1][1]
            if time.time() > deadline:
                return None
            await asyncio.sleep(0.02)

    async def _center_card_in_view(self, name, tol_m=0.03, gain=0.6, max_steps=12):
        """Bounded visual-centering: nudge the gantry so the named card sits under the gripper
        camera. measure_gantry_minus_card gives the room offset from card to gantry; moving the
        gantry by the negative of its horizontal part drives that toward zero (gantry over card,
        card centered). Stops when centered, when the card is lost, when a nudge grows the error
        (the room heading the error is expressed in is only as good as the spin calibration, and
        a bad one sends every nudge off in a fixed wrong direction), or after max_steps."""
        prev = None
        # the first look may use any sighting still inside the normal freshness bound
        after_ts = time.time() - ROUTE_TAG_MAX_AGE_S
        for step in range(max_steps):
            pose_cam = await self._await_card_pose(name, after_ts)
            if pose_cam is None:
                logger.info(f'Centering {name}: lost from view at step {step}; measuring as-is')
                return
            err_xy = self.gripper_client.measure_gantry_minus_card(pose_cam)[:2]
            err = float(np.linalg.norm(err_xy))
            if err < tol_m:
                logger.info(f'Centering {name}: within {err*100:.1f}cm after {step} steps')
                return
            if prev is not None and err > prev + 0.02:
                logger.warning(f'Centering {name}: error grew ({prev*100:.1f}->{err*100:.1f}cm); '
                               f'stopping. Check the room spin calibration.')
                return
            prev = err
            # err_xy points from the card to the gantry, so close it by moving the other way
            nudge = -gain * err_xy
            logger.info(f'Centering {name}: step {step} err {err*100:.1f}cm, '
                        f'nudging {np.round(nudge, 3)} ({np.linalg.norm(nudge)/NUDGE_SPEED_MPS:.1f}s)')
            after_ts = await self._nudge_gantry_xy(nudge)
        logger.info(f'Centering {name}: reached max steps')

    async def find_origin_card(self, card_pos=None, upper_z=None):
        """Park the gripper where its camera can see the origin card, climbing no higher than needed.

        Run after the 2nd optimization pass, where the geometry is good enough to fly to a point
        but not to trust the altitude it arrives at. Flying straight to a nominal height is what
        makes this step fragile: too low and the card never enters the narrow gripper FOV, too
        high and a card up on a bed or table puts the gantry into the ceiling and trips the
        tension limit, which aborts the whole calibration. So the search starts a hand's width
        over the card, where a sighting is nearly certain if the position estimate is good, and
        opens outward from there - upward while there is headroom, then, at the top of the work
        area, in a horizontal spiral around the card.

        Near the ceiling a poorly calibrated robot's horizontal moves pull the gantry up, so
        every spiral step carries a downward correction back to the altitude the search settled
        at, and line tension approaching the safety limit pushes that altitude down.

        Returns the gantry z it finished at, sighting or not: the caller's centering step is
        what makes use of the sighting, and its own approach altitude follows from this one.
        This is a motion task."""
        START_CAM_HEIGHT_M = 0.10    # camera height over the card the search starts at
        MAX_CAM_HEIGHT_M = 1.3       # highest over the card the card is still worth looking for from
        RISE_STEP_M = 0.15           # how far one upward search step climbs
        TOP_MARGIN_M = 0.1           # stay this far under the top of the work area
        SPIRAL_STEP_M = 0.2          # spacing between horizontal search points
        SPIRAL_MAX_R_M = 0.6         # widest the spiral searches around the card
        TENSION_BACKOFF_FRAC = 0.7   # descend once a line passes this fraction of the safe limit
        BACKOFF_DROP_M = 0.1         # how far to descend when tension says the gantry is too high
        LOOK_S = 1.5                 # how long to wait for a view taken after a step

        if self.gripper_client is None:
            logger.warning('Finding the origin card requires a connected gripper')
            return float(self.pe.gant_pos[2])

        if card_pos is None:
            # the room frame puts the origin card at (0, 0) by construction; the anchor cameras
            # are what know how high its perch is, and only if they can see it
            card_pos = self.card_room_positions().get('origin', np.zeros(3))
        card_pos = np.asarray(card_pos, dtype=float)
        if upper_z is None:
            upper_z = float(np.mean(self.pe.anchor_points[[0, 2], 2]))

        # altitudes here are the gantry's; the camera hangs self.pole[2] under it
        ceiling_z = upper_z - TOP_MARGIN_M
        top_z = min(ceiling_z, card_pos[2] + self.pole[2] + MAX_CAM_HEIGHT_M)
        start_z = min(top_z, card_pos[2] + self.pole[2] + START_CAM_HEIGHT_M)

        async def sighted_since(after_ts):
            return (await self._await_card_pose('origin', after_ts, timeout=LOOK_S)) is not None

        logger.info(f'Finding origin card: card at {np.round(card_pos, 3)}, starting at gantry z '
                    f'{start_z:.2f} (ceiling {ceiling_z:.2f}, climbing to at most {top_z:.2f})')

        # Fly to the start, but stop the moment the card appears: the approach can pass over it,
        # and carrying on would take it back out of the gripper camera's narrow view.
        seek = asyncio.create_task(self.seek_goal(
            np.array([card_pos[0], card_pos[1], start_z]), head_turn=False, auto_altitude=False))
        try:
            while not seek.done():
                if self.gripper_client.get_route_tag_pose('origin') is not None:
                    logger.info('Finding origin card: sighted during the approach')
                    break
                await asyncio.sleep(0.03)
        finally:
            if not seek.done():
                seek.cancel()
                try:
                    await seek
                except asyncio.CancelledError:
                    pass
        self.slow_stop_all_spools()
        if self.gripper_client.get_route_tag_pose('origin') is not None:
            return float(self.pe.gant_pos[2])

        # Climb: a higher camera sees more floor, so the card is likelier to fall inside the view
        # even where the position estimate has put the gantry off to one side.
        hold_z = float(self.pe.gant_pos[2])
        while True:
            if self._tension_near_limit(TENSION_BACKOFF_FRAC):
                logger.warning('Finding origin card: line tension near the limit; dropping instead '
                               'of climbing further')
                await self.nudge_gantry(np.array([0.0, 0.0, -BACKOFF_DROP_M]), speed=TRIM_SPEED_MPS)
                hold_z = float(self.pe.gant_pos[2])
                break
            headroom = top_z - float(self.pe.gant_pos[2])
            if headroom < 0.02:
                break
            after_ts = await self.nudge_gantry(np.array([0.0, 0.0, min(RISE_STEP_M, headroom)]),
                                                speed=TRIM_SPEED_MPS)
            hold_z = float(self.pe.gant_pos[2])
            logger.info(f'Finding origin card: looking from gantry z {hold_z:.2f} '
                        f'(camera {hold_z - self.pole[2] - card_pos[2]:.2f} over the card)')
            if await sighted_since(after_ts):
                logger.info('Finding origin card: sighted while climbing')
                return hold_z

        # Out of headroom, so widen the search sideways instead. Every step is a single nudge
        # combining the horizontal move with whatever descent it takes to undo the climb the last
        # one caused; the vertical part is never positive, since climbing is what trips tension.
        for idx, (dx, dy) in enumerate(_spiral_waypoints(SPIRAL_STEP_M, SPIRAL_MAX_R_M)):
            # start_z is the floor of this: it is a hand's width over the card, and any lower
            # looks under it rather than at it, so a robot whose tension never settles gives up
            # altitude only down to there.
            if self._tension_near_limit(TENSION_BACKOFF_FRAC) and hold_z > start_z:
                hold_z = max(start_z, hold_z - BACKOFF_DROP_M)
                logger.warning(f'Finding origin card: line tension near the limit; searching '
                               f'lower, at gantry z {hold_z:.2f}')
            delta = np.array([card_pos[0] + dx, card_pos[1] + dy, hold_z]) - self.pe.gant_pos
            delta[2] = min(0.0, delta[2])
            logger.info(f'Finding origin card: spiral point {idx} at '
                        f'{np.round([card_pos[0] + dx, card_pos[1] + dy], 3)}, '
                        f'moving {np.round(delta, 3)}')
            after_ts = await self.nudge_gantry(delta)
            if await sighted_since(after_ts):
                logger.info(f'Finding origin card: sighted at spiral point {idx}')
                return float(self.pe.gant_pos[2])

        logger.warning('Finding origin card: not seen from any search position; '
                       'continuing from where the search ended')
        return float(self.pe.gant_pos[2])

    async def settle_wrist(self, target, tol=2.0, timeout=6.0):
        """Command the wrist to an absolute angle and wait until telemetry agrees."""
        await self.gripper_client.send_commands({'set_wrist_angle': target})
        deadline = time.time() + timeout
        actual = None
        while time.time() < deadline:
            await asyncio.sleep(0.05)
            actual = self.datastore.winch_line_record.getLast()[1]
            if abs(actual - target) <= tol:
                return actual
        logger.warning(f'Wrist did not reach {target:.1f} within {timeout}s (at {actual})')
        return actual

    async def settle_fingers(self, target, tol=2.0, timeout=6.0):
        """Command the fingers to an absolute angle and wait until telemetry agrees."""
        await self.gripper_client.send_commands({'set_finger_angle': target})
        deadline = time.time() + timeout
        actual = None
        while time.time() < deadline:
            await asyncio.sleep(0.05)
            actual = self.datastore.finger.getLast()[1]
            if abs(actual - target) <= tol:
                return actual
        logger.warning(f'Fingers did not reach {target:.1f} within {timeout}s (at {actual})')
        return actual

    async def half_auto_calibration(self):
        """
        Set line lengths from observation
        tighten, wait for obs, estimate line lengths, move up slightly, estimate line lengths, move down slightly
        This is a motion task
        """
        NUM_SAMPLE_POINTS = 3
        OPTIMIZER_TIMEOUT_S = 60  # seconds
        
        try:
            if len(self.anchors) < N_ANCHORS:
                logger.warning('Cannot run half calibration until all anchors are connected')
                return

            need_sc_restart = self.set_swing_cancellation(False)

            for direction in [[0,0,1], [0,0,-1]]:
                await self.tension_and_wait()
                # wait for some new obs
                await asyncio.sleep(0.5)
                lengths = np.linalg.norm(self.pe.anchor_points - self.pe.visual_pos, axis=1)
                await self.sendReferenceLengths(lengths)
                await asyncio.sleep(0.25)
                # move in direction for short time
                await self.move_direction_speed(direction, 0.05, downward_bias=0)
                await asyncio.sleep(0.25)
                self.slow_stop_all_spools()

            if need_sc_restart:
                self.set_swing_cancellation(True)

        except asyncio.CancelledError:
            raise

    async def ensure_pole_upright(self):
        """Raise the gripper until its pole is within 10 degrees of vertical.

        Raising the gripper tends to pull a horizontal pole upright, and vertical
        motion is usable even before calibration. Move slowly upward until the
        accelerometer reports the pole is within tolerance, but give up after
        1 meter of travel or 6 seconds since further lifting could break things.
        On giving up, stops the spools, shows a popup, and raises RuntimeError.
        """
        VERTICAL_TOLERANCE_DEG = 10.0
        MAX_LIFT_M = 1.0
        MAX_LIFT_S = 10.0
        MAX_LIFT_GRACE_S = 1.0  # ignore the distance limit briefly so an early gant_pos jump can't trip it
        vertical_start_pos = self.pe.gant_pos
        vertical_start_time = time.time()
        while True:
            angle = await self.gripper_client.query_angle_from_vertical()
            if angle is None:
                # No reply means the gripper is running an older server
                logger.warning('Gripper did not answer angle_from_vertical query (server likely out of date); skipping ensure_pole_upright')
                self.slow_stop_all_spools()
                return
            if angle <= VERTICAL_TOLERANCE_DEG:
                break
            elapsed = time.time() - vertical_start_time
            if ((elapsed >= MAX_LIFT_GRACE_S
                    and np.linalg.norm(self.pe.gant_pos - vertical_start_pos) >= MAX_LIFT_M)
                    or elapsed >= MAX_LIFT_S):
                self.slow_stop_all_spools()
                self.send_ui(pop_message=telemetry.Popup(
                    message='Could not achive a vertical pose to begin calibration. manually position the gripper in the center of the room hovering just over the floor and restart calibration.'
                ))
                raise RuntimeError('Could not achieve a vertical gripper pose to begin calibration')
            await self.move_direction_speed([0, 0, 1], 0.1, downward_bias=0)
            await asyncio.sleep(0.25)
        self.slow_stop_all_spools()

    # A pass's fitness is its optimize_arp_anchors fit_info['total_cost']: the same weighted
    # sum-of-squares residual cost the optimizer itself minimizes (see multi_card_residuals in
    # eyelet_calibration.py), computed identically every call so it's directly comparable across
    # attempts. Lower is better. Warn if a pass costs more than this fraction above the best
    # (lowest) cost ever recorded for that pass name.
    CALIBRATION_FITNESS_REGRESSION_TOLERANCE = 0.15

    def _flush_calibration_diagnostics(self):
        """Write self._calibration_diagnostics to calibration_diagnostics.pkl.

        Writes to a temp file and renames over the target so a crash or hard kill mid-write
        can never corrupt/truncate the previously-flushed passes still sitting in the file.
        """
        tmp_path = 'calibration_diagnostics.pkl.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(self._calibration_diagnostics, f)
        os.replace(tmp_path, 'calibration_diagnostics.pkl')

    def _record_calibration_diagnostics(self, pass_name, func, args, kwargs=None):
        """Append one optimize_arp_anchors call's bound arguments to the running
        calibration diagnostics list and flush the whole list to a single pickle file.

        Writing on every call (rather than only at the end) means a hang or crash
        partway through calibration still leaves everything recorded so far on disk.
        Bind args to the function's actual parameter names so the pickle is readable
        offline without cross-referencing the call site.
        """
        bound = inspect.signature(func).bind(*args, **(kwargs or {}))
        bound.apply_defaults()
        self._calibration_diagnostics.append({
            'pass': pass_name,
            'timestamp': time.time(),
            'args': dict(bound.arguments),
        })
        self._flush_calibration_diagnostics()
        logger.info(
            f'Saved calibration diagnostics for {pass_name} '
            f'({len(self._calibration_diagnostics)} pass(es) so far) to calibration_diagnostics.pkl'
        )

    def _record_calibration_abort(self, reason, error=None):
        """Append why a calibration run stopped, and the step it was on, to the diagnostics
        pickle.

        The passes already in the file say what the optimizer was given; they cannot say that
        the run never reached the next one, or why. Offline that difference is the whole
        question: a file holding two passes reads the same whether the third was skipped for
        want of gripper card views or the run was killed on its way there.

        Carries 'abort' rather than 'pass', so a reader can tell the two apart."""
        if not self.rec_diagnostics:
            return
        percent, step = self._calibration_step
        self._calibration_diagnostics.append({
            'abort': reason,
            'step': step,
            'percent_complete': percent,
            'timestamp': time.time(),
            'error': error,
        })
        self._flush_calibration_diagnostics()
        logger.info(f'Saved calibration abort ({reason}) during step "{step}" '
                    f'to calibration_diagnostics.pkl')

    def _record_calibration_fitness(self, pass_name, fit_info):
        """Attach fit_info to this pass's diagnostics record, and compare its total_cost
        against the best (lowest) cost ever recorded for this pass name, so a regression is
        flagged live instead of only being visible from an offline pickle analysis.

        History of the best/last cost per pass persists across runs in
        calibration_fitness_history.json (survives process restarts, unlike
        self._calibration_diagnostics which is cleared at the start of every run).
        """
        for record in reversed(self._calibration_diagnostics):
            if record['pass'] == pass_name:
                record['fit_info'] = fit_info
                break
        self._flush_calibration_diagnostics()

        history_path = 'calibration_fitness_history.json'
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = {}

        cost = fit_info['total_cost']
        now = time.time()
        entry = history.get(pass_name)

        if entry is not None and cost > entry['best_cost'] * (1 + self.CALIBRATION_FITNESS_REGRESSION_TOLERANCE):
            msg = (
                f"Calibration {pass_name} fitness regressed: cost={cost:.4f} vs best-known "
                f"{entry['best_cost']:.4f} (recorded {time.ctime(entry['best_timestamp'])})"
            )
            logger.warning(msg)
        else:
            best_desc = f"best-known {entry['best_cost']:.4f}" if entry else "first recorded attempt"
            logger.info(f'Calibration {pass_name} fitness: cost={cost:.4f} ({best_desc})')

        if entry is None or cost < entry['best_cost']:
            entry = {**(entry or {}), 'best_cost': cost, 'best_timestamp': now}
        entry['last_cost'] = cost
        entry['last_timestamp'] = now
        history[pass_name] = entry

        tmp_path = history_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(history, f, indent=2)
        os.replace(tmp_path, history_path)

    async def full_auto_calibration(self):
        """Automatically determine anchor poses and zero angles
        This is a motion task"""
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=0.0,
            name="Calibration",
            current_action="Observing markers",
        ))
        finger_task = None
        DETECTION_WAIT_S = 0.1 # how often to recount the origin card detections
        # how far above the floor to hold the gripper fingertips at the diamond's bottom point
        floor_clearance_m = self.diamond_size[2]
        self.tension_over_limit = False  # clear any stale trip from a previous run
        self.gantry_marker_fault = None
        # re-arm the marker monitor, so a fault that is still standing after an aborted run
        # aborts this one too rather than being counted as already reported
        self._gantry_marker_warned.clear()
        self._marker_may_be_hidden = True
        self._calibration_step = (0.0, 'Starting')
        if self.rec_diagnostics:
            self._calibration_diagnostics = []  # clear any stale data from a previous run
        try:
            if len(self.anchors) < N_ANCHORS:
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=100.0,
                    name="Calibration",
                    current_action='Cannot run full calibration until all anchors are connected',
                ))
                self._marker_may_be_hidden = False
                return
            elif len(self.anchors) > N_ANCHORS:
                logger.warning(f'Too many anchors found \n{self.anchors}')
            await self.set_torque(True)
            # collect observations of origin card aruco marker to get initial guess of anchor poses.
            #   origin pose detections are actually always stored by all connected clients,
            #   it is only necessary to ensure enough have been collected from each client and average them.
            for a in self.anchors.values():
                a.save_raw = True
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=0.0,
                name="Calibration",
                current_action="Observing markers",
            ))
            ORIGIN_VISIBILITY_TIMEOUT_S = 30.0 # give up if some anchor camera never sees the origin card
            detecting_start = time.time()
            seeing = None
            for client in self.anchors.values():
                client.origin_poses['origin'].clear()
            while True:
                num_o_dets = [len(client.origin_poses['origin']) for client in self.anchors.values()]
                # only anchor nums which see the origin card
                now_seeing = [anum for anum, count in enumerate(num_o_dets) if count > 0]
                if now_seeing != seeing:
                    seeing = now_seeing
                    self.send_ui(visibility_states=telemetry.VisibilityStates(anchors_seeing_origin_card=seeing))
                if num_o_dets and min(num_o_dets) >= max_origin_detections:
                    break
                logger.debug(f'Waiting for enough origin card detections from every anchor camera {num_o_dets}')

                if time.time() - detecting_start >= ORIGIN_VISIBILITY_TIMEOUT_S:
                    self.slow_stop_all_spools()
                    self.send_ui(pop_message=telemetry.Popup(
                        message="The origin card must be placed at a location visible to both cameras. "
                                "If there is no overlap in the camera's views of the room. "
                                "either mount them closer, or install different camera tilt adapters."
                    ))
                    raise RuntimeError('Origin card not visible to all anchor cameras within timeout')

                await asyncio.sleep(DETECTION_WAIT_S)
            logger.info(f'Collected enough observations {num_o_dets} in '
                        f'{time.time() - detecting_start:.1f}s')
            self.send_ui(visibility_states=telemetry.VisibilityStates(anchors_seeing_origin_card=list(
                [anum for anum, count in enumerate(num_o_dets) if count > 0] # only anchor nums which see the origin card
            )))

            raw_obs = await self.snapshot_tag_observations_still()

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=1.0,
                name="Calibration",
                current_action="Running 1st optimization pass",
            ))
            r = await self.flush_tele_buffer()

            # Measure each camera's real pitch off the cards it just saw, rather than trusting
            # the adapter's nominal angle. A tilt that is wrong shows up as an anchor fitted
            # off plumb, which the anchor_tilt term then fights for the rest of calibration,
            # and the room it settles on drives the moves the later passes are surveyed from.
            # Persisted because arp_anchor_client projects live frames through the configured
            # value: fitting the poses under one tilt and viewing through another puts the two
            # back out of step.
            configured_tilts = (self.config.anchors[0].indirect_line.cam_tilt,
                                self.config.anchors[1].indirect_line.cam_tilt)
            tilts = estimate_cam_tilts(raw_obs, configured_tilts)
            if tilts != configured_tilts:
                logger.info(f'Measured camera tilts {tilts} (configured {configured_tilts})')
                for anchor_num, tilt in enumerate(tilts):
                    self.config.anchors[anchor_num].indirect_line.cam_tilt = tilt
                    self.anchors[anchor_num].updatePoseAndEye()
                save_config(self.config, self.config_path)
                self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                    poses=[a.pose for a in self.config.anchors],
                    eyelets=[a.indirect_line.eyelet_pos for a in self.config.anchors],
                    tilt=[a.indirect_line.cam_tilt for a in self.config.anchors],
                    swing_latency=self.config.swing_latency,
                    pole_type=self.config.gripper.pole_type,
                ))

            # determine position of two anchors visually and guess at external eyelets.
            pass1_args = (raw_obs, None, None, None, None, tilts)
            pass1_kwargs = {'diamond_size': self.diamond_size, 'gantry_marker_inv': self.gantry_april_inv}
            if self.rec_diagnostics:
                self._record_calibration_diagnostics('anchors_pass1', optimize_arp_anchors, pass1_args, pass1_kwargs)
            async_result = self.pool.apply_async(optimize_arp_anchors, pass1_args, pass1_kwargs)
            anchor_poses, eyelet_positions, floor_z, fit_info = async_result.get(timeout=30)
            if self.rec_diagnostics:
                self._record_calibration_fitness('anchors_pass1', fit_info)
            logger.info(f'Obtained result from optimize_arp_anchors anchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')

            # The room's yaw is a gauge freedom of the residuals, so later passes are free to
            # spin the whole solution about z and silently invalidate the room-spin constant and
            # swing cancellation. Hold every later pass at the orientation pass 1 landed on.
            yaw_reference = anchor_poses

            self.save_poses_arp(anchor_poses, eyelet_positions)
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=1.0,
                name="Calibration",
                current_action="Moving to safe position",
            ))

            # Tighten lines
            await self.half_auto_calibration()

            # This might be the first time the lines are tightened after connecting the carabiners, and the gripper pole could be horizontal.
            # even if predictable motion is not yet possible do some basic checks to ensure the gripper is veritcal and in the middle of the room
            await self.ensure_pole_upright()
            self._marker_may_be_hidden = False

            await self.move_direction_speed([0, 0, 1], 0.1, downward_bias=0)
            await asyncio.sleep(0.5)
            self.slow_stop_all_spools()

            # Top of work area, from the two anchor-side pull points (indices 0 and 2) only.
            upper_z = float(np.mean(self.pe.anchor_points[[0, 2], 2]))

            # even without full calibration we should be able to make crude movements. go to the center
            # of the room just above the floor. This is the diamond's bottom point, so place the gantry
            # such that the gripper fingertips (self.pole[2] + GRIPPER_FINGER_LEN_M below the gantry) sit
            # floor_clearance_m above the floor.
            gant_z = min(
                upper_z-0.1, # stay at least 0.1 under the top of the work area
                self.pole[2] + GRIPPER_FINGER_LEN_M + floor_clearance_m - floor_z # mind that the origin card might be on a bed or a table, with the origin under the bed
            )
            await self.seek_goal(np.array([0, 0, gant_z]))

            # measure finger contact and reset wrist while doing the diamond pattern to save time.
            async def wait_then_finger():
                await asyncio.sleep(10)
                await self.calibrate_finger_servo()
                await self.gripper_client.send_commands({'reset_wrist': None})
            finger_task = asyncio.create_task(wait_then_finger())

            # collect length_change_data data to estimate eyelets better
            diamond_data, line_deltas = await self.collect_arp_anchor_eyelet_experiment_data(anchor_poses, upper_z)
            # stop saving raw poses
            for a in self.anchors.values():
                a.save_raw = False

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=22.0,
                name="Calibration",
                current_action="Running 2nd optimization pass",
            ))
            r = await self.flush_tele_buffer()

            pass2_args = (raw_obs, diamond_data, None, None, line_deltas, tilts)
            pass2_kwargs = {'diamond_size': self.diamond_size, 'yaw_reference': yaw_reference,
                            'gantry_marker_inv': self.gantry_april_inv}
            if self.rec_diagnostics:
                self._record_calibration_diagnostics('anchors_pass2', optimize_arp_anchors, pass2_args, pass2_kwargs)
            async_result = self.pool.apply_async(optimize_arp_anchors, pass2_args, pass2_kwargs)
            anchor_poses, eyelet_positions, floor_z, fit_info = async_result.get(timeout=60)
            if self.rec_diagnostics:
                self._record_calibration_fitness('anchors_pass2', fit_info)
            logger.info(f'Obtained result from optimize_arp_anchors anchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')

            self.save_poses_arp(anchor_poses, eyelet_positions)
            self.config.calibrated_status = common.CalibratedStatus.POSES_ONLY

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=24.0,
                name="Calibration",
                current_action="Tensioning lines and Locating Gripper",
            ))
            r = await self.flush_tele_buffer()
            await self.half_auto_calibration()

            # open grip enough that we can see an unobstructed view from the palm camera
            r = await finger_task
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': -40}))

            # move over the origin card
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=27.0,
                name="Calibration",
                current_action="Finding the origin card",
            ))
            # The optimizer's frame puts the floor at z=0, so the card sits at -floor_z: its own
            # height if it is up on a bed or table. Searching from just above it beats flying to a
            # fixed height the ceiling may not have room for.
            gant_z = await self.find_origin_card(card_pos=np.array([0.0, 0.0, -floor_z]),
                                                 upper_z=upper_z)

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=29.0,
                name="Calibration",
                current_action="Measuring spin. Gripper camera must see origin card to complete this step.",
            ))
            # there should be some swing when we get there. 
            r = await self.half_auto_calibration()
            r = await self._center_card_in_view('origin')

            # roomspin
            r = await self.calibrate_spin(reset_wrist_first=False) # already did that during diamond to save time
            
            await self.half_auto_calibration()

            # Tune swing_latency by inducing swings and finding the value that damps
            # them best. Requires a connected gripper (IMU-driven swing model).
            if self.gripper_client is not None:
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=34.0,
                    name="Calibration",
                    current_action="Tuning swing cancellation",
                ))
                # Swing tuning runs lower than the spin measurement, and the rangefinder says
                # how much lower. A fixed drop from gant_z cannot: that height is chosen so the
                # camera can see the origin card, and the card may be up on a bed or a table, so
                # the same drop leaves a different gap over whatever is really underneath. This
                # descends until the gap itself reads right, from wherever the spin step ended.
                # More steps than the survey's trim takes, because that one starts near its
                # target and this one starts at whatever height found the card.
                SWING_MEASURE_RANGE_M = 0.25
                reached = await self.trim_altitude_to_range(
                    SWING_MEASURE_RANGE_M, max_steps=6, ceiling_z=upper_z - 0.1)
                if reached is None:
                    logger.warning('Swing tuning: no rangefinder reading; measuring at the '
                                   'spin-measurement height instead')
                await self.calibrate_swing_latency(progress_range=(30.0, 61.0))

            await self.half_auto_calibration()

            # Refine the pull-point geometry with close-range gripper-camera views of the
            # calibration cards. The cards are still in place at this point (they are only
            # removed once calibration reports complete), and tension reg + swing cancellation
            # keep all four lines taut while hovering, so the measured (gantry, line-length)
            # pairs are a strong constraint on the anchors and eyelets.
            if (self.config.anchor_type == common.AnchorType.ARPEGGIO
                    and self.gripper_client is not None
                    and self.feature_supported("gripper_card_survey")):
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=61.0,
                    name="Calibration",
                    current_action="Refining geometry with gripper card views",
                ))
                gripper_obs = await self.collect_gripper_card_observations(progress_range=(61.0, 98.0))
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=98.0,
                    name="Calibration",
                    current_action="Running 3rd optimization pass",
                ))
                r = await self.flush_tele_buffer()

                # move over the origin card
                await self.seek_goal(np.array([0,0,gant_z]), head_turn=False)

                # Come to a full stop before the refinement. Swing cancellation goes off first
                # because it re-issues velocities on its own key and would drive the spools again
                # right after the stop. It must also be off before the refined geometry is
                # applied: the new eyelets change the velocity->line-speed mapping it depends on,
                # so a bad refinement could make it pump. Turned back on below only if it damps.
                sc_was_running = self.set_swing_cancellation(False)
                self.slow_stop_all_spools()

                # Require a reading from all four cards (origin + 3 cal_assist). With fewer hovers the
                # gripper term has too few length-delta pairs to pin the two far eyelets, and the
                # under-constrained refinement distorts a good rectangular layout into a diamond.
                REQUIRED_GRIPPER_CARDS = 4
                if len(gripper_obs) >= REQUIRED_GRIPPER_CARDS:
                    # Anchors are free here so the gripper's close-range views can refine them
                    # too. The room's absolute rotation about z is unobservable to the
                    # distance-based constraints, so the gripper term (whose measured vectors
                    # live in the real room frame) could otherwise spin the whole solution about
                    # z, invalidating the room-spin constant from the spin step and flipping
                    # swing cancellation from damping to pumping. yaw_reference holds that one
                    # degree of freedom at the orientation pass 1 established.
                    # optimize_arp_anchors returns poses shifted so z=0 is the floor, but it solves
                    # in a frame with the origin card at z=0. Undo that shift on the way back in,
                    # or the warm start (and the eyelet_reg target built from it) sits floor_z off
                    # in z - which is the whole height of the origin card's perch, not a rounding
                    # error, when the card is on a bed or table.
                    warm_anchors = np.array(anchor_poses, dtype=float)
                    warm_anchors[:, 1, 2] += floor_z
                    warm_eyelets = np.array(eyelet_positions, dtype=float)
                    warm_eyelets[:, 2] += floor_z

                    args = (raw_obs, diamond_data, warm_eyelets, None, line_deltas, tilts, gripper_obs)
                    pass3_kwargs = {
                        'diamond_size': self.diamond_size,
                        'yaw_reference': yaw_reference,
                        'initial_anchor_guesses': warm_anchors,
                        'gantry_marker_inv': self.gantry_april_inv,
                    }
                    if self.rec_diagnostics:
                        self._record_calibration_diagnostics('anchors_pass3', optimize_arp_anchors, args, pass3_kwargs)
                    async_result = self.pool.apply_async(optimize_arp_anchors, args, pass3_kwargs)
                    refined_anchors, refined_eyelets, refined_floor_z, fit_info = async_result.get(timeout=60)
                    if self.rec_diagnostics:
                        self._record_calibration_fitness('anchors_pass3', fit_info)
                    # This pass carries far more residuals than the one before it, so its total
                    # cost falls even when it reaches that by leaning the anchors and spreading
                    # the pull points rather than by finding a better room. Weigh the structural
                    # terms on their own and keep pass 2's geometry when they have been spent.
                    plausible, reason = refinement_is_plausible(fit_info)
                    if refined_anchors is None:
                        logger.warning('Gripper-card refinement optimization failed; keeping previous geometry')
                    elif not plausible:
                        logger.warning(f'Rejected gripper-card refinement: it fit the card survey by '
                                       f'deforming the room ({reason}). Keeping previous geometry.')
                    else:
                        anchor_poses, eyelet_positions, floor_z = refined_anchors, refined_eyelets, refined_floor_z
                        logger.info(f'Refined with gripper card views ({reason}):\nanchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')
                        self.save_poses_arp(anchor_poses, eyelet_positions)

                    # Re-enable swing cancellation only if it still damps a test swing with the new
                    # geometry. _measure_swing_residual induces a swing, runs cancellation, and
                    # reports the leftover swing (or the safety cap / no reading if it pumped or
                    # drifted). Anything that isn't a clearly-damped low residual leaves it OFF.
                    self.send_ui(operation_progress=telemetry.OperationProgress(
                        percent_complete=99.0,
                        name="Calibration",
                        current_action="Verifying swing cancellation is safe",
                    ))
                    # This is the last word on the verdict, taken against the refined geometry,
                    # so it overwrites what the latency sweep earlier in this run concluded
                    # against the old one.
                    await self._verify_swing_cancellation('with refined geometry',
                                                          'after calibration refinement')
                else:
                    logger.warning(f'Only {len(gripper_obs)} of {REQUIRED_GRIPPER_CARDS} gripper card observations; need all four to refine. Skipping 3rd pass.')
                    # geometry is unchanged, so no damping re-test is needed to restore it, and
                    # nothing was measured, so the stored verdict stays whatever it already was
                    self.set_swing_cancellation(sc_was_running)

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=100.0,
                name="Calibration",
                current_action="Calibration completed. Sanity check anchor positions before moving. Cards can be removed from the floor. Parking location must be re-recorded.",
            ))
            r = await self.flush_tele_buffer()

        except asyncio.CancelledError:
            self._calibration_abort_cleanup()
            if finger_task is not None:
                finger_task.cancel()
                await finger_task
            # read and clear both, so whichever did not cause this abort cannot go on to
            # mislabel the next one
            marker_fault, self.gantry_marker_fault = self.gantry_marker_fault, None
            tension_trip, self.tension_over_limit = self.tension_over_limit, False
            if marker_fault is not None:
                # a bad marker makes the lines go where the geometry isn't, so it can trip the
                # tension limit on its way out; it is the reason, and the trip is the symptom
                current_action = f'Aborted: {marker_fault}'
            elif tension_trip:
                current_action = "Aborted: line tension exceeded the safe limit"
            else:
                current_action = "Cancelled by user"
            # before the send_ui below, which reports the abort itself as the current step
            self._record_calibration_abort(current_action)
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=100.0,
                name="Calibration",
                current_action=current_action,
            ))
            raise
        except Exception as e:
            self._calibration_abort_cleanup()
            if finger_task is not None:
                finger_task.cancel()
            self._record_calibration_abort(f'Failed: {e!r}', error=traceback.format_exc())
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=100.0,
                name="Calibration",
                current_action='Calibration failed, see motion controller console',
            ))
            raise

    def _calibration_abort_cleanup(self):
        """On any calibration abort (safety tension trip, user cancel, or error) stop all spools
        and disable swing cancellation so the gripper does not keep moving."""
        self.slow_stop_all_spools()
        self.set_swing_cancellation(False)
        self._marker_may_be_hidden = False

    async def calibrate_spin(self, reset_wrist_first=True):
        """Calibration of the relationship between the wrist and the room frame of reference.
        Must be done over the origin card.
        """
        client = self.gripper_client
        if client is None or client.last_output_frame is None:
            logger.warning('Cannot calibrate the relationship between gripper zero angle and camera if gripper camera is offline!')
            return None

        # record the z rotation of the gantry card from the perspective of the gripper camera,
        # with no existing z rotation term applied
        client.calibrating_room_spin = True
        try:
            # measurement must be taken at the wrist's zero point
            center_angle = 540
            if reset_wrist_first:
                asyncio.create_task(client.send_commands({'reset_wrist': None}))
                await asyncio.sleep(10)
            # wait till within 1 degree of target
            actual_wrist = 100
            end_time = time.time() + 2
            logger.info(f'Moved wrist to {center_angle}, waiting to reach position')
            while abs(actual_wrist - center_angle) > 2.0 and time.time() < end_time:
                await asyncio.sleep(0.2)
                actual_wrist = self.datastore.winch_line_record.getLast()[1]
            logger.info(f'Actual wrist position = {actual_wrist}')

            # detect origin card
            try:
                await asyncio.sleep(0.1)
                origin_card_pose = [None]
                def special_handle_det(timestamp, detections):
                    for d in detections:
                        if d['n'] == 'origin':
                            # a pose of the origin card in the frame of reference of the gripper cam.
                            origin_card_pose[0] = d['p']
                end_time = time.time() + 10
                logger.info('Collecting observations of origin card from gripper cam')
                while origin_card_pose[0] is None and time.time() < end_time:
                    async_result = self.pool.apply_async(
                        locate_markers,
                        (client.last_output_frame, self.config.camera_cal_wide),
                        callback=partial(special_handle_det, time.time()))
                    detections = async_result.get(timeout=5)
            except Exception as e:
                logger.exception(e)
                raise
            if origin_card_pose[0] is None:
                raise RuntimeError("Gripper camera was unable to make any observations of the origin card.")

            euler_rot = Rotation.from_rotvec(origin_card_pose[0][0]).as_euler('zyx')
            logger.info(f'Euler rotation of origin card relative to gripper camera {euler_rot}')
            roomspin = euler_rot[0]
            self.config.gripper.frame_room_spin = roomspin
            self.config.calibrated_status = common.CalibratedStatus.FULLY_CALIBRATED
            save_config(self.config, self.config_path)
        finally:
            # left set, the client would go on leaving the room spin out of every heading
            client.calibrating_room_spin = False

    async def recover_spin(self):
        """Re-measure frame_room_spin without the origin card. This is a motion task.

        get_spin is the wrist angle plus frame_room_spin, so a wrong frame_room_spin turns
        every camera direction the robot reports by the same angle. Moving the gantry and
        watching the floor slide the other way in the gripper camera measures that angle
        directly: the slide, turned into the room frame with the spin as it stands, points
        along the gantry's actual motion rotated by the error.

        The gantry flies a slow circle rather than straight legs, since a corner kicks the
        pole into a swing and a steady turn does not, and a circle takes every heading so
        nothing that favours one direction survives the average. It goes round once each
        way: a lag between frames and positions reads as a heading error of opposite sign
        on the two loops and cancels, where a spin error reads the same on both.

        Wants floor with some texture under it, the laser reading 0.25 to 2m, and room for a
        40cm circle beside the gantry, on the side towards the middle of the room. Swing
        cancellation is switched off first, since with the spin wrong it pumps swing rather
        than damping it. Once the spin is corrected it is re-tested as full calibration does,
        which records swing_cancellation_verified and turns it on only if it damps.
        """
        RADIUS_M = 0.20
        SPEED_MPS = 0.03
        RAMP_S = 6.0                   # easing into and out of the circle
        SETTLE_S = 3.0
        RANGE_M = (0.25, 2.0)          # laser range the floor texture is usable over
        MIN_CONFIDENCE = 0.5           # share of feature matches that must agree
        MIN_MOVE_M = 0.004             # a frame pair the gantry barely moved over says little
        OUTLIER_RAD = np.radians(30)   # pairs this far from their loop's consensus are dropped
        MIN_PAIRS = 10                 # per loop
        MAX_LOOP_DISAGREEMENT_RAD = np.radians(20)

        client = self.gripper_client
        if client is None or client.last_output_frame is None:
            logger.warning('Spin recovery needs a connected gripper with its camera running')
            return None
        laser = self.laser_range()
        if laser is None or not RANGE_M[0] <= laser <= RANGE_M[1]:
            self.send_ui(pop_message=telemetry.Popup(
                message=f'Spin recovery needs the gripper {RANGE_M[0]:.2f} to {RANGE_M[1]:.1f}m '
                        f'above the floor; the laser reads {laser}.'))
            return None

        start = self.gantry_position()
        room_middle = np.mean(self.anchor_points()[:, :2], axis=0)
        inward = room_middle - start[:2]
        inward = inward / np.linalg.norm(inward) if np.linalg.norm(inward) > 1e-3 else np.array([1.0, 0.0])
        center = start[:2] + inward * RADIUS_M
        rim = [center + RADIUS_M * np.array([np.cos(a), np.sin(a)])
               for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        if not all(self.inside_work_area_2d(p) for p in rim):
            self.send_ui(pop_message=telemetry.Popup(
                message='Not enough room for spin recovery here; move the gantry further '
                        'into the room and try again.'))
            return None

        before = self.config.gripper.frame_room_spin
        try:
            self.set_swing_cancellation(False)
            # fully open is fully retracted, out of the camera's view
            await self.settle_fingers(-90)
            await asyncio.sleep(SETTLE_S)

            loop_errors = []
            for turn in (1, -1):
                track, pairs = await self._fly_spin_circle(start, center, RADIUS_M, turn,
                                                           SPEED_MPS, RAMP_S)
                errors = self._spin_errors(track, pairs, MIN_CONFIDENCE, MIN_MOVE_M)
                if len(errors) < MIN_PAIRS:
                    raise RuntimeError(f'only {len(errors)} usable frame pairs going round '
                                       f'{"counterclockwise" if turn > 0 else "clockwise"}; '
                                       f'the floor may have too little texture')
                consensus, _ = mean_heading_error(errors)
                kept = [e for e in errors if abs(wrap_angle(e - consensus)) < OUTLIER_RAD]
                mean, spread = mean_heading_error(kept)
                logger.info(f'Spin recovery, {"counterclockwise" if turn > 0 else "clockwise"}: '
                            f'error {np.degrees(mean):+.1f} deg from {len(kept)} of '
                            f'{len(errors)} frame pairs, within {np.degrees(spread):.1f} deg')
                loop_errors.append(mean)
                self.slow_stop_all_spools()
                await asyncio.sleep(SETTLE_S)

            disagreement = abs(wrap_angle(loop_errors[0] - loop_errors[1]))
            if disagreement > MAX_LOOP_DISAGREEMENT_RAD:
                raise RuntimeError(f'the two loops disagree by {np.degrees(disagreement):.0f} '
                                   f'degrees, so neither can be trusted')
            correction, _ = mean_heading_error(loop_errors)
        except RuntimeError as e:
            logger.warning(f'Spin recovery failed: {e}')
            self.send_ui(pop_message=telemetry.Popup(message=f'Spin recovery failed: {e}'))
            return None
        finally:
            await self.move_direction_speed(np.zeros(3), 0, key=SPIN_VELOCITY_KEY)
            self.slow_stop_all_spools()

        self.config.gripper.frame_room_spin = wrap_angle(before + correction)
        save_config(self.config, self.config_path)
        logger.info(f'Spin recovered: frame_room_spin {np.degrees(before):.1f} -> '
                    f'{np.degrees(self.config.gripper.frame_room_spin):.1f} deg')

        # the same test full calibration ends with, since the spin is what swing
        # cancellation steers by
        verified = await self._verify_swing_cancellation('with the recovered spin',
                                                         'after spin recovery')
        self.send_ui(pop_message=telemetry.Popup(
            message=f'Spin corrected by {np.degrees(correction):+.1f} degrees. '
                    + ('Swing cancellation damps with it and has been turned on. ' if verified else '')
                    + 'Running this again should find almost nothing left to correct.'))
        return correction

    async def _fly_spin_circle(self, start, center, radius, turn, speed, ramp):
        """Fly once round a horizontal circle through start, easing in and out, while pairing
        up gripper frames. turn is 1 for counterclockwise, -1 for clockwise.

        Returns (track, pairs): track is (time, gantry position) every control step, and
        pairs is (earlier capture time, later capture time, dx, dy, confidence, inliers,
        laser range) for consecutive frames, the slide measured by image_shift.
        """
        LOOP_S = 0.1
        FRAME_GAP_S = 0.4              # enough travel between frames for a measurable slide
        TRACK_GAIN = 0.5               # 1/s pulling the gantry back onto the circle

        track, pairs = [], []

        async def pair_frames():
            last_ts, last_gray = await self.gripper_capture(timeout=2.0)
            if last_gray is not None:
                last_gray = cv2.cvtColor(last_gray, cv2.COLOR_RGB2GRAY)
            while True:
                await asyncio.sleep(FRAME_GAP_S)
                ts, frame = await self.gripper_capture(after=last_ts or 0, timeout=2.0)
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if last_gray is not None:
                    dx, dy, confidence, inliers = await asyncio.to_thread(image_shift, last_gray, gray)
                    pairs.append((last_ts, ts, dx, dy, confidence, inliers, self.laser_range()))
                last_ts, last_gray = ts, gray

        start_angle = np.arctan2(start[1] - center[1], start[0] - center[0])
        total, ramp = eased_move_time(2 * np.pi * radius, speed, ramp)
        sampler = asyncio.create_task(pair_frames())
        travelled = 0.0
        began = last = time.monotonic()
        try:
            while True:
                now = time.monotonic()
                elapsed = now - began
                if elapsed >= total:
                    break
                current = speed * eased_speed(elapsed, total, ramp)
                travelled += current * (now - last)
                last = now
                angle = start_angle + turn * travelled / radius
                on_circle = center + radius * np.array([np.cos(angle), np.sin(angle)])
                along = turn * np.array([-np.sin(angle), np.cos(angle)])
                here = self.gantry_position()
                track.append((time.time(), here))
                velocity = np.array([*(along * current + TRACK_GAIN * (on_circle - here[:2])),
                                     TRACK_GAIN * (start[2] - here[2])])
                await self.move_direction_speed(velocity, None, here, downward_bias=0,
                                                key=SPIN_VELOCITY_KEY)
                await asyncio.sleep(LOOP_S)
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
            await self.move_direction_speed(np.zeros(3), 0, key=SPIN_VELOCITY_KEY)
        return track, pairs

    def _spin_errors(self, track, pairs, min_confidence, min_move_m):
        """The heading error of each usable frame pair: the counterclockwise angle from the
        gantry's actual motion between the two captures to the motion the floor's slide
        implies, turned into the room frame with frame_room_spin as it stands."""
        if len(track) < 2:
            return []
        times = np.array([t for t, _ in track])
        positions = np.array([p for _, p in track])

        def position_at(t):
            return np.array([np.interp(t, times, positions[:, i]) for i in range(3)])

        intrinsics = self.gripper_camera_intrinsics()
        errors = []
        for t_a, t_b, dx, dy, confidence, inliers, laser in pairs:
            if confidence < min_confidence or laser is None:
                continue
            if t_a < times[0] or t_b > times[-1]:
                continue
            moved = (position_at(t_b) - position_at(t_a))[:2]
            if np.linalg.norm(moved) < min_move_m:
                continue
            # a camera that moves by t sees the scene slide by -f*t/depth
            camera_move = np.array([-dx * laser / intrinsics[0][0],
                                    -dy * laser / intrinsics[1][1], 0.0])
            implied = self.gripper_camera_to_room(camera_move, timestamp=t_b)[:2]
            errors.append(heading_error(moved, implied))
        return errors

    def record_drop_position(self):
        """Save where the gantry is standing now as the place to drop things, and route there.

        Stored in named_positions beside the hamper and the toybox, so everything that already
        knows how to fly to a named place can fly to this one.

        Those are positions of things on the floor, and a route hovers the dropoff height above
        them, so what is stored is the point the gantry is currently that height above. Saving
        while hovering where you want things to land therefore brings the gantry back to exactly
        where it was, which is the only behaviour that makes the button mean what it says. On a
        robot saved lower than the dropoff height that point is under the floor; nothing minds,
        since it is only ever used as something to hover over and to keep targets away from.

        Saving selects it as the route destination, because recording a place to drop things is
        how you say that is where things should go.
        """
        hover = tonp(self.config.pick_and_place.gantry_height_over_dropoff)
        gantry = self.gantry_position()
        self.set_named_position(DROP_POSITION_NAME, gantry - hover, save=False)
        logger.info(f'Drop position saved: gantry returns to {np.round(gantry, 3)}, '
                    f'stored as {np.round(gantry - hover, 3)} under it')
        # The To: field flipping to the drop position is the confirmation; no popup for a
        # button that did exactly what it says. set_route saves both.
        self.set_route(destination=common.RoutePoint.DROP_POSITION)

    async def settle_wrist_to_heading(self, angle_deg, tol=2.0, peak_dps=WRIST_EASE_DPS):
        """Settle the wrist on whichever of its equivalent angles faces the same way as
        angle_deg and is nearest where it is now.

        The camera only cares about the heading, which repeats every 360 degrees of the
        wrist's 0-1080 range, so reproducing a recorded angle exactly can mean winding the
        cable two turns to look at what is already in front of it.

        Walked there rather than commanded in one step: an absolute angle sent in one go has
        the servo start and stop at its own rate, and the gripper is a pendulum that gets
        flicked by both ends of that. The commanded angle eases in and out instead, and the
        wait at the end confirms the wrist arrived.
        """
        current = self.datastore.winch_line_record.getLast()[1]
        candidates = [angle_deg + 360.0 * k for k in (-2, -1, 0, 1, 2) if 0 <= angle_deg + 360.0 * k <= 1080]
        target = min(candidates or [angle_deg], key=lambda c: abs(c - current))

        travel = abs(target - current)
        if travel > WRIST_EASE_MIN_DEG:
            direction = np.sign(target - current)
            total, ramp = eased_move_time(travel, peak_dps, WRIST_RAMP_S)
            started = time.monotonic()
            commanded = current
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= total:
                    break
                commanded += (direction * peak_dps * eased_speed(elapsed, total, ramp)
                              * WRIST_STEP_S)
                await self.gripper_client.send_commands(
                    {'set_wrist_angle': float(clamp(commanded, min(current, target),
                                                    max(current, target)))})
                await asyncio.sleep(WRIST_STEP_S)
        return await self.settle_wrist(target, tol=tol)

    def fresh_gantry_sightings(self, window_s, after=None):
        """Room positions from anchor camera sightings of the gantry marker in the last window_s.

        Every row carries the time the frame was captured, so a short window is the
        difference between "the cameras can see the marker" and "a camera saw it once, a
        while ago". `after` additionally drops anything captured before a moment the caller
        cares about, such as the start of the move that was meant to bring it into view.
        """
        cutoff = time.time() - window_s
        if after is not None:
            cutoff = max(cutoff, after)
        return self.datastore.gantry_pos.deepCopy(cutoff=cutoff)[:, 2:]

    async def settle_visual_estimate(self, tol_m=0.05, window_s=1.0, min_sightings=3, timeout=15.0):
        """Wait until pe.visual_pos agrees with where the cameras are seeing the marker now.

        visual_pos is an exponential average that steps a tenth of the way toward each new
        sighting, so after a spell with no sightings at all - being parked, for instance - it
        takes a second or two of them before it stops describing where the gantry used to be.
        half_auto_calibration sets all four reference lengths from it, so calibrating before
        it has converged writes the stale position into every line.

        True once it has caught up, False if it never did.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            seen = self.fresh_gantry_sightings(window_s)
            if len(seen) >= min_sightings:
                error = float(np.linalg.norm(self.pe.visual_pos - np.mean(seen, axis=0)))
                if error < tol_m:
                    return True
            await asyncio.sleep(0.1)
        logger.warning('Visual position estimate never settled onto the live sightings')
        return False

    def on_service_state_change(self, 
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if 'cranebot' in name:
            if state_change is ServiceStateChange.Added:
                asyncio.create_task(self.add_service(zeroconf, service_type, name))
            if state_change is ServiceStateChange.Updated:
                asyncio.create_task(self.update_service(zeroconf, service_type, name))
            if state_change is ServiceStateChange.Removed:
                asyncio.create_task(self.remove_service(service_type, name))
            elif state_change is ServiceStateChange.Updated:
                pass

    async def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Records the information about a discovered service in the config"""
        info = AsyncServiceInfo(service_type, name)
        await info.async_request(zc, INFO_REQUEST_TIMEOUT_MS)
        if not info or info.server is None or info.server == '':
            return None;
        namesplit = name.split('.')
        kind = namesplit[1]
        key  = ".".join(namesplit[:3])

        address = socket.inet_ntoa(info.addresses[0])
        logger.debug(f'Service discovered: {namesplit}')

        is_arp_gripper = kind == arp_gripper_service_name
        is_arp_anchor = kind == arp_anchor_service_name

        # the number of lines is always four.
        # there are two arpeggio anchors, each controlling two lines.
        # anchor_num is 0 or 1. refrerences to anchor num that referred to a service, a camera or its pose
        # can still reference anchor num. references to anchor num that were referring to grommet positions
        # or line lengths and speeds, must now refer line numbers 0-3. sending a command to jog a spool or
        # set a line speed must be abstracted through a class that will send the message to the connected
        # server that manages that line.

        if is_arp_anchor:
            found_type = common.AnchorType.ARPEGGIO

            if self.config.anchor_type == common.AnchorType.UNSPECIFIED:
                # the first discovered anchor locks the config to an anchor type
                self.config.anchor_type = found_type
                # replace the default anchors in the config with two default arp anchors having unset addresses and service names
                self.config.anchors = default_arp_anchors() # imported from config_loader

            elif self.config.anchor_type != found_type:
                logger.warning(f'Ignored {found_type} anchor at {address} because config is locked to {self.config.anchor_type}')
                return

            # create a map from service name to anchor num
            anchor_num_map = {a.service_name: a.num for a in self.config.anchors if a.service_name is not None}
            if key in anchor_num_map:
                anchor_num = anchor_num_map[key]
            else:
                anchor_num = len(anchor_num_map)
                if anchor_num >= N_ANCHORS:
                    # Discovering more that four anchors could be a sign that another robot in the same network is turned on.
                    # We need a way to know that, but for now, you'll have to make sure only one is one at a time while discovering.
                    # After discovery, it should be ok to have more than one on at a time.
                    logger.warning(f"Discovered another {found_type} server on the network, but we already know of {N_ANCHORS} {key} {address}")
                    return None
            if self.config.anchors[anchor_num].address != address or self.config.anchors[anchor_num].port != info.port:
                self.config.anchors[anchor_num].num = anchor_num
                self.config.anchors[anchor_num].service_name = key
                self.config.anchors[anchor_num].address = address
                self.config.anchors[anchor_num].port = info.port
                save_config(self.config, self.config_path)

        elif is_arp_gripper:
            # a gripper has been discovered, assume it is ours only if we have never seen one before
            if self.config.gripper.service_name is None or self.config.gripper.service_name == "":
                self.config.gripper.service_name = key
                self.config.gripper.address = address
                self.config.gripper.port = info.port
                save_config(self.config, self.config_path)
                logger.info(f'Discovered gripper at "{address}" and adopted it as the gripper for this robot')
            elif address != self.config.gripper.address:
                logger.info(f'Discovered gripper at "{address}" and ignored it because ours is at {self.config.gripper.address}')

    async def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        # when zerconf has detected a change in address or port
        pass

    async def remove_service(self, service_type: str, name: str) -> None:
        """
        Finds if we have a client connected to this service. if so, ends the task if it is running, and deletes the client
        """
        namesplit = name.split('.')
        kind = namesplit[1]
        key  = ".".join(namesplit[:3])

        # only in this dict if we are connected to it.
        if key in self.bot_clients:
            # await self._handle_set_swing_cancellation(item=control.SetSwingCancellation(enabled=False, present='.'))
            client = self.bot_clients[key]
            await client.shutdown()
            if kind == arp_anchor_service_name:
                del self.anchors[client.anchor_num]
            elif kind == arp_gripper_service_name:
                self.gripper_client = None
                # persist the last observed named positions so they survive losing the gripper
                self.config.last_gantry_pos = fromnp(self.pe.gant_pos)
                save_config(self.config, self.config_path)
            del self.bot_clients[key]

    async def startup_action(self, event):
        """Wait for every component to turn up, then run the startup sequence.

        The waiting is kept out of the motion task: the sequence only becomes the motion
        task once it is about to move something, so a robot sitting waiting for an anchor is
        not occupying the slot that the stop button and every operator command cancel.
        """
        await event.wait()
        await self.invoke_motion_task(self.startup_sequence())

    async def startup_sequence(self):
        """Run the steps named by set_startup_sequence, in order. By default: unpark if the
        robot was left on the hook, work, then park again.

        A motion task, so the stop button ends it. Every step runs inside it, which is what
        makes one press of stop reach whichever step is actually running. Each step is held
        to its own maneuver's SafetyPolicy while it runs.
        This is a motion task.
        """
        try:
            for name in self.startup_sequence_names:
                step, safety, owner = self._startup_steps[name]
                logger.info(f'Startup step {name}')
                with self._running_as(owner, safety or SafetyPolicy()):
                    await step()
            # disconnect all components and set flag that they should not reconnect unless control input is received.
        except asyncio.CancelledError:
            logger.info('Startup sequence cancelled')
            raise
        finally:
            self.slow_stop_all_spools()

    async def keep_robot_connected(self):
        """
        Keep a connection open to every robot component known in the config
        components are keyed by their service name which is the first three components of info.name, eg
        123.cranebot-anchor-service.2ccf67bc3fc4
        """
        # If config is empty (first time startup) sleep until zeroconf discovers robot components
        while not config_has_any_address(self.config) and self.run_command_loop:
            await asyncio.sleep(0.5)

        ready = asyncio.Event()
        if self.auto_start:
            s_task = asyncio.create_task(self.startup_action(ready))

        while self.run_command_loop:
            # is everything up the way we want it to be? (N_ANCHORS anchors + 1 gripper)
            if len([b for b in self.bot_clients.values() if b.connected]) == N_ANCHORS + 1:
                ready.set()
                await asyncio.sleep(0.5)
                continue # All websocket connections are up.

            # make sure we have either a live connection to, or an ongoing attempt to connect to every component we know about.
            for cpt in [self.config.gripper, *self.config.anchors]:
                # assume only the common attributes between those two types
                key = cpt.service_name
                if key is None or cpt.address is None or cpt.port is None:
                    continue

                if key not in self.connection_tasks:
                    # Start a connection to this component. connect_component will also remove it when it completes regardless of success or failure.
                    self.connection_tasks[key] = asyncio.create_task(self.connect_component(key))

            await asyncio.sleep(0.5)

        if self.auto_start:
            s_task.cancel()
            r = await s_task

        for task in self.connection_tasks.values():
            task.cancel()
        result = await asyncio.gather(*self.connection_tasks.values())

    async def connect_component(self, service_name):
        """Connect to the component with the given name using the address stored in the config."""
        client = None
        try:
            name_component = service_name.split('.')[1]
        except IndexError:
            logger.warning(f'Invalid service name "{service_name}"')
            return

        is_arp_gripper = name_component == arp_gripper_service_name
        is_arp_anchor = name_component == arp_anchor_service_name

        if is_arp_gripper:
            client = ArpeggioGripperClient(self.config.gripper.address, self.config.gripper.port, self.datastore, self, self.pool, self.stat, self.pe, self.telemetry_env)
            self.gripper_client_connected.clear()
            client.connection_established_event = self.gripper_client_connected
            self.gripper_client = client
        elif is_arp_anchor:
            for a in self.config.anchors:
                if a.service_name != service_name:
                    continue
                client = ArpeggioAnchorClient(a.address, a.port, a.num, self.datastore, self, self.pool, self.stat, self.telemetry_env)
                client.connection_established_event = self.any_anchor_connected
                self.anchors[a.num] = client
        else:
            logger.warning(f"Don't know how to connect to {name_component}")

        if client:
            kind = 'gripper' if is_arp_gripper else 'anchor'
            client.on_connected = partial(self._announce_component, kind, client.anchor_num, True)
            client.on_disconnected = partial(self._announce_component, kind, client.anchor_num, False)
            self.bot_clients[service_name] = client
            # this function runs as long as the client is connected and returns true if the client was forced to disconnect abnormally
            abnormal_close = await client.startup()
            # build a friendly name and capture the address before the client is torn down
            if is_arp_anchor:
                display_name = f'Anchor {client.anchor_num}'
            elif is_arp_gripper:
                display_name = 'Gripper'
            else:
                display_name = name_component
            address = client.address
            # remove client
            r = await self.remove_service(None, service_name)
            # delete this task from the dict as it ends, so keep_robot_connected will try agian.
            # do this before the reconnect check below so a reconnect attempt can start.
            del self.connection_tasks[service_name]
            if abnormal_close:
                # don't alarm on a momentary drop (e.g. a firmware restart); only alert and
                # stop if the component is still gone after a brief grace period.
                asyncio.create_task(self._alert_if_not_reconnected(service_name, display_name, address))

    def _announce_component(self, kind, anchor_num, connected):
        """Tell every maneuver a component's websocket came up or went down."""
        for maneuver in self.maneuvers.values():
            try:
                if connected:
                    maneuver.on_component_connected(kind, anchor_num)
                else:
                    maneuver.on_component_disconnected(kind, anchor_num)
            except Exception:
                logger.exception(f'Maneuver {maneuver.name} failed to handle a component '
                                 f'{"connecting" if connected else "disconnecting"}')

    async def _alert_if_not_reconnected(self, service_name, display_name, address):
        """After a component disconnects abnormally, wait a couple seconds and only alert
        the user and stop the robot if it has not reconnected by then."""
        RECONNECT_GRACE_S = 2.0
        await asyncio.sleep(RECONNECT_GRACE_S)
        client = self.bot_clients.get(service_name)
        if client is not None and client.connected:
            logger.info(f'{display_name} reconnected within {RECONNECT_GRACE_S}s; suppressing lost-connection alert')
            return
        self.send_ui(pop_message=telemetry.Popup(
            message=f'Lost connection to {display_name} at {address}'
        ))
        await self.stop_all()

    def speed_limit(self):
        """Fastest total gantry velocity this robot will accept, in m/s.

        move_direction_speed enforces it by scaling the whole vector, which is worth
        knowing before asking for a large one: a descent commanded past the limit does not
        merely get shortened, it shrinks the lateral correction summed with it by the same
        factor. A caller that cares which component survives should budget against this.
        """
        return 0.45 if self.feature_supported("speed_0.45") else 0.35

    def feature_supported(self, feature_key):
        """Return True if every connected component runs an nf_robot version at or above the
        minimum required for the given feature (a key in VERSION_GATES). A component that has
        not reported a version (older firmware) is treated as not meeting the requirement."""
        required_v = parse_version(VERSION_GATES[feature_key])
        for client in self.bot_clients.values():
            if client.nf_robot_v is None:
                return False
            try:
                if parse_version(client.nf_robot_v) < required_v:
                    return False
            except InvalidVersion:
                logger.warning(f'component at {client.address} reported unparseable version {client.nf_robot_v!r}')
                return False
        return True

    def _handle_add_relay_creds(self, item: common.RelayCreds):
        """Store the id + key minted when this robot is bound to a control plane instance.

        Keyed by that instance's ws_protocol_and_host so the telemetry manager can look them
        up. Delivered over a control message (from the account bridge) once, so we persist it,
        then tell the manager to (re)connect with them right away.

        The issuing control plane is whichever one served the page the user bound from, which
        need not be the one this process was started against: binding is offered in LAN mode,
        where control_plane_host is only a default. So we key on the host the UI reports and
        fall back to our own default only when it reports none (older UI)."""
        host = normalize_control_plane_host(item.control_plane_host)
        if host is None:
            if item.control_plane_host:
                logger.warning(
                    f'Ignoring unusable control_plane_host {item.control_plane_host!r} in relay '
                    f'credentials; falling back to this robot\'s own control plane'
                )
            host = self.telemetry.control_plane_host
        logger.info(f'Storing relay credentials for {host} (robot id "{item.robot_id}")')
        # control_plane_host is deliberately not persisted: the map key already carries it,
        # and a second copy is just something that can disagree with it.
        self.config.relay_credentials[host] = common.RelayCreds(robot_id=item.robot_id, key=item.key)
        save_config(self.config, self.config_path)
        if host != self.telemetry.control_plane_host:
            # Bound against a different control plane than this run talks to. The creds are
            # saved and a run started against that plane will find them, but this session
            # stays unbound, so say so rather than leaving the user to wonder why the cloud
            # link never comes up.
            logger.warning(
                f'These credentials are for {host}, but this session is running against '
                f'{self.telemetry.control_plane_host}; restart with a matching --telemetry_env '
                f'to use them.'
            )
        self.telemetry.credentials_updated()

    def _handle_popup_ack(self, item: control.PopupAck):
        fut = self.pending_popup_acks.pop(item.id, None)
        if fut is not None and not fut.done():
            fut.set_result(item.button)

    async def send_popup_and_await_answer(self, message: str, buttons: list[str] | None = None, timeout: float | None = None) -> int | None:
        """
        Send a popup message to the UI and wait for the first PopupAck that answers it.
        Returns the index of the button clicked, or None if no UI answers within timeout.
        """
        popup_id = self._next_popup_id
        self._next_popup_id += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending_popup_acks[popup_id] = fut
        self.send_ui(pop_message=telemetry.Popup(
            message=message,
            id=popup_id,
            buttons=buttons or [],
        ))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.pending_popup_acks.pop(popup_id, None)

    def send_ui(self, **kwargs):
        """
        Ensure that the given telemetry item is sent to every connected UI
        keyword args are passed directly to telemetry item, so you can construct one like this

        self.send_ui(pop_message=telemetry.Popup('hello'))

        Thread safe. Nothing leaves the process until flush_tele_buffer.
        """
        # remember which calibration step is on screen, so an abort can record the step it
        # stopped on. An empty action carries no step and would only erase the last real one.
        progress = kwargs.get('operation_progress')
        if progress is not None and progress.name == 'Calibration' and progress.current_action:
            self._calibration_step = (progress.percent_complete, progress.current_action)
        self.telemetry.send(**kwargs)

    async def flush_tele_buffer(self):
        """
        Flush the teloperation buffer. sending all data to all UI clients.
        Normally called within position estimator's 60hz loop
        """
        await self.telemetry.flush()

    async def start_pe_when_ready(self):
        await self.any_anchor_connected.wait()
        r = await self.pe.main()

    async def main(self) -> None:
        self._check_startup_sequence()
        self.startup_complete.clear()
        if self.debug:
            from nf_robot.host.loop_monitor import LoopMonitor
            self.loop_monitor = LoopMonitor(interval=0.5, threshold=0.2)
            self.loop_monitor.start()

        self.passive_safety_task = asyncio.create_task(self.passive_safety())
        self.gantry_visibility_task = asyncio.create_task(self.monitor_gantry_visibility())
        self.spool_monitor_task = asyncio.create_task(self.monitor_spools())
        self._clear_item_task = asyncio.create_task(self._watch_for_clear_item())

        self.telemetry.start_cloud_link()

        # statistic counter - measures things like average camera frame latency
        asyncio.create_task(self.stat.stat_main())

        # A task that continuously estimates the position of the gantry
        # remains asleep until at least one anchor connects.
        self.pe_task = asyncio.create_task(self.start_pe_when_ready())

        # main process must own pool, and there's only one. multiple subprocesses may submit work.
        with Pool(processes=3, initializer=_ignore_sigint) as pool:
            self.pool = pool

            # zeroconf only discovers services and keeps their addresses and ports up to date in the config.
            # start a task to connect and reconnect to all known robot components.
            self.keeper = asyncio.create_task(self.keep_robot_connected())

            # the only reason it might not be none is if a unit test set before calling main.
            if self.aiozc is None:
                self.aiozc = AsyncZeroconf(ip_version=IPVersion.V4Only, interfaces=InterfaceChoice.All)

            try:
                services = list(
                    await AsyncZeroconfServiceTypes.async_find(aiozc=self.aiozc, ip_version=IPVersion.V4Only)
                )
                self.aiobrowser = AsyncServiceBrowser(
                    self.aiozc.zeroconf, services, handlers=[self.on_service_state_change]
                )
            except asyncio.exceptions.CancelledError:
                await self.aiozc.async_close()
                return

            # perception model — always started; target inference activates via SetTargetModel at runtime
            self.perception_task = asyncio.create_task(self.run_perception())

            # optionally self-host the playroom-ui frontend, so a browser elsewhere on the LAN
            # can load the full cockpit UI from this machine with no dependency on
            # neufangled.com — see webui_server.py and playroom-ui/README.md.
            if self.serve_ui:
                self.webui_server = WebUiServer(port=self.ui_port, bind_address=self.bind_address)
                try:
                    self.webui_server.start()
                except RuntimeError as e:
                    logger.error(str(e))
                    self.webui_server = None

            # start a websocket server to accept incoming connections from either a local UI or local Lerobot session
            async with self.telemetry.serving():
                # await something that will end when the program closes to keep serving and
                # keep zeroconf alive and discovering services.
                try:
                    await self._start_maneuvers()
                    self.startup_complete.set()

                    # Show an appropriate banner for the user to open in thier browser.
                    server_robotid = self.telemetry.cloud_robot_id
                    if self.webui_server is not None:
                        if self.bind_address in ("0.0.0.0", "::", ""):
                            advertised_host = get_local_ip() or "localhost"
                        else:
                            advertised_host = self.bind_address
                        message = f'To control visit http://{advertised_host}:{self.ui_port}/'
                    elif self.telemetry_env == None:
                        message = f'To control visit https://neufangled.com/playroom?robotid=lan on this machine'
                    elif self.telemetry_env == 'local':
                        message = f'To control visit http://localhost:5173/playroom?robotid={server_robotid}'
                    elif self.telemetry_env == 'production':
                        message = f'To control visit https://neufangled.com/playroom?robotid={server_robotid}'
                    elif self.telemetry_env == 'staging':
                        message = f'To control visit https://nf-site-monolith-staging-690802609278.us-east1.run.app/playroom?robotid={server_robotid}'
                    else:
                        print(f'invalid telemetry_env {self.telemetry_env}')

                    bar = '=' * (len(message) + 12)
                    print(bar)
                    print(f'===== {message} =====')
                    print(bar)

                    result = await self.keeper
                except asyncio.exceptions.CancelledError:
                    pass

            await self.async_close()

    async def async_close(self) -> None:
        print('Stringman Controller Shutdown')

        # Disable the per-client safety watchdogs first.
        for client in self.bot_clients.values():
            if client.safety_task is not None:
                client.safety_task.cancel()

        # Start watchdog that prints diagnostics if shutdown isn't fast
        # This runs in a *thread*, not an asyncio task, on purpose.
        loop = asyncio.get_running_loop()
        watchdog = threading.Timer(3.0, self._dump_shutdown_diagnostics, args=(loop,))
        watchdog.daemon = True
        watchdog.start()
        try:
            await self._async_close_impl()
        finally:
            watchdog.cancel()

    def _dump_shutdown_diagnostics(self, loop) -> None:
        """Watchdog callback (runs in a thread) when async_close() runs long."""
        print('\n=== async_close() still running after 3s — dumping diagnostics ===',
              file=sys.stderr, flush=True)
        # Every thread's Python stack. This reveals the main thread even when it
        # is blocked in synchronous code holding up the event loop.
        faulthandler.dump_traceback()
        # Suspended coroutines won't show up above (they aren't on any thread's
        # stack), so also list the pending asyncio tasks and where each parked.
        try:
            for task in asyncio.all_tasks(loop):
                if task.done():
                    continue
                print(f'--- pending task {task!r} ---', file=sys.stderr, flush=True)
                task.print_stack(file=sys.stderr)
        except Exception as e:
            print(f'  could not enumerate asyncio tasks: {e!r}', file=sys.stderr, flush=True)

    async def _async_close_impl(self) -> None:
        # persist the last observed named positions (e.g. hamper, parking_location) so they survive a restart
        self.config.last_gantry_pos = fromnp(self.pe.gant_pos)
        save_config(self.config, self.config_path)
        # Stop the loop monitor (also restores the patched Handle._run).
        if self.loop_monitor is not None:
            await self.loop_monitor.stop()
        result = await self.stop_all()
        await self._stop_maneuvers()
        if self.webui_server is not None:
            self.webui_server.stop()
        self.run_command_loop = False
        self.stat.run = False
        self.pe.run = False
        self.pe_task.cancel()
        tasks = [self.pe_task, self.keeper]
        tasks.extend([client.shutdown() for client in self.bot_clients.values()])
        tasks.append(self.telemetry.aclose())
        if self.aiobrowser is not None:
            tasks.append(self.aiobrowser.async_cancel())
        if self.aiozc is not None:
            tasks.append(self.aiozc.async_close())
        if self.locate_anchor_task is not None:
            tasks.append(self.locate_anchor_task)
        if self.gip_task is not None:
            tasks.append(self.gip_task)
        if self.swing_cancellation_task is not None:
            self.swing_cancellation_task.cancel()
            tasks.append(self.swing_cancellation_task)
        if self._clear_item_task is not None:
            self._clear_item_task.cancel()
            tasks.append(self._clear_item_task)
        if self.perception_task is not None:
            self.perception_task.cancel()
            tasks.append(self.perception_task)
        if self.passive_safety_task is not None:
            self.passive_safety_task.cancel()
            tasks.append(self.passive_safety_task)
        if self.gantry_visibility_task is not None:
            self.gantry_visibility_task.cancel()
            tasks.append(self.gantry_visibility_task)
        if self.spool_monitor_task is not None:
            self.spool_monitor_task.cancel()
            tasks.append(self.spool_monitor_task)
        self.stop_spool_log()

        try:
            result = await asyncio.gather(*tasks)
        except asyncio.exceptions.CancelledError:
            pass

    async def add_simulated_data_point2point(self):
        """Simulate the gantry moving from random point to random point.
        The only purpose of this simulation at the moment is to test the position estimator and it's feedback
        """
        LOWER_Z_BOUND = 1.0 # meters
        UPPER_Z_OFFSET = 0.3 # meters
        MAX_SPEED_MPS = 0.25 # m/s
        GOAL_PROXIMITY_THRESHOLD = 0.03 # meters
        SOFT_SPEED_FACTOR = 0.25
        RANDOM_EVENT_CHANCE = 0.5
        CAM_BIAS_STD_DEV = 0.2 # meters
        OBSERVATION_NOISE_STD_DEV = 0.01 # meters
        WINCH_LINE_LENGTH = 1.0 # meters
        RANGEFINDER_OFFSET = 1.0 # meters
        LOOP_SLEEP_S = 0.05 # seconds
        
        # each camera produces measurements with a position bias that can be around 20x larger than the position noise from a given camera.
        cam_bias = np.random.normal(0, CAM_BIAS_STD_DEV, (4, 3))

        pending_obs = deque()

        lower = np.min(self.pe.anchor_points, axis=0)
        upper = np.max(self.pe.anchor_points, axis=0)
        lower[2] = LOWER_Z_BOUND
        upper[2] = upper[2] - UPPER_Z_OFFSET
        # starting position
        gantry_real_pos = np.random.uniform(lower, upper)
        # initial goal
        travel_goal = np.random.uniform(lower, upper)
        t = time.time()
        while self.run_command_loop:
            try:
                now = time.time()
                elapsed_time = now - t
                t = now
                # move the gantry towards the goal
                to_goal_vec = travel_goal - gantry_real_pos
                dist_to_goal = np.linalg.norm(to_goal_vec)
                if dist_to_goal < GOAL_PROXIMITY_THRESHOLD:
                    # choose new goal
                    travel_goal = np.random.uniform(lower, upper)
                else:
                    soft_speed = dist_to_goal * SOFT_SPEED_FACTOR
                    # normalize
                    to_goal_vec = to_goal_vec / dist_to_goal
                    velocity = to_goal_vec * min(soft_speed, MAX_SPEED_MPS)
                    gantry_real_pos = gantry_real_pos + velocity * elapsed_time
                if random() > RANDOM_EVENT_CHANCE:
                    anchor_num = np.random.randint(4) # which camera it was observed from.
                    observed_position = gantry_real_pos + cam_bias[anchor_num] + np.random.normal(0, OBSERVATION_NOISE_STD_DEV, (3,))
                    dp = np.concatenate([[t], [anchor_num], observed_position])
                    # simulate delayed data
                    pending_obs.appendleft(dp)
                    if len(pending_obs) > 10:
                        dp = pending_obs.pop()
                        self.datastore.gantry_pos.insert(dp)
                        self.datastore.gantry_pos_event.set()
                        self.send_ui(gantry_sightings=telemetry.GantrySightings(sightings=[fromnp(dp[2:])]))
                
                # winch line always 1 meter
                self.datastore.winch_line_record.insert(np.array([t, WINCH_LINE_LENGTH, 0.0]))
                
                # range always perfect
                self.datastore.range_record.insert(np.array([t, gantry_real_pos[2]-RANGEFINDER_OFFSET]))

                # anchor lines always perfectly agree with gripper position
                for i, simanc in enumerate(self.pe.anchor_points):
                    dist = np.linalg.norm(simanc - gantry_real_pos)
                    last = self.datastore.anchor_line_record[i].getLast()
                    timesince = t-last[0]
                    travel = dist-last[1]
                    speed = travel/timesince # referring to the specific speed of this line, not the gantry
                    self.datastore.anchor_line_record[i].insert(np.array([t, dist, speed, 1.0]))
                    self.datastore.anchor_line_record_event.set()
                tt = self.datastore.anchor_line_record[0].getLast()[0]
                await asyncio.sleep(LOOP_SLEEP_S)
            except asyncio.exceptions.CancelledError:
                break

    async def send_gripper_move(self, line_speed, finger_speed, wrist_speed):
        """Command the gripper's motors in one update.
        finger speed is in degrees per second (but it's the fake degrees of the finger which range from -90 (open) to 90 (closed))
        positive values close the fingers.
        wrist speed is in real degrees per second."""
        update = {}

        if self.gripper_client is not None:
            cg = telemetry.CommandedGrip()
            if finger_speed is not None:
                finger_speed = clamp(finger_speed, -90, 90)
                update['set_finger_speed'] = finger_speed
                cg.finger_speed = finger_speed
            if wrist_speed is not None:
                wrist_speed = clamp(wrist_speed, -120, 120)
                update['set_wrist_speed'] = wrist_speed
                cg.wrist_speed = wrist_speed
            self.send_ui(last_commanded_grip=cg)
            r = await self.flush_tele_buffer()

        if update:
            asyncio.create_task(self.gripper_client.send_commands(update))
        return line_speed, finger_speed, wrist_speed

    async def send_gripper_move_legacy(self, line_speed, finger_angle, wrist_angle):
        """Command the gripper's motors in one update."""
        update = {}
        if line_speed is not None:
            update['aim_speed'] = line_speed
        if finger_angle is not None:
            update['set_finger_angle'] = clamp(finger_angle, -90, 90)
        if wrist_angle is not None:
            clamped = clamp(wrist_angle, 0, 1080)
            update['set_wrist_angle'] = clamped
        if update and self.gripper_client is not None:
            asyncio.create_task(self.gripper_client.send_commands(update))
        return line_speed, finger_angle, wrist_angle

    async def clear_goal(self):
        """End the seek in flight, if any, wherever it has got to."""
        self._goal_pos = None
        self.send_ui(named_position=telemetry.NamedObjectPosition(name='gantry_goal_marker')) # not setting position causes it to be hidden

    async def seek_goal(self, goal_pos, head_turn=False, auto_altitude=True, timeout=None):
        """
        Fly the gantry to goal_pos, using the constantly updating gantry position provided
        by the position estimator. True once it arrives.

        goal_pos is where the GANTRY goes, not the gripper. The gripper hangs self.pole
        below it, so a caller aiming the gripper at something must add self.pole to the goal.

        A seek already in flight is re-aimed at goal_pos rather than started over, so calling
        this again with a better goal steers the same flight with no stop between. With a
        timeout, returns False after that many seconds and leaves the flight going, for a
        caller that wants to look again while it travels; clear_goal() ends it, as does
        cancelling any caller that is waiting on it.
        head_turn and auto_altitude are fixed when a flight starts.
        This is a motion task.
        when head_turn, turn gripper to face direction of motion.
        when auto_altitude, room traversal is performed at an ideal altitude
        """
        if goal_pos is None:
            return False
        self._goal_pos = np.asarray(goal_pos, dtype=float)
        self.send_ui(named_position=telemetry.NamedObjectPosition(position=fromnp(self._goal_pos), name='gantry_goal_marker'))
        if self._seek_task is None or self._seek_task.done():
            self._seek_task = asyncio.create_task(self._fly_to_goal(head_turn, auto_altitude))
        flight = self._seek_task
        try:
            done, _ = await asyncio.wait([flight], timeout=timeout)
        except asyncio.CancelledError:
            # waited out here, so the flight's own cleanup cannot land on a seek started
            # right after this one
            flight.cancel()
            await asyncio.gather(flight, return_exceptions=True)
            raise
        if flight not in done:
            return False
        return flight.result()

    async def _end_seek(self):
        """Cancel the seek in flight and wait for it to have stopped the spools."""
        flight = self._seek_task
        if flight is not None and not flight.done():
            flight.cancel()
            await asyncio.gather(flight, return_exceptions=True)

    async def _fly_to_goal(self, head_turn, auto_altitude):
        """The flight seek_goal starts: steers onto self._goal_pos, which may change under it,
        until it arrives (True) or the goal is cleared (False)."""
        GOAL_PROXIMITY_M = 0.08
        MAX_SPEED = 0.4 # GANTRY_SPEED_MPS
        ACCEL = 0.15     # m/s^2
        ARRIVAL_SPEED_MPS = 0.03 # what it should still be doing when it lets go of the goal
        LOOP_SLEEP_S = 0.1
        IDEAL_GANTRY_ALTITUDE = 1.3 # meters. ideal gantry height for room traversal
        CLIMB_RATE = 0.15 # m/s, constant rate of altitude change for auto_altitude
        ALTITUDE_DEADBAND_M = 0.05 # meters, tolerance to avoid hunting around target altitude

        current_speed = 0.0
        final_approach = False # latches once True so the altitude target doesn't flip back to cruise
        arrived = False

        try:
            dist_to_goal = 10
            while self._goal_pos is not None:
                vector = self._goal_pos - self.pe.gant_pos
                dist_to_goal = np.linalg.norm(vector)

                if dist_to_goal < GOAL_PROXIMITY_M:
                    arrived = True
                    logger.info(f'Goal reached {tuple(self._goal_pos)}')
                    break

                # Ramp down as the goal approaches: v = sqrt(2 * a * d). The d that matters
                # is the distance to where this loop lets go, not to the goal itself: a ramp
                # planned to reach zero at the centre is still asking for 0.15m/s at the
                # proximity radius, and the gantry carries that straight into an overshoot.
                # It bottoms out at a crawl rather than at zero so the last few centimetres
                # are actually covered.
                ramp_dist_to_goal = np.linalg.norm(vector[:2]) if auto_altitude else dist_to_goal
                braking_dist = max(0.0, ramp_dist_to_goal - GOAL_PROXIMITY_M)
                speed_ramp_down = max(ARRIVAL_SPEED_MPS, np.sqrt(2 * ACCEL * braking_dist))

                # Target speed is the ramp-down limit or the max allowable speed
                target_speed = min(speed_ramp_down, MAX_SPEED)

                # Smoothly interpolate current_speed toward target_speed to prevent
                # instantaneous velocity jumps between loop iterations. Slowing is allowed to
                # be twice as brisk as speeding up, so that what governs the approach is the
                # ramp above and not this limit tracking it a step behind.
                step = ACCEL * LOOP_SLEEP_S
                if current_speed < target_speed:
                    current_speed = min(current_speed + step, target_speed)
                else:
                    current_speed = max(current_speed - 2 * step, target_speed)

                if head_turn:
                    self.gripper_client.look_towards_vector(vector[:2])

                if auto_altitude:
                    # Like an aircraft: climb/descend at a constant rate, cruising at
                    # IDEAL_GANTRY_ALTITUDE, then ramp down to the goal's altitude.
                    # Start descending as soon as the remaining horizontal travel time
                    # (at best case speed) wouldn't be enough to reach the goal altitude
                    # at CLIMB_RATE, so short traversals may never reach cruise altitude.
                    horizontal_dist = np.linalg.norm(vector[:2])
                    current_altitude = self.pe.gant_pos[2]
                    goal_altitude = self._goal_pos[2]
                    altitude_error = goal_altitude - current_altitude
                    time_to_arrive = horizontal_dist / MAX_SPEED
                    time_to_descend = abs(altitude_error) / CLIMB_RATE
                    if time_to_arrive <= time_to_descend:
                        final_approach = True
                    target_altitude = goal_altitude if final_approach else IDEAL_GANTRY_ALTITUDE

                    altitude_diff = target_altitude - current_altitude
                    if abs(altitude_diff) < ALTITUDE_DEADBAND_M:
                        vertical_speed = 0.0
                    else:
                        vertical_speed = np.sign(altitude_diff) * CLIMB_RATE

                    horizontal_uvec = vector[:2] / horizontal_dist if horizontal_dist > 1e-5 else np.zeros(2)
                    velocity = np.array([*(horizontal_uvec * current_speed), vertical_speed])
                    await self.move_direction_speed(velocity, None, self.pe.gant_pos)
                else:
                    # Normalize vector and command movement
                    await self.move_direction_speed(vector / dist_to_goal, current_speed, self.pe.gant_pos)
                await asyncio.sleep(LOOP_SLEEP_S)
            return arrived
        except asyncio.CancelledError:
            logger.debug('Goal move cancelled')
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_goal()

    async def send_line_speed(self, line_no, speed, jog=False):
        # send the line speed to the client that controls that line
        # when jog==True, speed is interpreted as a length in meters by which to lengthen the line
        command = 'jog' if jog else 'aim_speed'
        self.line_speed_cmds[line_no] = (time.time(), float(speed), 'jog' if jog else 'aim')
        if line_no//2 in self.anchors:
            spool_no = line_no%2
            # we consider the lower line number to be the direct line
            r = await self.anchors[line_no//2].send_commands({command: (speed, spool_no)})

    async def set_line_tension_target(self, line_no, value):
        """Set (or clear, with None) the onboard two-sided tension hold target in newtons
        for one arpeggio line. The onboard loop then holds that line at the target."""
        if line_no//2 in self.anchors:
            spool_no = line_no % 2
            await self.anchors[line_no//2].send_commands({'set_tension_target': (value, spool_no)})

    async def move_direction_speed(self, uvec, speed=None, starting_pos=None, downward_bias=-0.04, key=DEFAULT_VELOCITY_KEY):
        """Move in the direction of the given unit vector at the given speed.
        Any move must be based on some assumed starting position. if none is provided,
        we will use the last one sent from position_estimator

        Due to inaccuaracy in the positions of the anchors and lengths of the lines,
        the speeds we command from the spools will not be perfect.
        On average, half will be too high, and half will be too low.
        Because there are four lines and the gantry only hangs stably from three,
        the actual point where the gantry ends up hanging after any move will always be higher than intended
        So a small downward bias is introduced into the requested direction to account for this.
        The size of the bias should theoretically be a function of the the magnitude of position and line errors,
        but we don't have that info. alternatively we could calibrate the bias to make horizontal movements level
        according to the laser rangefinder.

        if speed is None, uvec is assumed to be velocity and used directly with no bias

        If key is supplied, the resulting vector overwrites the last one with the same key
        Whenever one of the keys from the set that is being combined changes, all keys in the active set are summed and sent to the anchors.
        """
        KINEMATICS_STEP_SCALE = 10.0 # Determines the size of the virtual step to calculate line speed derivatives

        if starting_pos is None:
            starting_pos = self.pe.gant_pos

        # when speed is not provided, use uvec as a velocity vector in m/s (mode used with lerobot)
        if speed is None:
            speed = np.linalg.norm(uvec)

        # when a very small speed is provided, clamp it to zero.
        if speed < 0.001:
            speed = 0

        if speed == 0:
            velocity = np.zeros(3)
        else:
            # normalize, apply downward bias and renormalize
            uvec  = uvec / (np.linalg.norm(uvec) + 1e-5)
            uvec = uvec + np.array([0,0,downward_bias])
            uvec  = uvec / (np.linalg.norm(uvec) + 1e-5)
            velocity = uvec * speed

        # An empty/unset source key maps to the shared 'default' source.
        if not key:
            key = DEFAULT_VELOCITY_KEY
        # this commanded velocity overwrites the last velocity with the same key and all velocities are summed
        # currently this is only used to combine swing cancellation with user inputs.
        self.input_velocities[key] = (velocity, time.monotonic())
        # ensure this source contributes to the sum; stale sources expire lazily via TTL pruning.
        self.active_set.add(key)
        self._prune_input_velocities() # drop any source keys that have gone stale
        # the key we just set is always fresh and in the active set, so the sum is guaranteed a 3-vector
        total_velocity = np.sum([self.input_velocities[k][0] for k in self.active_set if k in self.input_velocities], axis=0)
        
        # Determine the total requested speed before limits
        speed = np.linalg.norm(total_velocity)

        # enforce a model dependent speed limit
        speed_limit = self.speed_limit()

        if speed > speed_limit:
            total_velocity = total_velocity * (speed_limit / speed)
            speed = speed_limit

        # line lengths at starting pos
        lengths_a = np.linalg.norm(starting_pos - self.pe.anchor_points, axis=1)
        # line lengths at new pos
        new_pos = starting_pos + (total_velocity / KINEMATICS_STEP_SCALE)
        
        # zero the speed if this would move the gantry out of the work area
        if not self.pe.point_inside_work_area(new_pos):
            speed = 0
            total_velocity = np.zeros(3)
            
        lengths_b = np.linalg.norm(new_pos - self.pe.anchor_points, axis=1)
        deltas = lengths_b - lengths_a
        line_speeds = deltas * KINEMATICS_STEP_SCALE

        # send the move on every line at once
        await asyncio.gather(*[
            self.send_line_speed(i, line_speed)
            for i, line_speed in enumerate(line_speeds)
        ])
            
        self.pe.record_commanded_vel(total_velocity)
        return total_velocity

    def get_last_frame(self, camera_key):
        """gets the last frame of video from the given camera if possible
        camera_key should be one of 'g' 0, 1, 2, 3
        """
        image = None
        if camera_key == 'g':
            if self.gripper_client is not None:
                image = self.gripper_client.lerobot_jpeg_bytes
        else:
            image = self.anchors[int(camera_key)].lerobot_jpeg_bytes
        if image is not None:
            return image
        return bytes()

    def _ortho_worker(self, ortho_floor_vs):
        """
        Sync thread driven by self.ortho_event, which anchor stream_video_loops set on every
        new processed frame.  Projects all anchor views onto the floor and stores the result so
        the AI task can read it without re-running the projection.
        """
        from nf_robot.host.floor_view import generate_orthographic_floor_maps
        EXTENT = 5.0
        while self.run_command_loop:
            if not self.ortho_event.wait(timeout=1.0):
                continue
            self.ortho_event.clear()
            try:
                valid_clients = [
                    c for c in list(self.anchors.values())
                    if c.last_output_frame is not None and c.anchor_num in self.config.preferred_cameras
                ]
                if not valid_clients:
                    continue

                ortho_rgb = generate_orthographic_floor_maps(
                    valid_clients, self.config.camera_cal,
                    map_size_px=1000, map_extent_meters=EXTENT,
                )
                self.last_ortho_rgb = ortho_rgb

                if ortho_floor_vs is not None:
                    # the streamer's encoders take BGR
                    ortho_floor_vs.send_frame(cv2.cvtColor(ortho_rgb, cv2.COLOR_RGB2BGR))
            except Exception:
                logger.exception('_ortho_worker iteration failed')

    async def run_perception(self):
        """
        Orthographic floor projection, published as latest_ortho() and as video feed 3.
        Target inference on it belongs to the pick_and_place maneuver.
        """
        LOOP_DELAY = 0.1

        # wait until at least one preferred camera is producing frames
        logging.info('waiting for camera frames')
        while True:
            await asyncio.sleep(1)
            have_frames = (
                (self.gripper_client is not None and self.gripper_client.last_output_frame is not None)
                or any(
                    anum in self.config.preferred_cameras and c.last_output_frame is not None
                    for anum, c in self.anchors.items()
                )
            )
            if have_frames:
                break

        ortho_floor_vs = None
        if self.run_ortho:
            from nf_robot.host.video_streamer import NfVideoStreamer

            def _make_on_ready(feed_number):
                def on_ready(local_uri, stream_path):
                    t = telemetry.VideoReady(
                        is_gripper=None,
                        anchor_num=None,
                        local_uri=local_uri,
                        stream_path=stream_path,
                        feed_number=feed_number,
                    )
                    logger.debug(f'sending {t}')
                    self.send_ui(video_ready=t)
                return on_ready

            ortho_floor_vs = NfVideoStreamer(
                width=1000, height=1000, fps=10,
                mjpeg_port=8747,
                stream_path=f'stringman/{self.telemetry.cloud_robot_id}/3',
                telemetry_env=self.telemetry_env,
                on_ready=_make_on_ready(3),
                bind_address=self.bind_address,
            )
            ortho_floor_vs.start()
            self.ortho_streamers = [(ortho_floor_vs, 3)]

        ortho_thread = threading.Thread(
            target=self._ortho_worker,
            args=(ortho_floor_vs,),
            daemon=True,
        )
        ortho_thread.start()

        while self.run_command_loop:
            await asyncio.sleep(LOOP_DELAY)

        if self.run_ortho:
            ortho_floor_vs.stop()

    async def grasp(self):
        """Try to grasp whatever is directly below the gripper. True if it is now held."""
        lerobot = self.maneuvers.get('lerobot')
        if lerobot is not None and lerobot.use_for_grasp:
            # A lerobot session may be driving from our own subprocess or connected
            # remotely through the prod telemetry relay, so we can't tell locally if one
            # is present. Its grasp broadcasts the eval-start and returns None if no
            # session answers, which is recoverable: servoing needs nothing but the robot.
            result = await lerobot.grasp()
            if result is not None:
                return result
            logger.warning('--lerobot_grasp is set but no session answered; servoing instead')
        if not await self.servo.ensure_model():
            logger.warning('No visual servoing model loaded; cannot grasp')
            return False
        # the model steers from the palm camera, so a swinging gripper moves the target under
        # it between the frame it decided on and the descent it decided to make
        async with self.prefer_swing_cancellation():
            return await self.servo.run(mode=SERVO_MODE_GRASP)

def main():
    """
    Run stringman in a headless manner

    note that connecting to a local telemetry enviroment is distinct from lan mode
    To run in LAN mode, do not pass --telemetry_env
    observer.py will listen on port 4245
    
    Whenever --telemetry_env is set, observer.py is connecting to some telemetry server
    even if it is the full stack running on the local machine
    """
    parser = argparse.ArgumentParser(description="Stringman motion controller")
    parser.add_argument("--config", type=str, default='configuration.json')
    parser.add_argument(
            '--telemetry_env',
            type=str,
            choices=['local', 'staging', 'production'],
            default=None,
            help="The cloud telemetry server to connect to (choices: local, staging, production) Used in development only. The default is None, which allows local connections on port 4245 only"
        )
    parser.add_argument("--prod", action="store_true", help="Shorthand for --telemetry_env=production")
    parser.add_argument("--no_ortho", action="store_true", help="Disable orthographic floor projection and its video streams")
    parser.add_argument("--auto_start", action="store_true", help="Automatically unpark and start cleaning when all components connect")
    parser.add_argument("--local_models", action="store_true", help="Use local models from models/ rather than downloading the production models from huggingface")
    parser.add_argument(
        "--lerobot_grasp",
        action="store_true",
        help="Grasp with a connected lerobot policy session rather than the visual servoing "
             "model (see ml/visual_servoing/readme.md), which is the default. Falls back to "
             "servoing if no session answers."
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
    parser.add_argument(
        "--rec_diagnostics",
        action="store_true",
        help="Record the arguments of every optimize_arp_anchors call during full_auto_calibration "
             "to calibration_diagnostics.pkl, for offline analysis. Arpeggio hardware only."
    )
    parser.add_argument(
        "--bind_address",
        type=str,
        default="127.0.0.1",
        help="Interface for the local telemetry websocket (port 4245) and all local mjpeg video "
             "streams. Set to 0.0.0.0 to access from elsewhere on your network."
    )
    parser.add_argument(
        "--no_serve_ui",
        action="store_true",
        help="Don't serve the playroom-ui frontend from this machine."
    )
    parser.add_argument(
        "--ui_port",
        type=int,
        default=8090,
        help="Port to serve the self-hosted UI on, unless --no_serve_ui is set. Defaults to 8090."
    )
    parser.add_argument(
        "--diamond_size",
        type=float,
        nargs=3,
        metavar=("HALF_HEIGHT", "HALF_WIDTH", "FLOOR_CLEARANCE"),
        default=list(DIAMOND_SIZE),
        help="Calibration diamond geometry in meters: half-height, half-width, and the floor "
             "clearance of the bottom (starting) point. Defaults to %s." % (tuple(DIAMOND_SIZE),)
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        if sys.platform == "darwin":
            install_cmd = "brew install ffmpeg"
        else:
            install_cmd = "sudo apt install ffmpeg"
        print(f"ffmpeg is required but was not found on your PATH. Install it with:\n\n    {install_cmd}\n", file=sys.stderr)
        sys.exit(1)

    if args.prod:
        if args.telemetry_env not in (None, 'production'):
            parser.error("--prod conflicts with --telemetry_env=%s" % args.telemetry_env)
        args.telemetry_env = 'production'

    if args.debug:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s %(message)s', datefmt='%H:%M:%S')
        logging.getLogger('nf_robot').setLevel(logging.DEBUG)

    async def run_async():
        runner = AsyncObserver(
            False,
            args.config,
            telemetry_env=args.telemetry_env,
            run_ortho=(not args.no_ortho),
            auto_start=args.auto_start,
            local_models=args.local_models,
            debug=args.debug,
            bind_address=args.bind_address,
            rec_diagnostics=args.rec_diagnostics,
            serve_ui=(not args.no_serve_ui),
            ui_port=args.ui_port,
            diamond_size=tuple(args.diamond_size),
            lerobot_grasp=args.lerobot_grasp,
        )

        # Idempotent stop trigger. Runs as a signal-handler callback on the event
        # loop thread, so it must not block: schedule the telemetry-socket abort
        # for later instead of time.sleep()-ing on the loop.
        def stop():
            runner.run_command_loop = False
            asyncio.get_running_loop().call_later(0.5, runner.telemetry.abort_cloud_socket)

        # On Unix, register signal handler.
        # On Windows, catch keyboard interrupt
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, stop)
        
        try:
            r = await runner.main()
        except KeyboardInterrupt:
            stop()

    asyncio.run(run_async())

if __name__ == "__main__":
    main()
