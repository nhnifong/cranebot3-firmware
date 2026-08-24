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
from collections import deque, defaultdict
import uuid
from functools import partial
from pathlib import Path
import json
import re
import subprocess
import zipfile
from packaging.version import parse as parse_version, InvalidVersion

from nf_robot.common.pose_functions import compose_poses, invert_pose
from nf_robot.common.cv_common import *
from nf_robot.common.config_loader import *
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common
import nf_robot.generated.nf.config as nf_config
from nf_robot.host.data_store import DataStore
from nf_robot.host.fake_progress import FakeProgress
from nf_robot.host.stats import StatCounter
from nf_robot.host.target_queue import TargetQueue
from nf_robot.host.eyelet_calibration import optimize_arp_anchors, analyze_diamond_data, DIAMOND_SIZE
from nf_robot.host.component_client import max_origin_detections
from nf_robot.host.arp_gripper_client import (ArpeggioGripperClient, rotate_vector,
                                              ROUTE_TAG_MAX_AGE_S, CAPTURE_RESOLUTION_SIZE,
                                              OPEN, CLOSED, RANGE_MAX_AGE_S)
from nf_robot.host import swing
from nf_robot.host.visual_servo import VisualServo, SERVO_MODE_GRASP, SERVO_MODE_OBSERVE, SERVO_MODE_CENTER
from nf_robot.host.arp_anchor_client import ArpeggioAnchorClient
from nf_robot.host.position_estimator import Positioner2
from nf_robot.host.telemetry_manager import TelemetryManager, LOCAL
from nf_robot.host.webui_server import WebUiServer

logger = logging.getLogger(__name__)

# Define the service names for network discovery
arp_gripper_service_name = 'cranebot-gripper-arpeggio-service'
arp_anchor_service_name = 'cranebot-anchor-arpeggio-service'

N_ANCHORS = 2
N_LINES = 4
INPUT_VELOCITY_TTL_S = 2.0 # a commanded velocity keyed by a source expires this long after its last update
INFO_REQUEST_TIMEOUT_MS = 3000 # milliseconds
# visual centering nudges. The move is open loop (commanded speed for a computed duration), so
# the speed trades travel time against how much overshoot and swing each step leaves behind.
NUDGE_SPEED_MPS = 0.12
NUDGE_SETTLE_S = 0.3
NUDGE_REFRESH_S = 0.5 # must stay under INPUT_VELOCITY_TTL_S or the nudge expires mid-move
NUDGE_VELOCITY_KEY = 'centering'
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
# Motion tasks that turn sightings of the gantry marker into stored geometry, so a marker
# fault does not degrade them, it gets fitted. monitor_gantry_visibility aborts these.
MARKER_DEPENDENT_TASKS = ('full_auto_calibration',)

# Capture runs for the synthetic visual servoing dataset; see ml/visual_servoing/readme.md.
PLATE_OUTPUT_DIR = 'plates'
# Finger sweep bounds. The server clamps finger angle to -90 (open) .. 90 (closed), and a
# plate is wanted at every aperture the fingers are actually driven to during a grasp.
FINGERPLATE_ANGLE_MIN = -76
FINGERPLATE_ANGLE_MAX = 90
FINGERPLATE_ANGLE_STEP = 2
# Frames per wrist turn. The matte keys each one and takes the median, so it wants enough
# of them that anything which rotated past is outvoted - and they are cheap next to the
# finger moves between turns.
FINGERPLATE_WRIST_STEPS = 18
# (seconds) extra wait after the wrist reports arrival, due to the higher video latency from this format
FINGERPLATE_SETTLE_S = 0.3
# How long to wait for the camera to come back at the capture resolution. rpicam-vid is
# killed and relaunched to change resolution, and the client retries the connection a few
# times before giving up, so this has to cover all of that.
CAPTURE_STREAM_TIMEOUT_S = 30.0
# Consecutive missing frames that mean the stream has gone rather than lagged.
FINGERPLATE_MAX_MISSES = 5
# (metres) rangefinder readings to capture floor and object plates at. Spans the heights
# a gripper actually approaches from, and is what calibrates a plate's apparent scale
# against the range it was taken at, so the compositor can rescale it to any simulated
# height. Trimmed to by measurement, not by commanded altitude.
PLATE_RANGES_M = (0.12, 0.28, 0.44, 0.60, 0.74)
# (degrees/second, degrees) the continuous wrist sweep floor and object plates are
# captured during. Slow enough that the pole does not swing and frames stay sharp.
PLATE_WRIST_SPEED_DPS = 30.0
PLATE_SWEEP_DEGREES = 360.0
# (seconds) how often a wrist speed command is repeated to keep the sweep going.
WRIST_SPEED_REFRESH_S = 0.1
# (seconds) how often the robot's state is sampled beside a recorded video sweep.
TELEMETRY_SAMPLE_S = 0.05
# (seconds) grace for the demux loop to notice a recording has been asked to stop.
RECORDING_CLOSE_S = 1.0
# (degrees) fingers parked out of frame while capturing floor and object plates. -90 is
# fully open; anything the camera can see would be composited into every synthetic frame
# built from these plates.
PLATE_FINGERS_RETRACTED = -90.0
USER_TARGETS_DIR = "user_targets_data"
METADATA_PATH = os.path.join(USER_TARGETS_DIR, "metadata.jsonl")

# threshold of non slack tension in newtons for arp anchors
TENSION_THRESH = 1.38


# What visual_servo_grasp is allowed to do. Only GRASP descends, closes the fingers or
# reports success; the other two run until cancelled and exist for judging a checkpoint on
# a live robot, which is the only place this model can really be judged.
SERVO_MODE_GRASP = 'grasp'
SERVO_MODE_OBSERVE = 'observe'
SERVO_MODE_CENTER = 'center'
SERVO_MODES = (SERVO_MODE_GRASP, SERVO_MODE_OBSERVE, SERVO_MODE_CENTER)

# distance from the tip of the pole (self.pole[2] below the gantry) down to the bottom of
# the arp gripper fingers when they hang straight. gantry -> fingertip is self.pole[2] + this.
GRIPPER_FINGER_LEN_M = 0.18
GRIPPER_HEIGHT_OVER_TARGET = np.array([0,0,0.3])

# mapping from enums to MARKER_NAMES in cv_common
ROUTE_POINT_TAG_NAMES = {
    common.RoutePoint.HAMPER: "hamper",
    common.RoutePoint.TOYBOX: "toys",
    common.RoutePoint.TRASH: "trash",
    common.RoutePoint.GAMEPAD: "gamepad",
}

# feature key -> minimum nf_robot version every connected component must run to use it
VERSION_GATES = {
    "speed_0.45": "4.1.0",
    "gripper_card_survey": "4.2.0",
}

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
        # convenience reference to gripper client
        self.gripper_client = None
        # TODO allow a command line argument to override the config file path
        self.config_path = config_path
        self.config = load_config(config_path)
        # What the configured pole affects on this robot: how far the gripper hangs below the
        # gantry, which marker the gantry has, and the pendulum it swings as.
        self.pole_geometry = model_constants.pole_geometry(self.config)
        self.pole = np.array([0, 0, self.pole_geometry.gantry_to_gripper])
        self.gantry_april_inv = invert_pose(self.pole_geometry.gantry_april)
        self.pendulum = swing.pendulum_for(self.config)
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
        # set by passive_safety when line tension exceeds the safe limit during a running
        # calibration. Swing latency cal polls it to back off and retry the current trial;
        # any other calibration step is aborted (passive_safety cancels the task).
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
        # true while swing latency cal is running, so passive_safety recovers instead of
        # aborting on a tension trip during that step.
        self.swing_cal_in_progress = False
        # set by monitor_gantry_visibility to the phrase describing why it aborted a running
        # calibration, so the calibration's own cancel handler can report the real reason.
        self.gantry_marker_fault = None
        # which gantry marker faults have already been reported, so a standing fault is not a
        # repeating popup. A key is dropped when its condition clears, and the whole set is
        # cleared when a calibration starts, so a re-run of a calibration that was aborted by
        # a fault says so again instead of proceeding quietly.
        self._gantry_marker_warned = set()
        # only used for integration test only to allow some code to run right after sending the gantry to a goal point
        self.test_gantry_goal_callback = None
        # event used to notify tasks that gripper is connected.
        self.gripper_client_connected = asyncio.Event()
        self.last_user_move_time = time.time()
        # last known positions of named tags/objects live in self.config.named_positions
        # (the single source of truth). It's written to disk on shutdown, in async_close.
        self.target_model = None
        # Grasps with the visual servoing model, which is how grasping works unless
        # --lerobot_grasp hands it to a policy instead. Holds the checkpoint, loaded on
        # first use.
        self.servo = VisualServo(self)
        self.use_lerobot_grasp = lerobot_grasp
        self.perception_task = None
        self.webui_server = None
        # targets
        self.target_queue = TargetQueue()
        self.last_snapshot_hash = None # to spare the UI from too many updates
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
        self.input_velocities = {'default': (np.zeros(3), time.monotonic())}
        self.active_set = set(['default'])
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
        # list of (NfVideoStreamer, feed_number) for ortho feeds, so send_setup_telemetry can replay them
        self.ortho_streamers: list = []
        self.lerobot_process_watcher = None
        self.last_ep_ctrl_status = common.LerobotStatus.NA
        self.lerobot_process_pid = None
        # fires whenever any lerobot session (our own subprocess or one connected remotely
        # through the telemetry relay) reports a status. Used to detect whether a session is
        # actually listening after we broadcast an eval-start.
        self.lerobot_session_status_event = asyncio.Event()
        # futures awaiting a PopupAck, keyed by the Popup.id they were sent with
        self.pending_popup_acks: dict[int, asyncio.Future] = {}
        self._next_popup_id = 1
        # source and destination for pick and place. self.config is the source of truth;
        # these are kept in sync with self.config.last_route_source/last_route_destination.
        self.pnp_src = self.config.last_route_source
        self.pnp_dst = self.config.last_route_destination

    async def send_setup_telemetry(self):
        logger.debug('Sending setup telemetry')
        if self.config.anchor_type == common.AnchorType.ARPEGGIO:
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                poses=[a.pose for a in self.config.anchors],
                eyelets=[a.indirect_line.eyelet_pos for a in self.config.anchors],
                tilt=[a.indirect_line.cam_tilt for a in self.config.anchors],
                swing_latency=self.config.swing_latency,
                calibrated=self.config.calibrated_status,
            ))
        else:
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                poses=[a.pose for a in self.config.anchors],
                calibrated=self.config.calibrated_status,
            ))
        if self.config.park_data is not None:
            self.send_ui(named_position=telemetry.NamedObjectPosition(
                name = 'parking_location',
                position = self.config.park_data.pos
            ))
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
        if self.lerobot_process_watcher is None or self.lerobot_process_watcher.done():
            self.last_ep_ctrl_status = common.LerobotStatus.NA
        if isinstance(self.last_ep_ctrl_status, common.LerobotSessionStatus):
            ep_status = self.last_ep_ctrl_status
        else:
            ep_status = common.LerobotSessionStatus(
                status=self.last_ep_ctrl_status,
                policy_repo_id=self.config.last_lerobot_policy,
                dataset_repo_id=self.config.last_lerobot_dataset_repo_id,
            )
        self.send_ui(episode_control=common.EpisodeControl(
            status=ep_status,
            prompt=self.config.last_lerobot_prompt,
        ))
        self.send_ui(task_status=telemetry.TaskStatus(
            route_source=self.pnp_src, route_destination=self.pnp_dst,
        ))
        self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=('swingc' in self.active_set), present='.'))
        self.send_ui(tension_regulation_state=telemetry.TensionRegulationState(enabled=self.tension_reg_enabled, present=True))
        self.send_ui(torque_state=telemetry.TorqueState(enabled=self.torque_enabled, present=True))
        self.send_ui(auto_targeting_state=telemetry.AutoTargetingState(enabled=self.target_model is not None, present=True))
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
        self.input_velocities = {'default': (np.zeros(3), time.monotonic())}

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

        # Lerobot Episode Control (Start/Stop Recording)
        elif item.episode_control is not None:
            self._handle_add_episode_control_events(item.episode_control)

        elif item.scale_room is not None:
            self._handle_scale_room(item.scale_room)

        elif item.add_cam_target is not None:
            self._handle_add_cam_target(item.add_cam_target)

        elif item.add_room_target is not None:
            self._handle_add_room_target(item.add_room_target)

        elif item.delete_target is not None:
            r = await self._handle_delete_target(item.delete_target)

        elif item.debug is not None:
            r = await self._handle_debug_command(item.debug)

        elif item.set_swing_cancellation is not None:
            r = await self._handle_set_swing_cancellation(item.set_swing_cancellation)

        elif item.single_component_action is not None:
            r = await self._handle_single_component_action(item.single_component_action)

        elif item.manage_lerobot_session is not None:
            self.lerobot_process_watcher = asyncio.create_task(self.lerobot_process(item.manage_lerobot_session))

        elif item.move_gripper_to is not None:
            r = await self._handle_move_gripper_to(item.move_gripper_to)

        elif item.set_point is not None:
            asyncio.create_task(self._handle_set_point(item.set_point))

        elif item.set_target_model is not None:
            asyncio.create_task(self._handle_set_target_model(item.set_target_model))

        elif item.popup_ack is not None:
            self._handle_popup_ack(item.popup_ack)

        elif item.add_relay_creds is not None:
            self._handle_add_relay_creds(item.add_relay_creds)

    async def _handle_set_point(self, item: control.SetPoint):
        """Set either the route source or destination (the To: and From: fields in the UI)"""
        logger.debug(f'_handle_set_point {item}')
        if item.route_source:
            self.pnp_src = item.route_source
            self.config.last_route_source = item.route_source
        if item.route_destination:
            self.pnp_dst = item.route_destination
            self.config.last_route_destination = item.route_destination
        self.send_ui(task_status=telemetry.TaskStatus(
            route_source=self.pnp_src, route_destination=self.pnp_dst, 
        ))
        r = await self.flush_tele_buffer()

    async def _handle_move_gripper_to(self, item: control.MoveGripperTo):
        """Handle the Go Here command"""
        goal_pos = None
        if item.target_id is not None:
            # derive target position from target
            target = self.target_queue.get_target_info(item.target_id)
            if target is not None:
                goal_pos = tonp(target.position) + GRIPPER_HEIGHT_OVER_TARGET + self.pole
        elif item.pos is not None:
            goal_pos = tonp(item.pos) + GRIPPER_HEIGHT_OVER_TARGET + self.pole

        if goal_pos is None:
            return
        r = await self.invoke_motion_task(self.seek_goal(goal_pos))

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
                ))

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
            self.active_set.add('swingc')
            while self.run_command_loop:
                if self.gripper_client is None:
                    await asyncio.sleep(1)
                vel2 = self.gripper_client.compute_swing_correction(time.time() + self.config.swing_latency)
                if vel2 is not None:
                    await self.move_direction_speed(np.array([vel2[0], vel2[1], 0]), key='swingc', downward_bias=0)
                await asyncio.sleep(1/100)
        except asyncio.CancelledError:
            pass
        finally:
            self.active_set.remove('swingc')
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

    def _broadcast_swing_latency(self, latency):
        """Set config.swing_latency (in memory) and tell the UI. Does not persist;
        callers save_config only once a value is committed."""
        self.config.swing_latency = float(latency)
        self.send_ui(new_anchor_poses=telemetry.AnchorPoses(swing_latency=self.config.swing_latency))

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
        self.active_set.add('swingc')
        self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=True, present='.'))
        start = time.time()
        aborted = None
        try:
            while (t := time.time() - start) < RUN_PERIODS * self.pendulum.period:
                now = time.time()
                v = gc.compute_swing_correction(now + latency)
                vx, vy = (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0)
                vz = swing.altitude_hold_velocity(center_pos[2] - self.pe.gant_pos[2])
                await self.move_direction_speed(np.array([vx, vy, vz]), key='swingc', downward_bias=0)
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
            self.input_velocities['swingc'] = (np.zeros(3), time.monotonic())
            self.active_set.discard('swingc')
            self.slow_stop_all_spools()
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=False, present='.'))

        return self.pendulum.trial_residual(ts, amps, aborted), aborted

    async def calibrate_swing_latency(self, fine_pass=False, progress_range=None):
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

        0.3 is tried first because it is usually the answer, and a coarse trial that already
        damps well is close enough to refine around directly. Stopping there skips the two
        candidates most likely to pump. Set fine_pass=True to add a second pass that refines
        around the coarse best; the early stop only applies then, since the fine pass is what
        supplies the trials MIN_TRIALS wants.
        """
        DRIFT_LIMIT_M = 0.6          # recenter between trials once drift exceeds this
        MIN_TRIALS = 3               # need at least this many good trials to choose
        TENSION_BACKOFF_S = 1.1      # wait this long after a tension trip before retrying a trial
        MAX_TENSION_RETRIES = 3      # give up (and abort) if a single trial keeps tripping tension

        if self.gripper_client is None:
            logger.warning('Swing latency calibration requires a connected gripper')
            return None

        original_latency = self.config.swing_latency
        center_pos = np.array(self.pe.gant_pos, dtype=float)
        all_results = []      # (latency, residual) from every reliable trial

        async def sweep(cands, stop_below=None):
            out = []
            for idx,lat in enumerate(cands):
                if progress_range is not None:
                    start_pct, end_pct = progress_range
                    pct = start_pct + (end_pct - start_pct) * (idx + 1) / (len(cands) + 1)
                    self.send_ui(operation_progress=telemetry.OperationProgress(
                        percent_complete=pct,
                        name="Calibration",
                        current_action=f"Tuning swing cancellation {idx + 1}/{len(cands)} ({lat})",
                    ))

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
                    if stop_below is not None and residual < stop_below:
                        logger.info(f'swing_latency {lat:.3f}s already damps below {stop_below*1000:.0f} mrad; '
                                    f'skipping the remaining coarse candidates and refining around it')
                        return out
                else:
                    logger.info(f'swing_latency {lat:.3f}s -> unreliable, excluded{tag}')
                await asyncio.sleep(0.3)
            return out

        self.tension_over_limit = False  # clear any stale trip so the first trial isn't cut short
        self.swing_cal_in_progress = True  # let passive_safety recover (not abort) on a tension trip here
        try:
            coarse = await sweep(swing.COARSE_CANDS,
                                 stop_below=swing.COARSE_GOOD_ENOUGH_RAD if fine_pass else None)
            if coarse and fine_pass:
                best_coarse = min(coarse, key=lambda r: r[1])[0]
                # Recenter before the fine pass so the trials we care about start with
                # full drift headroom and don't get cut short.
                await self._recenter_gantry(center_pos)
                await sweep(swing.fine_candidates(best_coarse))
        finally:
            # Do not clear tension_over_limit here: on a max-retry abort it must survive to the
            # calibration's CancelledError handler so it can report the tension reason.
            self.swing_cal_in_progress = False
            self.input_velocities['swingc'] = (np.zeros(3), time.monotonic())
            self.active_set.discard('swingc')
            self.slow_stop_all_spools()
            self.send_ui(swing_cancellation_state=telemetry.SwingCancellationState(enabled=False, present='.'))
            await self._recenter_gantry(center_pos)

        if len(all_results) < MIN_TRIALS:
            logger.warning(f'Swing latency calibration got only {len(all_results)} usable trials; keeping existing value')
            self._broadcast_swing_latency(original_latency)
            return None

        best = swing.select_min_residual(all_results)
        self._broadcast_swing_latency(best)
        save_config(self.config, self.config_path)
        logger.info(f'Calibrated swing_latency = {best:.3f}s')
        return best

    async def _handle_debug_command(self, item: control.Debug):
        logger.debug(f'Debug action "{item.action}"')
        if item.action == "spincal":
            r = await self.calibrate_spin()
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
        if item.action == 'floorplates':
            # Park the gripper over clean, clear floor first; this moves only height and wrist.
            r = await self.invoke_motion_task(self.collect_floorplates())
        if item.action == 'objectplates':
            r = await self.invoke_motion_task(self.collect_objectplates())
        if item.action == 'fingerplates':
            # Park the gripper over clear textured floor before running this; see the
            # docstring for what an unlucky spot does to the matte.
            r = await self.invoke_motion_task(self.collect_fingerplates())
        if item.action == 'stow':
            r = await self.stow_lines()
        if item.action == 'upright':
            r = await self.invoke_motion_task(self.ensure_pole_upright())
        if item.action.startswith('swinglatency '):
            parts = item.action.split(' ')
            self.config.swing_latency = float(parts[1])
            save_config(self.config, self.config_path)
        if item.action == 'swinglatencycal':
            # Run the fine pass and emit progress so the debug-triggered run refines around
            # the coarse best and reports status just like the in-calibration invocation.
            r = await self.invoke_motion_task(self.calibrate_swing_latency(fine_pass=True, progress_range=(0.0, 100.0)))
        if item.action.startswith('polecal'):
            # 'polecal [seconds]' - how long to record the free decay for
            parts = item.action.split()
            decay_s = float(parts[1]) if len(parts) == 2 else None
            r = await self.invoke_motion_task(self.measure_pendulum_length(decay_s))
        if item.action == 'reset_wrist':
             r = await self.gripper_client.send_commands({'reset_wrist': None})
        if item.action == 'spind':
            print(self.gripper_client.get_spin(True))
        if item.action == 'ferry':
            r = await self.invoke_motion_task(self.ferry('hamper', 'trash'))
        if item.action == 'linear':
            r = await self.invoke_motion_task(self.linear_height_check_task())
        if item.action == 'goalseek':
            r = await self.invoke_motion_task(self.goalseek_diagnostic_task())
        if item.action == 'sync_timezone':
            await self.sync_timezone_to_bots()
        if item.action == 'pull_logs':
            asyncio.create_task(self.pull_logs_to_zip())
        if item.action.startswith('untwist'):
            parts = item.action.split()
            if len(parts)==2 and parts[0]=='untwist':
                r = await self.gripper_client.send_commands({'untwist': int(parts[1])})
        if item.action.startswith('setvar '):
            # 'setvar KEY VALUE' broadcasts a live config override to every component.
            # used for bench tuning of onboard loop constants without restarting firmware.
            parts = item.action.split()
            if len(parts) == 3:
                key = parts[1]
                try:
                    value = float(parts[2])
                except ValueError:
                    value = parts[2]
                logger.info(f'Broadcasting set_config_vars {key}={value} to all components')
                await asyncio.gather(*[
                    client.send_commands({'set_config_vars': {key: value}})
                    for client in self.bot_clients.values()
                ])
            else:
                logger.warning(f'invalid setvar command, expected "setvar KEY VALUE": {item.action}')
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
        if item.action == 'centerorigin':
            r = await self.invoke_motion_task(self._center_card_in_view('origin'))
        if item.action == 'servograsp':
            # One visual servoing grasp from wherever the gripper is now, without the
            # pick and place loop. Park it over the object first, roughly - closing the
            # rest is the model's job.
            async def servo_grasp_once():
                if not await self.servo.ensure_model():
                    return
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

    async def chase_tag(self, name):
        """Keep the gripper at the named location"""
        try:
            chase_task = None
            while self.run_command_loop:
                await asyncio.sleep(0.1)
                if not name in self.config.named_positions:
                    continue
                goal = tonp(self.config.named_positions[name]) + self.pole
                if chase_task is None or chase_task.done():
                    chase_task = asyncio.create_task(self.seek_goal(goal))
                else:
                    self.goal_pos = goal # retarget the seek already in flight
        except asyncio.CancelledError:
            if chase_task is not None:
                chase_task.cancel()
            raise

    async def ferry(self, source, dest):
        """Carry objectes between one named tag and another.
        Moves to source, attempt auto grasp, move to test, drop, repeat"""
        try:
            while self.run_command_loop:
                await asyncio.sleep(0.1)

                # wait for source position to be seen
                while not source in self.config.named_positions:
                    await asyncio.sleep(0.5)
                # go to position
                goal = tonp(self.config.named_positions[source]) + self.pole + GRIPPER_HEIGHT_OVER_TARGET
                await self.seek_goal(goal)

                # auto grasp
                # await self.gripper_client.send_commands({'set_finger_angle': 30})
                # await asyncio.sleep(1)
                await self.execute_grasp()

                # wait for destination position to be seen
                while not dest in self.config.named_positions:
                    await asyncio.sleep(0.5)
                # go to position
                goal = tonp(self.config.named_positions[dest]) + self.pole + GRIPPER_HEIGHT_OVER_TARGET
                await self.seek_goal(goal)

                # drop
                await self.gripper_client.send_commands({'set_finger_angle': -30})
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            raise

    async def lerobot_process(self, item: control.ManageLerobotSession):
        if self.lerobot_process_pid is not None:
            logger.warning(f"Cannot start lerobot session, one is already active.")
            return

        repo_id = item.repo_id
        action = item.action
        # Sanitize and validate repo_id to prevent code injection.
        # Enforces the Hugging Face Hub format: 'namespace/dataset_name'
        if not re.match(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$", str(repo_id)):
            logger.warning(f"Invalid repo_id format '{repo_id}'. Expected 'namespace/dataset_name'. Aborting.")
            return

        # Run the python function as a command-line script to hook into its stdout and stderr streams asynchronously and use the same virtualenv
        if action == control.LerobotSessionAction.START_RECORD:
            func_name = 'record_until_disconnected'
            self.config.last_lerobot_dataset_repo_id = repo_id
        elif action == control.LerobotSessionAction.START_EVAL:
            func_name = 'eval_until_disconnected'
            self.config.last_lerobot_policy = repo_id

        up = ''
        if item.suppress_upload:
            up = ' upload=False'

        # A lerobot session running on the local machine must connect to the telemetry socket of the robot.
        # When telemetry_env is not None, there are two options. connect to the remote stream - this introduces needless latency and requires a token
        # Or spin up the local telemetry socket and the MJepeg streamers while the lerobot process is active.
        tele_addr = 'ws://localhost:4245'

        command = [
            sys.executable,
            '-u', '-c',
            f"from nf_robot.ml.stringman_lerobot import {func_name}; "
            f"{func_name}('{tele_addr}', '{repo_id}', '{self.config.robot_id}'{up})"
        ]

        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        logger.info(f"Lerobot process started with PID: {process.pid}")
        self.lerobot_process_pid = process.pid

        async def log_stream(stream, stream_name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                sline = line.decode('utf-8').rstrip()
                if not sline.startswith('[swscaler'):
                    logger.info(f"[{stream_name}] {sline}")

        # Create concurrent background tasks to monitor stdout and stderr
        stdout_task = asyncio.create_task(log_stream(process.stdout, "LEROBOT STDOUT"))
        stderr_task = asyncio.create_task(log_stream(process.stderr, "LEROBOT STDERR"))

        try:
            return_code = await process.wait()
            logger.info(f"Lerobot process exited with code: {return_code}")
            
        except asyncio.CancelledError:
            logger.info("Cancellation requested. Terminating Lerobot process...")
            try:
                process.terminate()
            except ProcessLookupError:
                pass # Process already died
            await process.wait()
            logger.info("Lerobot process terminated.")
            
        finally:
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            self.lerobot_process_pid = None

    async def calibrate_finger_servo(self):
        self.gripper_client.finger_contact_calibration_complete.clear()
        await asyncio.create_task(self.gripper_client.send_commands({'measure_finger_contact': None}))
        await asyncio.wait_for(self.gripper_client.finger_contact_calibration_complete.wait(), 20)

    async def _handle_delete_target(self, item: control.DeleteTarget):
        if item.target_id is not None:
            self.target_queue.remove_target(item.target_id);
        self.send_tq_to_ui()
        await self.flush_tele_buffer()

    def _handle_add_cam_target(self, item: control.AddTargetFromAnchorCam):
        # Add the target
        targets2d = [[item.img_norm_x, item.img_norm_y]]
        if item.anchor_num not in self.anchors:
            return
        floor_points = project_pixels_to_floor(targets2d, self.anchors[item.anchor_num].camera_pose, self.config.camera_cal)
        logger.info(f'Adding target at floor point ({floor_points}) from image point ({targets2d[0]}) in anchor cam {item.anchor_num}')
        if (len(floor_points) == 1):
            if item.target_id is not None:
                self.target_queue.set_target_position(item.target_id, floor_points[0])
            else:   
                new_id = self.target_queue.add_user_target(floor_points[0], dropoff='hamper')
        self.send_tq_to_ui()

    def _handle_add_room_target(self, item: control.AddTargetInRoom):
        # Used when the position arrives already in room coordinates.
        logger.info(f'Adding target at floor point ({item.x}, {item.y}) from the 3d view')
        self.target_queue.add_user_target((item.x, item.y), dropoff='hamper')
        self.send_tq_to_ui()

    def _handle_scale_room(self, item: control.ScaleRoom):
        # not implemented for arpeggio anchor
        if item.scale:
            # move positions of anchors towards or away from origin
            logger.info(f'Scaling by {item.scale}')
            anchor_poses = [(client.anchor_pose[0], client.anchor_pose[1]*item.scale) for client in self.anchors.values()]

            # update everything
            for client in self.anchors.values():
                self.config.anchors[client.anchor_num].pose = poseTupleToProto(anchor_poses[client.anchor_num])
                client.updatePose(anchor_poses[client.anchor_num])
            save_config(self.config, self.config_path)
            # inform UI
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(poses=[
                poseTupleToProto(p)
                for p in anchor_poses
            ]))
            # inform position estimator
            anchor_points = np.array([compose_poses([pose, model_constants.anchor_grommet])[1] for pose in anchor_poses])
            self.pe.set_anchor_points(anchor_points)

        if item.tiltcams:
            logger.info(f'Tilting cams inward by {item.tiltcams} deg')
            for client in self.anchors.values():
                client.extratilt += item.tiltcams
                client.updatePose(client.anchor_pose)


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
                r = await self.invoke_motion_task(self.full_auto_calibration())
            case control.Command.PICK_AND_DROP:
                r = await self.invoke_motion_task(self.pick_and_place_loop())
            case control.Command.HORIZONTAL_CHECK:
                r = await self.invoke_motion_task(self.linear_height_check_task())
            case control.Command.SHUTDOWN:
                self.run_command_loop = False
            case control.Command.RECORD_PARK:
                r = await self.record_park()
            case control.Command.PARK:
                r = await self.invoke_motion_task(self.park())
            case control.Command.UNPARK:
                r = await self.invoke_motion_task(self.unpark())
            case control.Command.GRASP:
                r = await self.invoke_motion_task(self.execute_grasp())
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

    async def passive_safety(self):
        """If any line becomes too tight, switch all motors to damped movement for one second.
        If the overload happens while a motion task is running, abort it by cancelling the
        task, since backing off mid-motion corrupts whatever it was doing. The one exception
        is swing latency cal: it sets swing_cal_in_progress so we only raise the
        tension_over_limit flag, which it polls to back off and retry the current trial."""
        max_safe_tension = 16.0
        if self.config.max_safe_tension is not None:
            max_safe_tension = self.config.max_safe_tension

        ema = np.zeros(4)
        while self.run_command_loop and self.pe.tension is not None:
            ema = ema * 0.9 + self.pe.tension * 0.1
            if np.any(ema > max_safe_tension):
                logger.warning(f'Tension limit reached! backing off. limit={max_safe_tension} actual={ema}')
                if self.motion_task is not None and not self.motion_task.done():
                    self.tension_over_limit = True
                    if not self.swing_cal_in_progress:
                        logger.warning(f'Tension overload during motion task "{self.motion_task.get_name()}" - aborting it')
                        self.motion_task.cancel()
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

    def _report_gantry_marker_fault(self, key, phrase, message, detail):
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
        self.send_ui(pop_message=telemetry.Popup(message=message))
        if (self.motion_task is not None and not self.motion_task.done()
                and self.motion_task.get_name() in MARKER_DEPENDENT_TASKS):
            logger.warning(f'Gantry marker fault during motion task '
                           f'"{self.motion_task.get_name()}" - aborting it')
            self.gantry_marker_fault = phrase
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
        POLL_S = 1.0
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
        UNSEEN_LIMIT_S = 10.0

        await self.any_anchor_connected.wait()
        history = {}                # anchor num -> its recent rows, older than the buffer holds
        newest_per_anchor = {}      # anchor num -> newest capture time already taken from it
        # host clock, so a component whose clock is skewed cannot fake staleness
        last_advance = time.time()

        while self.run_command_loop:
            await asyncio.sleep(POLL_S)
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
            elif not self.anchors:
                # nothing is looking, which is a connection problem and reported as one
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

    async def invoke_motion_task(self, coro):
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

        # If lerobot scripts are connected this must also stop them
        self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.ABANDON))

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

        self.slow_stop_all_spools()

    def slow_stop_all_spools(self):
        for name, client in self.bot_clients.items():
            # Slow stop all spools. gripper too
            asyncio.create_task(client.slow_stop_spool())
        self.pe.record_commanded_vel(np.zeros(3))
        # this stops the spools directly, bypassing move_direction_speed, so the stale
        # 'default' velocity must be cleared here too or it'll get summed back in the
        # next time anything (e.g. swing cancellation) triggers a combined move.
        self.input_velocities['default'] = (np.zeros(3), time.monotonic())

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
                a.save_raw = True

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

        # keep the gripper vertical and the lines taut throughout the survey
        self.set_swing_cancellation(True)

        card_positions = self.card_room_positions()
        if not card_positions:
            logger.warning('No calibration cards visible to the anchor cameras; cannot run gripper card survey')
            return {}

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
            camera_height = await self._trim_altitude_to_range(target_range_m, ceiling_z=upper_z)
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
                        await self._nudge_gantry(np.array([0.0, 0.0, delta_z]), max_step=1.0)
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

        total = sum(len(s) for s in gripper_obs.values())
        logger.info(f'Gripper card survey collected {total} hover samples across {len(gripper_obs)} cards: '
                    f'{ {k: len(v) for k, v in gripper_obs.items()} }')
        return gripper_obs

    async def _nudge_gantry_xy(self, delta_xy, speed=NUDGE_SPEED_MPS):
        """Move the gantry a small horizontal step of (approximately) delta_xy meters, then stop."""
        return await self._nudge_gantry(np.array([delta_xy[0], delta_xy[1], 0.0]), speed=speed)

    async def _nudge_gantry(self, delta, speed=NUDGE_SPEED_MPS, max_step=0.35):
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
        # arrived. Sources sum, so an idle 'default' adds nothing to ours. Re-issue it while
        # the nudge runs, both to stay inside INPUT_VELOCITY_TTL_S and to recompute the line
        # speeds from where the gantry has actually got to.
        end = time.monotonic() + dist / speed
        while True:
            await self.move_direction_speed(uvec * speed, None, self.pe.gant_pos, key=NUDGE_VELOCITY_KEY)
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(NUDGE_REFRESH_S, remaining))
        await self.move_direction_speed(np.zeros(3), 0, key=NUDGE_VELOCITY_KEY)
        self.slow_stop_all_spools()
        await asyncio.sleep(NUDGE_SETTLE_S)
        return time.time()

    async def _trim_altitude_to_range(self, target_range_m, tol_m=0.02, max_steps=4, ceiling_z=None):
        """Close the gantry's altitude onto the height where the downward rangefinder reads
        target_range_m, and report the range finally measured (None if it never got a reading).

        The rangefinder is coplanar with the gripper camera, so its reading is the camera's
        height above whatever is beneath it - the card, once centering has put the gripper over
        it. Seeking to a computed gantry z only lands within GOAL_PROXIMITY_M and inherits any
        bias in the position estimate, so the sampled hover heights are otherwise approximate.
        Measuring the height directly makes them what was asked for.

        Call this with the card already centered, or the beam may be reading the floor beside a
        raised card rather than the card itself."""
        laser_range = None
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
            logger.info(f'Altitude trim: step {step} range {laser_range:.3f}m vs target '
                        f'{target_range_m:.3f}m, moving z by {delta_z:+.3f}m')
            await self._nudge_gantry(np.array([0.0, 0.0, delta_z]), speed=TRIM_SPEED_MPS)
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

    async def _settle_fingers(self, target, tol=2.0, timeout=6.0):
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

    async def collect_fingerplates(self, finger_angles=None, wrist_steps=FINGERPLATE_WRIST_STEPS,
                                   output_dir=PLATE_OUTPUT_DIR, settle_s=FINGERPLATE_SETTLE_S):
        """Capture the frames a finger matte is extracted from, one wrist turn per finger angle.

        Park the gripper over the green backdrop first: the matte is a chroma key, so
        anything ungreen under the fingers comes out as hardware.

        The wrist turn is what makes that robust. The camera is in the palm and turns with
        the wrist, so the fingers stay on the same pixels while the world rotates behind
        them; keying every frame and taking the median leaves anything that passed
        underneath outvoted.

        Only the raw frames are written. Deciding what is finger is an offline judgement
        with thresholds nobody has tuned, and it should be revisable without asking the
        robot to do this again.
        """
        from nf_robot.ml.visual_servoing.plates import PlateWriter, provenance

        if self.gripper_client is None:
            logger.error('No gripper connected; cannot collect fingerplates')
            return None
        if finger_angles is None:
            finger_angles = list(range(FINGERPLATE_ANGLE_MIN, FINGERPLATE_ANGLE_MAX + 1,
                                       FINGERPLATE_ANGLE_STEP))

        start_wrist = self.datastore.winch_line_record.getLast()[1]
        # A full turn has to fit inside the wrist's 0-1080 range without winding the cable
        # up against its limit, so start low enough that base + 360 still fits.
        base_wrist = float(min(max(start_wrist, 0.0), 1080.0 - 360.0))
        wrist_angles = [base_wrist + 360.0 * i / wrist_steps for i in range(wrist_steps)]

        writer = PlateWriter(output_dir, 'fingerplates',
                             notes='wrist turn per finger angle; matte offline by chroma key')
        logger.info(f'Fingerplates: {len(finger_angles)} finger angles x {wrist_steps} wrist '
                    f'steps = {len(finger_angles) * wrist_steps} frames, wrist {base_wrist:.0f}'
                    f'-{base_wrist + 360:.0f}, writing to {output_dir}')

        expect = CAPTURE_RESOLUTION_SIZE
        await self.gripper_client.use_capture_stream()
        try:
            # Hold out for a frame at the capture resolution, not merely a recent one: the
            # old stream keeps delivering for seconds after the new settings are sent, and
            # accepting those would run the whole sweep at 684x384 while reporting success.
            _, probe = await self.gripper_client.capture_raw_frame(
                time.time(), timeout=CAPTURE_STREAM_TIMEOUT_S, expect_size=expect)
            if probe is None:
                logger.error(
                    f'Fingerplates: no {expect[0]}x{expect[1]} frames within '
                    f'{CAPTURE_STREAM_TIMEOUT_S}s of switching to the capture stream. '
                    f'Check the gripper log for rpicam-vid "ERROR: ***" lines - it may not '
                    f'be able to start at this resolution.')
                return None
            logger.info(f'Fingerplates: capture stream up at {probe.shape[1]}x{probe.shape[0]}')

            missed = 0
            for wrist_angle in wrist_angles:
                actual_wrist = await self.settle_wrist(wrist_angle)
                # Telemetry reports the motor arrived before the video shows it: the
                # capture stream's settings put the frames further behind than that.
                await asyncio.sleep(settle_s)
                # Always the same direction, never serpentine. There is enough slop in
                # the finger gearing that the same commanded angle approached from above
                # and from below puts the hardware in visibly different places, which
                # comes out of the matte as doubled fingers.
                for finger_angle in finger_angles:
                    actual_finger = await self._settle_fingers(finger_angle)
                    # await asyncio.sleep(settle_s)
                    after = time.time()
                    timestamp, frame = await self.gripper_client.capture_raw_frame(
                        after, expect_size=expect)
                    if frame is None:
                        missed += 1
                        logger.warning(f'Fingerplates: no frame at finger {finger_angle} '
                                       f'wrist {wrist_angle:.0f} ({missed} in a row)')
                        if missed >= FINGERPLATE_MAX_MISSES:
                            # The stream is gone, not merely late. Continuing means half an
                            # hour of moving the wrist around for nothing.
                            logger.error(f'Fingerplates: {missed} consecutive frames missing; '
                                         f'abandoning the run with {len(writer)} captured')
                            return None
                        continue
                    missed = 0
                    range_ts, laser = self.datastore.range_record.getLast()
                    writer.add(
                        frame, captured_at=timestamp,
                        finger_angle=actual_finger, wrist_angle=actual_wrist,
                        laser_rangefinder=laser if time.time() - range_ts < RANGE_MAX_AGE_S else None,
                        finger_pressure=self.datastore.finger.getLast()[2],
                        commanded_finger_angle=finger_angle, commanded_wrist_angle=wrist_angle,
                    )
                logger.info(f'Fingerplates: wrist {wrist_angle:.0f} done ({len(writer)} frames)')
        finally:
            # the capture stream stays selected for the rest of the session; switching
            # back costs a stream restart and the next plate command would undo it
            await self.settle_wrist(start_wrist)

        return writer.close(
            finger_angles=list(finger_angles), wrist_steps=wrist_steps,
            base_wrist_angle=base_wrist, **provenance(self.config.robot_id),
        )

    async def _sweep_wrist_sampling(self, kind, writer, degrees, speed_dps, extra,
                                    timeout_margin=1.5):
        """Turn the wrist steadily through `degrees`, sampling telemetry as it goes.

        The frames themselves are being recorded as video by the client; what this adds
        is the state track they get matched against. The speed command is repeated
        because the gripper zeroes it after ACTION_TIMEOUT, which is also what stops the
        wrist if this is cancelled.
        """
        client = self.gripper_client
        start = self.datastore.winch_line_record.getLast()[1]
        direction = 1.0 if degrees >= 0 else -1.0
        deadline = time.time() + abs(degrees) / speed_dps + timeout_margin
        next_command = 0.0
        samples = 0

        try:
            while time.time() < deadline:
                now = time.time()
                if now >= next_command:
                    await client.send_commands({'set_wrist_speed': direction * speed_dps})
                    next_command = now + WRIST_SPEED_REFRESH_S

                range_ts, laser = self.datastore.range_record.getLast()
                fresh = time.time() - range_ts < RANGE_MAX_AGE_S
                writer.add_telemetry(
                    captured_at=time.time(),
                    wrist_angle=self.datastore.winch_line_record.getLast()[1],
                    finger_angle=self.datastore.finger.getLast()[1],
                    finger_pressure=self.datastore.finger.getLast()[2],
                    laser_rangefinder=laser if fresh else None,
                    **extra,
                )
                samples += 1

                travelled = (self.datastore.winch_line_record.getLast()[1] - start) * direction
                if travelled >= abs(degrees):
                    break
                await asyncio.sleep(TELEMETRY_SAMPLE_S)
        finally:
            await client.send_commands({'set_wrist_speed': 0.0})

        actual = self.datastore.winch_line_record.getLast()[1]
        logger.info(f'{kind}: swept wrist {start:.0f} -> {actual:.0f} '
                    f'({samples} telemetry samples at {speed_dps:.0f} deg/s)')
        return samples

    async def _height_wrist_sweep(self, kind, ranges, output_dir, settle_s,
                                  notes='', run_attrs=None, frame_attrs=None,
                                  speed_dps=PLATE_WRIST_SPEED_DPS,
                                  degrees=PLATE_SWEEP_DEGREES):
        """Frames at each of several heights, sweeping the wrist through a circle at each.

        The shape floorplates and objectplates share. Heights are reached by trimming to
        a measured rangefinder reading, so what each plate records is how far away its
        subject actually was. The fingers are parked out of frame first, since hardware
        in the corner of a plate would be composited into every frame built from it.
        """
        from nf_robot.ml.visual_servoing.plates import VideoRunWriter, provenance

        if self.gripper_client is None:
            logger.error(f'No gripper connected; cannot collect {kind}')
            return None

        start_wrist = self.datastore.winch_line_record.getLast()[1]
        # start low enough in the wrist's 0-1080 range that a full sweep fits
        base_wrist = float(min(max(start_wrist, 0.0), 1080.0 - abs(degrees)))

        writer = VideoRunWriter(output_dir, kind, notes=notes)
        seconds = len(ranges) * abs(degrees) / speed_dps
        logger.info(f'{kind}: {len(ranges)} heights, {degrees:.0f} deg at {speed_dps:.0f} '
                    f'deg/s each, about {seconds / 60:.0f} min of sweeping, '
                    f'writing to {output_dir}')

        expect = CAPTURE_RESOLUTION_SIZE
        client = self.gripper_client
        await client.use_capture_stream()
        try:
            _, probe = await client.capture_raw_frame(
                time.time(), timeout=CAPTURE_STREAM_TIMEOUT_S, expect_size=expect)
            if probe is None:
                logger.error(f'{kind}: no {expect[0]}x{expect[1]} frames within '
                             f'{CAPTURE_STREAM_TIMEOUT_S}s of switching to the capture '
                             f'stream. Check the gripper log for rpicam-vid errors.')
                return None
            logger.info(f'{kind}: capture stream up at {probe.shape[1]}x{probe.shape[0]}')

            await self._settle_fingers(PLATE_FINGERS_RETRACTED)
            await self.settle_wrist(base_wrist)

            client.recording_path = writer.video_path
            heading = 1.0
            for target_range in ranges:
                reached = await self._trim_altitude_to_range(target_range)
                if reached is None:
                    logger.warning(f'{kind}: no rangefinder reading at target '
                                   f'{target_range:.2f}m; skipping this height')
                    continue
                await asyncio.sleep(settle_s)
                await self._sweep_wrist_sampling(
                    kind, writer, heading * degrees, speed_dps,
                    extra={'target_range_m': target_range,
                           'start_wrist_angle': start_wrist,
                           **(frame_attrs or {})})
                heading = -heading
                logger.info(f'{kind}: range {target_range:.2f}m done '
                            f'({client.recorded_packets} packets recorded)')
        finally:
            packets, stream_start_ts = client.recorded_packets, client.recording_stream_start_ts
            client.recording_path = None
            # the demux loop closes the file when it next sees a packet
            await asyncio.sleep(RECORDING_CLOSE_S)
            # the capture stream stays selected for the rest of the session; see
            # collect_fingerplates
            await self.settle_wrist(start_wrist)

        return writer.close(stream_start_ts or 0.0, packets=packets,
                            target_ranges=list(ranges), sweep_degrees=degrees,
                            sweep_speed_dps=speed_dps, start_wrist_angle=start_wrist,
                            **provenance(self.config.robot_id), **(run_attrs or {}))

    async def collect_floorplates(self, ranges=None, output_dir=PLATE_OUTPUT_DIR,
                                  settle_s=FINGERPLATE_SETTLE_S):
        """Capture bare floor at a range of heights, for synthetic backgrounds.

        The operator flies the gripper somewhere clean and clear and only then triggers
        this; it moves nothing but height and wrist. An autonomous room sweep would come
        back with a library of beds, furniture and feet, none of which is a floor plate.
        """
        return await self._height_wrist_sweep(
            'floorplates', ranges or PLATE_RANGES_M, output_dir, settle_s,
            notes='bare floor at several heights; fingers retracted')

    async def collect_objectplates(self, ranges=None,
                                   output_dir=PLATE_OUTPUT_DIR,
                                   settle_s=FINGERPLATE_SETTLE_S):
        """Capture one object on the green board, for compositing onto floor plates.

        Two things the operator sets before triggering this, and both are labels rather
        than settings: the object's intended grasp point goes under the camera, which
        makes the grasp point the principal point by construction, and the wrist is
        turned to the ideal grasping angle, which makes the grasp axis zero at the start
        and a known offset at every later frame. Neither needs marks on the board.
        """
        label = f'object-{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:4]}'
        start_wrist = self.datastore.winch_line_record.getLast()[1]
        logger.info(f'objectplates: labelling this object {label}')
        return await self._height_wrist_sweep(
            'objectplates', ranges or PLATE_RANGES_M, output_dir, settle_s,
            notes=f'object on green board: {label}',
            run_attrs={'label': label, 'grasp_axis_wrist_angle': start_wrist},
            frame_attrs={'label': label})
        self.send_ui(pop_message=telemetry.Popup(
            message='Objectplate capture complete.'
        ))

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

            tilts = (self.config.anchors[0].indirect_line.cam_tilt, self.config.anchors[1].indirect_line.cam_tilt)
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
            await finger_task
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': -40}))

            # move over the origin card
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=27.0,
                name="Calibration",
                current_action="Moving gripper to origin",
            ))
            gant_z = min(upper_z-0.1, self.pole[2] + 0.8 - floor_z)
            await self.seek_goal(np.array([0,0,gant_z]), head_turn=False)

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=29.0,
                name="Calibration",
                current_action="Measuring spin. Gripper camera must see origin card to complete this step.",
            ))
            # there should be some swing when we get there. 
            await self.half_auto_calibration()
            await self._center_card_in_view('origin')

            # roomspin
            await self.calibrate_spin(reset_wrist_first=False) # already did that during diamond to save time

            # Tune swing_latency by inducing swings and finding the value that damps
            # them best. Requires a connected gripper (IMU-driven swing model).
            if self.gripper_client is not None:
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=34.0,
                    name="Calibration",
                    current_action="Tuning swing cancellation",
                ))
                # Perform swing cancellation measurements lower than the spin-measurement
                SWING_MEASURE_DROP_M = 0.4
                await self.seek_goal(np.array([0, 0, gant_z - SWING_MEASURE_DROP_M]), head_turn=False)
                await self.calibrate_swing_latency(fine_pass=True, progress_range=(30.0, 61.0))

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
                    if refined_anchors is not None:
                        anchor_poses, eyelet_positions, floor_z = refined_anchors, refined_eyelets, refined_floor_z
                        logger.info(f'Refined with gripper card views:\nanchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')
                        self.save_poses_arp(anchor_poses, eyelet_positions)
                    else:
                        logger.warning('Gripper-card refinement optimization failed; keeping previous geometry')

                    # Re-enable swing cancellation only if it still damps a test swing with the new
                    # geometry. _measure_swing_residual induces a swing, runs cancellation, and
                    # reports the leftover swing (or the safety cap / no reading if it pumped or
                    # drifted). Anything that isn't a clearly-damped low residual leaves it OFF.
                    self.send_ui(operation_progress=telemetry.OperationProgress(
                        percent_complete=99.0,
                        name="Calibration",
                        current_action="Verifying swing cancellation is safe",
                    ))
                    DAMPING_RESIDUAL_MAX_RAD = 0.15  # a settled swing sits well below this; pumping hits the cap
                    center_pos = np.array(self.pe.gant_pos, dtype=float)
                    residual, aborted = await self._measure_swing_residual(self.config.swing_latency, center_pos)
                    if residual is not None and residual < DAMPING_RESIDUAL_MAX_RAD:
                        logger.info(f'Swing cancellation damps with refined geometry (residual {np.degrees(residual):.1f} deg); enabling.')
                        self.set_swing_cancellation(True)
                    else:
                        detail = aborted or (f'{np.degrees(residual):.1f} deg residual' if residual is not None else 'no reading')
                        logger.warning(f'Swing cancellation did not damp with refined geometry ({detail}); leaving it OFF.')
                        self.send_ui(pop_message=telemetry.Popup(
                            message='Swing cancellation did not damp after calibration refinement and was left OFF. Re-check the calibration before running.'))
                else:
                    logger.warning(f'Only {len(gripper_obs)} of {REQUIRED_GRIPPER_CARDS} gripper card observations; need all four to refine. Skipping 3rd pass.')
                    # geometry is unchanged, so no damping re-test is needed to restore it
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

    async def calibrate_spin(self, reset_wrist_first=True):
        """Calibration of the relationship between the wrist and the room frame of reference.
        Must be done over the origin card.
        """
        if self.gripper_client.last_output_frame is None:
            logger.warning('Cannot calibrate the relationship between gripper zero angle and camera if gripper camera is offline!')
            return None

        # record the z rotation of the gantry card from the perspective of the gripper camera,
        # with no existing z rotation term applied
        self.gripper_client.calibrating_room_spin = True

        if self.gripper_client is not None:
            # measurement must be taken at the wrist's zero point
            center_angle = 540
            if reset_wrist_first:
                asyncio.create_task(self.gripper_client.send_commands({'reset_wrist': None}))
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
                    (self.gripper_client.last_output_frame, self.config.camera_cal_wide),
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
        self.gripper_client.calibrating_room_spin = False

    async def linear_height_check_task(self):
        """
        Measure the average deviation from an ideal constant height, as reported by the
        laser rangefinder, while traversing the floor along the currently selected route.
        Triggered by the debug command "linear". This is a motion task.

        Every room is different and only the operator can pick a path across the floor with
        no obstructions, so the traverse runs between the route source and destination
        (self.pnp_src -> self.pnp_dst), both at 1.5m altitude. The gantry flies directly to
        the source, pauses for 2 seconds, then traverses to the
        destination. Through an ideal move the laser should read (1.5 - self.pole - laser_offset)
        the whole way. Aborts if the laser altitude drops below 0.2m or if the gantry comes
        within 0.4m of the ceiling (the z position of anchor 0).
        """
        TEST_ALTITUDE_M = 1.5
        MIN_LASER_ALTITUDE_M = 0.2
        CEILING_MARGIN_M = 0.4
        SAMPLE_INTERVAL_S = 0.02
        ideal_laser_range = TEST_ALTITUDE_M - self.pole[2] - model_constants.laser_offset

        # ceiling height for the proximity abort
        ceiling_z = self.pe.anchor_points[0][2]

        # Resolve the route endpoints to floor positions chosen by the operator.
        def route_point_floor_pos(route_point, label):
            if route_point in ROUTE_POINT_TAG_NAMES:
                name = ROUTE_POINT_TAG_NAMES[route_point]
                if name not in self.config.named_positions:
                    logger.warning(f'Linear height check: no saved position for {label} tag "{name}"')
                    return None
                return tonp(self.config.named_positions[name])
            if route_point == common.RoutePoint.ORIGIN:
                return np.zeros(3)
            logger.warning(f'Linear height check needs the {label} to be a tag or the origin, not {route_point}')
            return None

        src_pos = route_point_floor_pos(self.pnp_src, 'route source')
        dst_pos = route_point_floor_pos(self.pnp_dst, 'route destination')
        if src_pos is None or dst_pos is None:
            return
        point_a = np.array([src_pos[0], src_pos[1], TEST_ALTITUDE_M])
        point_b = np.array([dst_pos[0], dst_pos[1], TEST_ALTITUDE_M])

        # Fly directly to the route source with auto altitude, then pause before the test.
        await self.seek_goal(point_a, auto_altitude=True)
        await asyncio.sleep(2.0)

        # Traverse to the route destination, sampling the laser the whole way.
        # disable altitude cruise during test
        deviations = []
        aborted = None
        move_task = asyncio.create_task(self.seek_goal(point_b, auto_altitude=False))
        try:
            while not move_task.done():
                await asyncio.sleep(SAMPLE_INTERVAL_S)
                laser_range = self.datastore.range_record.getLast()[1]
                gant_z = self.pe.gant_pos[2]
                if laser_range < MIN_LASER_ALTITUDE_M:
                    aborted = f'laser altitude {laser_range:.3f}m dropped below {MIN_LASER_ALTITUDE_M}m'
                    break
                if ceiling_z - gant_z < CEILING_MARGIN_M:
                    aborted = (f'gantry came within {CEILING_MARGIN_M}m of the ceiling '
                               f'(gantry z={gant_z:.3f}m, ceiling z={ceiling_z:.3f}m)')
                    break
                deviations.append(laser_range - ideal_laser_range)
        finally:
            move_task.cancel()
            try:
                await move_task
            except asyncio.CancelledError:
                pass
            self.slow_stop_all_spools()

        if aborted is not None:
            logger.warning(f'Linear height check aborted: {aborted}')
            return

        if not deviations:
            logger.warning('Linear height check collected no laser samples')
            return

        deviations_cm = np.array(deviations) * 100
        result_message = (
            f'Linear height check complete over {len(deviations_cm)} samples. '
            f'Ideal laser range {ideal_laser_range * 100:.1f}cm. '
            f'Mean deviation {deviations_cm.mean():+.2f}cm, '
            f'mean abs deviation {np.abs(deviations_cm).mean():.2f}cm, '
            f'RMS {np.sqrt((deviations_cm ** 2).mean()):.2f}cm, '
            f'min {deviations_cm.min():+.2f}cm, max {deviations_cm.max():+.2f}cm')
        logger.info(result_message)
        self.send_ui(pop_message=telemetry.Popup(message=f'RMS {np.sqrt((deviations_cm ** 2).mean()):.2f}cm'))

    async def goalseek_diagnostic_task(self):
        """
        Measure how accurately seek_goal parks the gripper over a route-point tag.
        Triggered by the debug command "goalseek". This is a motion task.

        Cycles through the four floor tags ("gamepad", "trash", "hamper", "toys"),
        goal-seeking to each one's saved position in turn until every tag has been visited
        VISITS_PER_TAG times. Flying between tags naturally provides varied approaches, so
        no random points are generated and the operator is never prompted to move anything.
        Once parked over a tag, read where it appears in the gripper camera and compare it
        against the ideal pose it would have if it were directly under the gripper at the
        correct altitude. The RMS of those deviations across all trials is reported in cm.
        """
        TAG_CYCLE = ['gamepad', 'trash', 'hamper', 'toys']
        VISITS_PER_TAG = 3
        SETTLE_S = 2.0           # let the gripper swing settle before measuring
        MEASURE_WINDOW_S = 50.0   # average tag readings over this window
        MEASURE_TIMEOUT_S = 5.0  # give up on a trial if the tag isn't seen in this long

        GANTRY_HEIGHT_OVER_TARGET = 0.9

        # TODO(nathaniel): the gripper camera is tilted, so when the gripper is centered
        # over the tag at the correct altitude the tag does not appear straight down.
        IDEAL_TAG_POSITION_IN_CAMERA = np.array([0.0, 0.03, GANTRY_HEIGHT_OVER_TARGET])

        async def measure_tag_position(tag_name):
            """Average the tag position seen in the gripper camera over a short window."""
            start = time.time()
            deadline = start + MEASURE_TIMEOUT_S
            while time.time() < deadline:
                samples = self.gripper_client.get_route_tag_samples(tag_name, since=start)
                if samples and time.time() - samples[0][0] >= MEASURE_WINDOW_S:
                    break
                await asyncio.sleep(0.1)

            samples = self.gripper_client.get_route_tag_samples(tag_name, since=start)
            if not samples:
                return None
            return np.mean([np.array(pose[1]) for _, pose in samples], axis=0)

        # the order of visits: each tag VISITS_PER_TAG times, cycling through the list
        visit_order = TAG_CYCLE * VISITS_PER_TAG
        num_trials = len(visit_order)

        deviations = []
        for trial, tag_name in enumerate(visit_order):
            if tag_name not in self.config.named_positions:
                logger.warning(f'Goalseek trial {trial + 1}: no saved position for tag "{tag_name}", skipping')
                continue

            logger.info(f'Goalseek trial {trial + 1}/{num_trials}: seeking to tag "{tag_name}"')

            # goal-seek to the tag's saved position
            goal_pos = tonp(self.config.named_positions[tag_name]) + np.array([0,0,GANTRY_HEIGHT_OVER_TARGET])
            await self.seek_goal(goal_pos, auto_altitude=True)
            await asyncio.sleep(SETTLE_S)

            observed = await measure_tag_position(tag_name)
            if observed is None:
                logger.warning(f'Goalseek trial {trial + 1}: tag "{tag_name}" not seen in gripper camera, skipping')
                continue
            deviation = observed - IDEAL_TAG_POSITION_IN_CAMERA
            logger.info(f'Goalseek trial {trial + 1}: "{tag_name}" deviation {deviation * 100}cm '
                        f'(magnitude {np.linalg.norm(deviation) * 100:.2f}cm)\nobserved={observed}')
            deviations.append(deviation)

        if not deviations:
            logger.warning('Goalseek diagnostic collected no measurements')
            return

        deviations = np.array(deviations)
        magnitudes_cm = np.linalg.norm(deviations, axis=1) * 100
        rms_cm = np.sqrt((magnitudes_cm ** 2).mean())
        per_axis_rms_cm = np.sqrt((deviations ** 2).mean(axis=0)) * 100
        logger.info(
            f'Goalseek diagnostic complete over {len(deviations)} trials. '
            f'RMS deviation {rms_cm:.2f}cm '
            f'(per-axis x={per_axis_rms_cm[0]:.2f}cm y={per_axis_rms_cm[1]:.2f}cm z={per_axis_rms_cm[2]:.2f}cm)')

    async def record_park(self):
        """Record that the current location is reseted in the parking saddle and save in the config"""
        # confirm we can actually see the parking target in the grip camera
        if self.gripper_client.park_pose_relative_to_camera is not None:
            self.config.park_data.pos = fromnp(self.pe.gant_pos)

            # save marker pose in rested position
            self.config.park_data.marker_resting = poseTupleToProto(self.gripper_client.park_pose_relative_to_camera)

            # move up 10cm
            await self.move_direction_speed(np.array([0, 0, 0.1]))
            await asyncio.sleep(1.0)
            self.slow_stop_all_spools()
            await asyncio.sleep(1.0)

            # save marker pose while 10cm over target
            self.config.park_data.marker_over = poseTupleToProto(self.gripper_client.park_pose_relative_to_camera)

            # move down 10cm
            await self.move_direction_speed(np.array([0, 0, -0.1]))
            await asyncio.sleep(1.0)
            self.slow_stop_all_spools()
            await asyncio.sleep(1.0)

            save_config(self.config, self.config_path)
            self.send_ui(named_position=telemetry.NamedObjectPosition(
                name = 'parking_location',
                position = self.config.park_data.pos
            ))
            self.send_ui(pop_message=telemetry.Popup(
                message=f'Saved parking location as {self.config.park_data.pos}'
            ))
        else:
            self.send_ui(pop_message=telemetry.Popup(
                message=f'Cannot save location here. The parking marker is not in view of the gripper camera.'
            ))


    async def park(self):
        """ Park on the parking hook for safe power down. """
        FINGER_ANGLE_FOR_CLEAR_VIEW = -30
        STAGING_HOR_OFFSET_M = 0.2
        STAGING_VER_OFFSET_M = 0.0
        LOOK_FOR_MARKER_INITIAL_S = 2.0
        HOMING_TIME_S = 16.0
        MARKER_DIST_CLOSE_ENOUGH = 0.16
        HOMING_SPEED_MPS = 0.02
        HOMING_LOOP_DELAY = 0.1

        try:
            # TODO check if holding something, if so warn user and do not proceed.

            # perform half cal.

            # open gripper
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': FINGER_ANGLE_FOR_CLEAR_VIEW}))

            # move to position above and in front of saddle,
            parkpos = tonp(self.config.park_data.pos)
            away = get_inward_wall_normal(parkpos, self.pe.anchor_points) * STAGING_HOR_OFFSET_M
            await self.seek_goal(parkpos + np.array([away[0], away[1], STAGING_VER_OFFSET_M]))

            # TODO rotate to face wall because camera is under nose and it lets us see a little further.

            # use observed position of park marker to adjust slowly towards
            # the park-over position
            park_over_pose = poseProtoToTuple(self.config.park_data.marker_over)
            over = park_over_pose[1]


            pos = None
            timeout = time.time()+LOOK_FOR_MARKER_INITIAL_S
            while time.time() < timeout:
                try:
                    pos = self.gripper_client.park_pose_relative_to_camera[1]
                    direction = pos - over
                    break
                except TypeError:
                    continue
            if pos is None:
                logger.warning("Can't see parking tag right now")
                return

            timeout = time.time()+HOMING_TIME_S
            while np.linalg.norm(direction) > MARKER_DIST_CLOSE_ENOUGH  and time.time() < timeout:
                move = np.array([direction[1], direction[0], 0])
                await self.move_direction_speed(move, HOMING_SPEED_MPS)
                logger.debug(f'Distance {np.linalg.norm(direction)} and moving {move}')
                await asyncio.sleep(HOMING_LOOP_DELAY)
                try:
                    pos = self.gripper_client.park_pose_relative_to_camera[1]
                    direction = pos - over
                except TypeError:
                    pass
                
            self.slow_stop_all_spools()

            # move down 20cm
            # TODO or until any two lines become slack
            # or until laser range reaches same distance recorded during set park
            await self.move_direction_speed(np.array([0, 0, -0.1]))
            await asyncio.sleep(2.0)
            self.slow_stop_all_spools()

            # for looks, as well as to let me know it finished.
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': 10}))

        except asyncio.CancelledError:
            logger.info('Park cancelled')
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_goal()


    async def unpark(self):
        """ Unpark from the saddle and move clear of it. """
        try:
            # assume gantry position based on parking location since we probably can't see it
            parkpos = tonp(self.config.park_data.pos)
            self.pe.kf.reset_biases(parkpos)
            # move up 10cm
            await self.move_direction_speed(np.array([0, 0, 0.1]))
            await asyncio.sleep(1.0)
            # move directly away from the wall.
            away = get_inward_wall_normal(parkpos, self.pe.anchor_points)
            await self.move_direction_speed(np.array([away[0], away[1], 0]), 0.15)
            await asyncio.sleep(2.0)
            # move towards center of room.
            task = asyncio.create_task(self.seek_goal(np.array([0,0,1])))
            # but don't go all the way, just stop after a bit
            await asyncio.sleep(5.0)
            await self.clear_goal()
            await self.half_auto_calibration()
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_goal()

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
        """A sequence of actions to run when all components are discovered."""
        # wait for event
        await event.wait()

        # unpark if we were parked.
        r = await self.unpark()
        # start pick_and_place_loop
        r = await self.pick_and_place_loop()
        # pick and place finishes if no targets appear during a timeout
        # park robot
        r = await self.park()
        # disconnect all components and set flag that they should not reconnect unless control input is received.

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
        then tell the manager to (re)connect with them right away."""
        host = self.telemetry.control_plane_host
        logger.info(f'Storing relay credentials for {host} (robot id "{item.robot_id}")')
        self.config.relay_credentials[host] = common.RelayCreds(robot_id=item.robot_id, key=item.key)
        save_config(self.config, self.config_path)
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
        # remember the latest lerobot status here; send_setup_telemetry replays it to peers
        # that connect later, and the manager only handles transport.
        status = getattr(kwargs.get('episode_control'), 'status', None)
        if status is not None:
            self.last_ep_ctrl_status = status
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
        self.startup_complete.clear()
        if self.debug:
            from nf_robot.host.loop_monitor import LoopMonitor
            self.loop_monitor = LoopMonitor(interval=0.5, threshold=0.2)
            self.loop_monitor.start()

        self.passive_safety_task = asyncio.create_task(self.passive_safety())
        self.gantry_visibility_task = asyncio.create_task(self.monitor_gantry_visibility())

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
        if self.lerobot_process_watcher is not None:
            self.lerobot_process_watcher.cancel()
            tasks.append(self.lerobot_process_watcher)
        if self.perception_task is not None:
            self.perception_task.cancel()
            tasks.append(self.perception_task)
        if self.passive_safety_task is not None:
            self.passive_safety_task.cancel()
            tasks.append(self.passive_safety_task)
        if self.gantry_visibility_task is not None:
            self.gantry_visibility_task.cancel()
            tasks.append(self.gantry_visibility_task)

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
        self.goal_pos = None
        self.send_ui(named_position=telemetry.NamedObjectPosition(name='gantry_goal_marker')) # not setting position causes it to be hidden

    async def seek_goal(self, goal_pos, head_turn=False, auto_altitude=True):
        """
        Fly the gantry to goal_pos, using the constantly updating gantry position provided
        by the position estimator.

        goal_pos is where the GANTRY goes, not the gripper. The gripper hangs self.pole
        below it, so a caller aiming the gripper at something must add self.pole to the goal.

        The goal is also published as self.goal_pos so it can be steered while in flight:
        assigning self.goal_pos retargets a running seek, and clear_goal() ends it.
        This is a motion task.
        when head_turn, turn gripper to face direction of motion.
        when auto_altitude, room traversal is performed at an ideal altitude
        """
        GOAL_PROXIMITY_M = 0.08
        MAX_SPEED = 0.4 # GANTRY_SPEED_MPS
        ACCEL = 0.15     # m/s^2
        LOOP_SLEEP_S = 0.1
        IDEAL_GANTRY_ALTITUDE = 1.3 # meters. ideal gantry height for room traversal
        CLIMB_RATE = 0.15 # m/s, constant rate of altitude change for auto_altitude
        ALTITUDE_DEADBAND_M = 0.05 # meters, tolerance to avoid hunting around target altitude

        if goal_pos is None:
            return
        self.goal_pos = np.asarray(goal_pos, dtype=float)

        # Calculate the distance needed to stop from MAX_SPEED: d = v^2 / (2a)
        braking_distance = (MAX_SPEED**2) / (2 * ACCEL)
        current_speed = 0.0
        final_approach = False # latches once True so the altitude target doesn't flip back to cruise
        
        try:
            self.send_ui(named_position=telemetry.NamedObjectPosition(position=fromnp(self.goal_pos), name='gantry_goal_marker'))
            dist_to_goal = 10
            while self.goal_pos is not None:
                vector = self.goal_pos - self.pe.gant_pos
                dist_to_goal = np.linalg.norm(vector)

                if dist_to_goal < GOAL_PROXIMITY_M:
                    break

                # Ramp down as the goal approaches: v = sqrt(2 * a * d).
                ramp_dist_to_goal = np.linalg.norm(vector[:2]) if auto_altitude else dist_to_goal
                speed_ramp_down = np.sqrt(2 * ACCEL * ramp_dist_to_goal)

                # Target speed is the ramp-down limit or the max allowable speed
                target_speed = min(speed_ramp_down, MAX_SPEED)

                # Smoothly interpolate current_speed toward target_speed to prevent
                # instantaneous velocity jumps between loop iterations
                step = ACCEL * LOOP_SLEEP_S
                if current_speed < target_speed:
                    current_speed = min(current_speed + step, target_speed)
                else:
                    current_speed = max(current_speed - step, target_speed)

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
                    goal_altitude = self.goal_pos[2]
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

            logger.info(f'Goal reached {tuple(self.goal_pos)}')
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

    async def move_direction_speed(self, uvec, speed=None, starting_pos=None, downward_bias=-0.04, key='default'):
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
        if speed < 0.005:
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
            key = 'default'
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

    def _handle_add_episode_control_events(self, data: common.EpisodeControl):
        if data.prompt:
            self.config.last_lerobot_prompt = data.prompt
        # A status here means some lerobot session is alive and answering, wherever it's connected.
        if data.status is not None:
            self.lerobot_session_status_event.set()
        # forward episode control events back to all telemetry listeners
        self.send_ui(episode_control=data)
        asyncio.create_task(self.flush_tele_buffer())

    def send_tq_to_ui(self):
        snapshot = self.target_queue.get_queue_snapshot()
        # Create a deterministic hash
        current_hash = hash(bytes(snapshot))
        if current_hash != self.last_snapshot_hash:
            self.send_ui(target_list=snapshot)
            self.last_snapshot_hash = current_hash

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
        Orthographic floor projection and target inference.
        The target model is loaded at runtime via SetTargetModel control messages, and reads
        the floor projection the ortho worker renders, so run_ortho must be on for it to see
        anything.
        """
        LOOP_DELAY = 0.1
        FIND_TARGETS_EVERY = 5

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
                stream_path=f'stringman/{self.config.robot_id}/3',
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

        counter = 0
        while self.run_command_loop:
            await asyncio.sleep(LOOP_DELAY)
            if self.target_model is None:
                continue
            counter += 1
            if counter < FIND_TARGETS_EVERY:
                continue
            counter = 0

            floor_targets = await self._find_targets_ortho()

            # None means "no opinion this round" (no input frame yet), which must not be
            # confused with the empty list, which retires every AI target in the queue.
            if floor_targets is None:
                continue
            floor_targets = self._reject_targets_at_dropoff(floor_targets)
            self.target_queue.add_ai_targets(floor_targets)
            self.send_tq_to_ui()

        if self.run_ortho:
            ortho_floor_vs.stop()

    def _route_dst_floor_pos(self):
        """Floor position of the current route destination, or None if it has none.

        Quiet about failures: this is consulted every targeting round, and NA (drop where
        each target says) genuinely has no single destination.
        """
        if self.pnp_dst == common.RoutePoint.ORIGIN:
            return np.zeros(3)
        name = ROUTE_POINT_TAG_NAMES.get(self.pnp_dst)
        if name is None or name not in self.config.named_positions:
            return None
        return tonp(self.config.named_positions[name])

    def _reject_targets_at_dropoff(self, targets):
        """Drop targets sitting on the route destination, whatever model proposed them.

        This is to prevent the robot from repeatedly picking and dropping the same thing forver.
        """
        DROPOFF_EXCLUSION_M = 0.10

        dst = self._route_dst_floor_pos()
        if dst is None:
            return targets
        kept = []
        for t in targets:
            # Horizontal distance only
            if np.linalg.norm(np.asarray(t['position'])[:2] - dst[:2]) < DROPOFF_EXCLUSION_M:
                logger.debug(f'discarding target at {t["position"]}, inside the dropoff exclusion')
                continue
            kept.append(t)
        return kept

    def _floor_target(self, x, y):
        """A target dict for the queue, or None if it lies outside the work area."""
        position = np.array([x, y, 0])
        if not self.pe.point_inside_work_area_2d(position[:2]):
            return None
        return {'position': position, 'dropoff': 'hamper'}

    async def _find_targets_ortho(self):
        """Every confident target in the ortho floor view, per the ortho_target model.

        The model reads the same projection the ortho worker already renders, so nothing
        per-camera is inferred and no warping is needed.
        """
        from nf_robot.ml import ortho_target

        # Scores have no absolute scale - the softmax mass is split across every cell and
        # split again by every object in frame - so a second target is recognised by being
        # a rival to the best peak, not by clearing a fixed bar. Logged crowded frames put
        # the real objects at 0.020-0.05 of the winner and the noise at 0.004, so the
        # ratio sits between with roughly 2x margin either way. Chance is only the "is
        # anything here" bar, low enough that a lone object never trips it.
        ORTHO_SCORE_RATIO = 0.038
        ORTHO_SCORE_OVER_CHANCE = 4.0
        ORTHO_MAX_CANDIDATES = 16  # NMS peaks to consider before thresholding

        ortho_frame = self.last_ortho_rgb
        if ortho_frame is None:
            if not self.run_ortho:
                logger.warning('ortho target model needs the floor projection, which run_ortho disables')
            return None

        predictions = await asyncio.to_thread(
            partial(ortho_target.predict_room_targets, self.target_model, ortho_frame, self._device,
                    top_k=ORTHO_MAX_CANDIDATES, min_score_over_chance=ORTHO_SCORE_OVER_CHANCE,
                    min_score_ratio=ORTHO_SCORE_RATIO),
        )
        targets = [self._floor_target(x, y) for x, y, _ in predictions]
        return [t for t in targets if t is not None]

    async def pick_and_place_loop(self):
        """
        Long running motion task that repeatedly identifies targets picks them up and drops them over the hamper
        """
        ppc = self.config.pick_and_place
        GANTRY_HEIGHT_OVER_TARGET = tonp(ppc.gantry_height_over_target)
        GANTRY_HEIGHT_OVER_DROPOFF = tonp(ppc.gantry_height_over_dropoff)
        RELAXED_OPEN = ppc.relaxed_open # Open enough to drop and that fingers cannot be seen in frame
        DELAY_AFTER_DROP = ppc.delay_after_drop # long enough that the payload is not visible anymore in the hand
        LOOP_DELAY = ppc.loop_delay
        END_LOOP_TIMEOUT = ppc.end_loop_timeout

        # Only --lerobot_grasp needs a session; the default servoing grasp does not, and
        # execute_grasp falls back to it anyway, so there is nothing to prompt about.
        if self.use_lerobot_grasp and not await self.check_lerobot_session_connected():
            answer = await self.send_popup_and_await_answer(
                "--lerobot_grasp is set but no session is connected. Start a subprocess of "
                "stringman-headless to run the grasping model? Answering No grasps with the "
                "visual servoing model instead.",
                buttons=["Yes", "No"],
            )
            if answer == 0:
                self.lerobot_process_watcher = asyncio.create_task(self.lerobot_process(
                    control.ManageLerobotSession(
                        action=control.LerobotSessionAction.START_EVAL,
                        repo_id="naavox/dit-grasp-3",
                    )
                ))

        drop_point = np.zeros(3)
        target_seen_t = time.time()
        try:
            gtask = None
            while self.run_command_loop:

                if self.pnp_src in (common.RoutePoint.ALL_TARGETS, common.RoutePoint.USER_TARGETS):
                    next_target = self.target_queue.get_best_target()
                    if next_target is None:
                        if gtask is not None:
                            gtask.cancel()
                        self.goal_pos = None
                        if time.time() > target_seen_t + END_LOOP_TIMEOUT:
                            logger.info('Looks clean enough to me!')
                            return
                        await asyncio.sleep(LOOP_DELAY)
                        continue
                    target_seen_t = time.time()

                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SELECTED)
                    self.send_tq_to_ui()

                    # pick Z position for gantry
                    # if we are too close to the drop point right now, the z position has to be our current z so we don't get hung up on the basket by going down too soon.
                    # otherwise use the normal value
                    if np.linalg.norm(self.pe.gant_pos - (drop_point + GANTRY_HEIGHT_OVER_DROPOFF[2])) < 0.5:
                        z_pos = self.pe.gant_pos[2]
                    else:
                        z_pos = GANTRY_HEIGHT_OVER_TARGET[2]
                    goal_pos = next_target.position + np.array([0, 0, z_pos])

                elif self.pnp_src in ROUTE_POINT_TAG_NAMES:
                    next_target = None
                    goal_pos = tonp(self.config.named_positions[ROUTE_POINT_TAG_NAMES[self.pnp_src]]) + GANTRY_HEIGHT_OVER_TARGET
                elif self.pnp_src == common.RoutePoint.ORIGIN:
                    next_target = None
                    goal_pos = GANTRY_HEIGHT_OVER_TARGET # over origin

                if gtask is None or gtask.done():
                    gtask = asyncio.create_task(self.seek_goal(goal_pos))
                else:
                    self.goal_pos = goal_pos # retarget the seek already in flight onto the newly chosen target
                done, pending = await asyncio.wait([gtask], timeout=1)
                
                if gtask in pending:
                    # if doesn't arrive in one second, run target selection again since a better one might have appeared or the user might have put one in their queue
                    if next_target is not None:
                        self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                    continue

                if self.gripper_client is None:
                    logger.warning('Pick and place aborted because we lost the gripper connection')
                    break

                # when we reach this point we arrived over the item. commit to it unless it proves impossible to pick up.
                logger.info('Attempt grasp')
                start = time.time()
                success = await self.execute_grasp()
                logger.info(f'Grasp succeeded={success} took {time.time() - start:.2f}s')
                if not success:
                    if next_target is not None:
                        # just pick another target, but consider downranking this object or something.
                        self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.SEEN)
                        self.send_tq_to_ui()
                    await asyncio.sleep(LOOP_DELAY)
                    continue
                else:
                    if next_target is not None:
                        self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.PICKED_UP)
                        self.send_tq_to_ui()
                    logger.info('Object picked up')

                # tension now just in case.
                # await self.tension_and_wait()

                # Choose drop point. default to origin
                drop_point = np.zeros(3)

                if self.pnp_dst == common.RoutePoint.NA and next_target is not None:
                    # read drop point from target
                    # TODO currently these are not populated with useful data.
                    if not isinstance(next_target.dropoff, str):
                        drop_point = next_target.dropoff
                    # otherwise go to the named drop point
                    if next_target.dropoff in self.config.named_positions:
                        drop_point = tonp(self.config.named_positions[next_target.dropoff])

                elif self.pnp_dst in ROUTE_POINT_TAG_NAMES:
                    # Typical path
                    drop_point = tonp(self.config.named_positions[ROUTE_POINT_TAG_NAMES[self.pnp_dst]])
                elif self.pnp_dst == common.RoutePoint.ORIGIN:
                    drop_point = np.zeros(3)

                # fly to to drop point
                logger.info(f'Flying to drop point {drop_point}')
                await self.seek_goal(drop_point + GANTRY_HEIGHT_OVER_DROPOFF)
                # open gripper
                current_finger_angle = self.datastore.finger.getLast()[1]
                open_target = max(-90, min(RELAXED_OPEN, current_finger_angle - 10))
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': open_target}))
                if next_target is not None:
                    # don't immediately select a new target, because there's a chance it'll be the sock you're holding.
                    await asyncio.sleep(DELAY_AFTER_DROP)
                    self.target_queue.set_target_status(next_target.id, telemetry.TargetStatus.DROPPED)
                    self.send_tq_to_ui()
                # keep score


        except asyncio.CancelledError:
            raise
        finally:
            if gtask is not None:
                logger.info('Pick and place cancelled')
                gtask.cancel()
            self.slow_stop_all_spools()
            await self.clear_goal()

    async def execute_grasp(self):
        """Try to grasp whatever is directly below the gripper"""
        if self.use_lerobot_grasp:
            # A lerobot session may be driving from our own subprocess or connected
            # remotely through the prod telemetry relay, so we can't tell locally if one
            # is present. lerobot_grasp broadcasts the eval-start and returns None if no
            # session answers, which is recoverable: servoing needs nothing but the robot.
            result = await self.lerobot_grasp()
            if result is not None:
                return result
            logger.warning('--lerobot_grasp is set but no session answered; servoing instead')
        if not await self.servo.ensure_model():
            logger.warning('No visual servoing model loaded; cannot grasp')
            return False
        return await self.servo.run(mode=SERVO_MODE_GRASP)

    def _set_target_model(self, model):
        """Set self.target_model, notifying the UI via auto_targeting_state whenever
        whether a model is loaded (not the model itself) changes."""
        was_loaded = self.target_model is not None
        self.target_model = model
        if (model is not None) != was_loaded:
            self.send_ui(auto_targeting_state=telemetry.AutoTargetingState(enabled=model is not None, present=True))

    async def _load_target_model(self):
        """Load the ortho target model and make it the active one."""
        import torch
        from huggingface_hub import hf_hub_download
        DEVICE = self._device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self._device = DEVICE

        if DEVICE == "cpu":
            logger.warning("Refusing to load targeting model on CPU; hardware acceleration required.")
            self._set_target_model(None)
            self.send_ui(pop_message=telemetry.Popup(
                message="Automatic target identification (targeting model) cannot be used without "
                        "some kind of hardware acceleration. Loading was aborted because the torch "
                        "device is CPU."
            ))
            return

        def load_sync():
            from nf_robot.ml import ortho_target
            filename = ortho_target.TARGETING_MODEL_FILENAME
            path = (f"models/{filename}" if self.local_models
                    else hf_hub_download(repo_id=ortho_target.TARGETING_MODEL_REPOID, filename=filename))
            logger.info(f"Loading ortho target model from {path}...")
            model, _ = ortho_target.load_checkpoint(path, DEVICE)
            return model

        # The checkpoint fetch and torch import inside load_sync report nothing, so the
        # bar is on a 5s timer; it holds at 99% for as long as the load actually takes.
        async with FakeProgress(
            self.send_ui,
            name="Target Model",
            current_action="Loading target model...",
            done_action="Target model ready",
            failed_action="Could not load the target model",
            expected_s=5.0,
            interval_s=0.2,
            suppress_completion_popup=True,
        ):
            model = await asyncio.to_thread(load_sync)
        self._set_target_model(model)

    async def _handle_set_target_model(self, item: control.SetTargetModel):
        # ortho_target is the only target model; every enable action loads it. The enum
        # still carries the retired per-model choices, which are all treated as the default.
        if item.action == control.TargetModelAction.TARGET_MODEL_DISABLE:
            self._set_target_model(None)
            logger.info('Target model disabled')
        elif item.action != control.TargetModelAction.TARGET_MODEL_ACTION_UNUSED:
            logger.info('Loading target model...')
            await self._load_target_model()
            logger.info('Target model ready')

    async def check_lerobot_session_connected(self, timeout=2) -> bool:
        """
        Broadcast a ping and see whether any lerobot session (local subprocess or one
        connected remotely through the relay) answers with a status within `timeout` seconds.
        """
        self.lerobot_session_status_event.clear()
        self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.PING))
        try:
            await asyncio.wait_for(self.lerobot_session_status_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.debug(f'No lerobot session answered the ping within {timeout}s; no session active.')
            return False

    async def lerobot_grasp(self):
        """
        Execute a grasp on an arp gripper using a lerobot ACT policy.
        End the episode either when a timeout is reached, when motion ceases for some time, or when a grasp condition is reached.
        A grasp condition is a certain amount of force being exerted by the fingers while being at a certain altitude off the floor.

        Returns True/False for grasp success once a session takes over, or None if no session
        answered the ping (so the caller can fall back to the visual servoing model).

        A seperate process must be connected to the telemetry stream to manage the act policy at this time. It can be started with

        python -m nf_robot.ml.stringman_lerobot eval   --robot_id=lan   --server_address=ws://localhost:4245   --policy_id=outputs/train/grasp_remote_act_eggs_2/checkpoints/last/pretrained_model/   --dataset_id=naavox/grasping_dataset_eggs_fix
        """
        self.pe.finger_pressure_rising.clear()
        try:
            if not await self.check_lerobot_session_connected():
                return None

            # A session is listening; tell it to start controlling.
            self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.EVAL_START))

            timeout = time.time() + 30
            lifted = False
            applying_force = False
            while not (lifted and applying_force) and time.time() < timeout:
                await asyncio.sleep(0.2)
                applying_force = self.pe.finger_pressure_rising.is_set()
                gripper_height = self.pe.grip_pose[1][2]
                lifted = gripper_height > 0.4
            logger.debug(f'Ended grasp lifted={lifted} applying_force={applying_force} time_rem={timeout - time.time():.1f}s')
            # return value indicates whether grasp was successful
            # todo future models will predict grasp success on their own
            return lifted # and applying_force
        except asyncio.CancelledError:
            raise
        finally:
            self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.EVAL_STOP))
            await asyncio.sleep(0.01)
            self.slow_stop_all_spools()

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
