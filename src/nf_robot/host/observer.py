from __future__ import annotations

import signal
import sys
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
from collections import deque, defaultdict
import uuid
import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from functools import partial
from pathlib import Path
import json
import re
import subprocess
from packaging.version import parse as parse_version, InvalidVersion

from nf_robot.common.pose_functions import compose_poses
from nf_robot.common.cv_common import *
from nf_robot.common.config_loader import *
import nf_robot.common.definitions as model_constants
from nf_robot.common.util import *
from nf_robot.generated.nf import telemetry, control, common
import nf_robot.generated.nf.config as nf_config
from nf_robot.host.data_store import DataStore
from nf_robot.host.stats import StatCounter
from nf_robot.host.target_queue import TargetQueue
from nf_robot.host.calibration import optimize_anchor_poses
from nf_robot.host.eyelet_calibration import optimize_arp_anchors, analyze_diamond_data, DIAMOND_SIZE
from nf_robot.host.anchor_client import RaspiAnchorClient, max_origin_detections
from nf_robot.host.gripper_client import RaspiGripperClient
from nf_robot.host.arp_gripper_client import ArpeggioGripperClient, rotate_vector, OMEGA
from nf_robot.host.arp_anchor_client import ArpeggioAnchorClient
from nf_robot.host.position_estimator import Positioner2

logger = logging.getLogger(__name__)

# Define the service names for network discovery
anchor_service_name = 'cranebot-anchor-service'
anchor_power_service_name = 'cranebot-anchor-power-service'
gripper_service_name = 'cranebot-gripper-service'
arp_gripper_service_name = 'cranebot-gripper-arpeggio-service'
arp_anchor_service_name = 'cranebot-anchor-arpeggio-service'

N_ANCHORS = {
    common.AnchorType.PILOT: 4,
    common.AnchorType.ARPEGGIO: 2,
}
N_LINES = 4
INPUT_VELOCITY_TTL_S = 2.0 # a commanded velocity keyed by a source expires this long after its last update
INFO_REQUEST_TIMEOUT_MS = 3000 # milliseconds
CONTROL_PLANE_PRODUCTION = "wss://neufangled.com"
CONTROL_PLANE_STAGING = "wss://nf-site-monolith-staging-690802609278.us-east1.run.app"
CONTROL_PLANE_LOCAL = "ws://localhost:8080"
UNPROCESSED_DIR = "square_centering_data_unlabeled"
USER_TARGETS_DIR = "user_targets_data"
METADATA_PATH = os.path.join(USER_TARGETS_DIR, "metadata.jsonl")

# threshold of non slack tension in newtons for arp anchors
TENSION_THRESH = 1.38

CRANEBOT_SERVICE_TYPES = [
    "_cranebot-gripper-arpeggio-service._tcp.local.",
    "_cranebot-gripper-service._tcp.local.",
    "_cranebot-anchor-power-service._tcp.local.",
    "_cranebot-anchor-service._tcp.local.",
]

# finger positions
OPEN = -30
CLOSED = 90

POLE = np.array([0,0,0.5334])
# distance from the tip of the pole (POLE[2] below the gantry) down to the bottom of the
# arp gripper fingers when they hang straight. gantry -> fingertip is POLE[2] + this.
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

def capture_gripper_image(ndimage, gripper_occupied=False):
    """
    Saves an image to the unprocessed directory. 
    Encodes gripper state in filename: {uuid}_g{1|0}.jpg
    """
    if not os.path.exists(UNPROCESSED_DIR):
        os.makedirs(UNPROCESSED_DIR)
    
    h, w = ndimage.shape[:2]
        
    state_str = "g1" if gripper_occupied else "g0"
    file_id = str(uuid.uuid4())
    img_filename = f"{file_id}_{state_str}.jpg"
    img_full_path = os.path.join(UNPROCESSED_DIR, img_filename)
    
    # Save (ensure RGB/BGR consistency)
    cv2.imwrite(img_full_path, ndimage)
    logger.info(f"Captured: {img_filename} (Gripper: {gripper_occupied})")

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
    It starts a websocket server to accept connections from local UIs 

    It starts a websocket server to accept connections from local UIs 
    It reads from the config file to find any components it already knows about.
    It starts zeroconf to discover any components it doesn't know about and add them to the config.
    As soon as a component in the config has a known address, it starts keep_robot_connected to continually reconnect to all known components.
    As soon as the first component websocket is connected, It starts position_estimator to continually run kalman filters on the observed variables.
    As soon as a feed from the first preferred camera is up, It starts run_perception to continually run inference on the camera feeds.

    Since this class serves as the coordination center of all the robot compnents, it also contains methods to perform
    various actions like calibration and the pick and place routine.
    """
    def __init__(self, terminate_with_ui, config_path, telemetry_env=None, run_ortho=True, stream_heatmap=False, auto_start=False, local_models=False, port=4245, debug=False) -> None:
        self.port = port
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
        self.telemetry_env = telemetry_env
        self.debug = debug
        self.loop_monitor = None  # only created in main() when --debug is passed
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
        # true while swing latency cal is running, so passive_safety recovers instead of
        # aborting on a tension trip during that step.
        self.swing_cal_in_progress = False
        # only used for integration test only to allow some code to run right after sending the gantry to a goal point
        self.test_gantry_goal_callback = None
        # event used to notify tasks that gripper is connected.
        self.gripper_client_connected = asyncio.Event()
        self.last_user_move_time = time.time()
        # last known positions of named tags/objects live in self.config.named_positions
        # (the single source of truth). It's written to disk on shutdown, in async_close.
        self.target_model = None
        self.centering_model = None
        self.predicted_lateral_vector = None
        self.perception_task = None
        # targets
        self.target_queue = TargetQueue()
        self.last_snapshot_hash = None # to spare the UI from too many updates
        # websockets to locally connected UIs
        self.connected_local_clients = set()
        self.telemetry_buffer = deque(maxlen=100)
        self.telemetry_buffer_lock = threading.RLock()
        self.startup_complete = asyncio.Event()
        self.any_anchor_connected = asyncio.Event() # fires as soon as first anchor connects, starting pe
        self.cloud_telem_websocket = None
        self.gip_task = None
        self.cloud_telem = None
        self.passive_safety_task = None
        # last attempt to connect, keyed by service name
        self.connection_tasks: dict[str, asyncio.Task] = {}
        self.run_collect_images = False
        self.time_last_grip_sensors_retain_key = 0
        # {key: (velocity, monotonic_timestamp)} last velocities commanded by different subsystems. all keys in active_set are summed.
        # Entries expire INPUT_VELOCITY_TTL_S after their last update; expiration is lazy (pruned at read time in move_direction_speed),
        # so a source key that stops sending moves stops contributing without needing any timer or background task.
        self.input_velocities = {'default': (np.zeros(3), time.monotonic())}
        self.active_set = set(['default'])
        self.run_ortho = run_ortho
        self.stream_heatmap = stream_heatmap
        self.auto_start = auto_start
        self._device = None
        self._telem_log_handler: TelemetryLogHandler | None = None
        self.swing_cancellation_task = None
        self.local_models = local_models
        # ortho projection state - written by _ortho_worker thread, read by run_perception AI task
        self.ortho_event = threading.Event()
        self.last_ortho_bgr = None
        self.last_ortho_heatmap = None
        self.last_heatmaps_np = None
        # list of (NfVideoStreamer, feed_number) for ortho feeds, so send_setup_telemetry can replay them
        self.ortho_streamers: list = []
        self.lerobot_process_watcher = None
        self.last_ep_ctrl_status = common.LerobotStatus.NA
        self.lerobot_process_pid = None
        # fires whenever any lerobot session (our own subprocess or one connected remotely
        # through the telemetry relay) reports a status. Used to detect whether a session is
        # actually listening after we broadcast an eval-start.
        self.lerobot_session_status_event = asyncio.Event()
        self.grip_angle = 0
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
            ))
        else:
            self.send_ui(new_anchor_poses=telemetry.AnchorPoses(
                poses=[a.pose for a in self.config.anchors]
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
        r = await self.flush_tele_buffer()

    async def handle_local_client(self, websocket):
        # Called when Ursina connects to a websocket that is opened to accept control commands
        self.connected_local_clients.add(websocket)
        logger.info('Connection received from local UI process')

        # send anything that it would need up-front
        r = await self.send_setup_telemetry()
        try:
            async for message in websocket:
                r = await self.handle_command(message) # Handle 'ControlBatchUpdate'
                # warning, any uncaught exception here will kill this websocket connection
                # but the observer would go on running, possibly in a bad state.
        except (ConnectionClosedError, ConnectionClosedOK) as e:
            pass
        # except Exception as e:
        #     print(e)
        #     traceback.print_exc()
        finally:
            self.connected_local_clients.remove(websocket)
            self.zero_input_velocities()
            if len(self.connected_local_clients) == 0 and self.terminate_with_ui:
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

        elif item.delete_target is not None:
            self._handle_delete_target(item.delete_target)

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

    async def _handle_set_point(self, item: control.SetPoint):
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
                goal_pos = tonp(target.position) + GRIPPER_HEIGHT_OVER_TARGET + POLE
        elif item.pos is not None:
            goal_pos = tonp(item.pos) + GRIPPER_HEIGHT_OVER_TARGET + POLE

        if goal_pos is None:
            return
        self.gantry_goal_pos = goal_pos
        r = await self.invoke_motion_task(self.seek_gantry_goal())

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
            if not isinstance(self.gripper_client, ArpeggioGripperClient):
                self.send_ui(pop_message=telemetry.Popup(
                    message=f'Swing cancellation only supported on Arpeggio Gripper'
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

        Moving the pivot (gantry) one way for half a pendulum period and back for
        the other half repeatedly adds energy in phase with the swing, the same
        way you pump a playground swing. A couple of cycles builds a clean,
        repeatable swing to measure against. The gantry returns to roughly where
        it started, so this does not require an accurate absolute position.
        """
        half_period = np.pi / OMEGA  # half of one pendulum swing
        direction = np.asarray(direction, dtype=float)
        try:
            for _ in range(cycles):
                await self.move_direction_speed(direction, speed, downward_bias=0)
                await asyncio.sleep(half_period)
                await self.move_direction_speed(-direction, speed, downward_bias=0)
                await asyncio.sleep(half_period)
        finally:
            self.slow_stop_all_spools()

    def _broadcast_swing_latency(self, latency):
        """Set config.swing_latency (in memory) and tell the UI. Does not persist;
        callers save_config only once a value is committed."""
        self.config.swing_latency = float(latency)
        self.send_ui(new_anchor_poses=telemetry.AnchorPoses(swing_latency=self.config.swing_latency))

    async def _recenter_gantry(self, center_pos):
        """Drive the gantry back to center_pos and stop."""
        self.gantry_goal_pos = np.array(center_pos, dtype=float)
        await self.seek_gantry_goal(head_turn=False, auto_altitude=False)
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

        Returns (residual, abort_reason):
          - pumped past the safety cap  -> residual = cap (definitively bad)
          - drifted out of the workspace -> residual = None (never settled, ignore)
        """
        RUN_PERIODS = 6.4          # how many pendulum periods to run cancellation per trial (main time cost)
        MEASURE_PERIODS = 3        # average the swing over this many final periods
        SETTLE_S = 0.5             # pause after inducing, before turning cancellation on
        SAFETY_AMP_RAD = 0.4       # stop early if the swing grows past this
        DRIFT_LIMIT_M = 0.6        # stop early if the gantry wanders this far
        LOOP_S = 1 / 100
        MIN_SAMPLES = 10
        ALTITUDE_HOLD_GAIN = 4.0       # 1/s, proportional gain pulling z back to the start altitude
        ALTITUDE_HOLD_MAX_MPS = 0.15   # cap on the vertical hold speed

        gc = self.gripper_client
        period = 2 * np.pi / OMEGA

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
            while (t := time.time() - start) < RUN_PERIODS * period:
                now = time.time()
                v = gc.compute_swing_correction(now + latency)
                vx, vy = (float(v[0]), float(v[1])) if v is not None else (0.0, 0.0)
                # Actively hold altitude. 
                z_error = center_pos[2] - self.pe.gant_pos[2]
                vz = float(np.clip(ALTITUDE_HOLD_GAIN * z_error, -ALTITUDE_HOLD_MAX_MPS, ALTITUDE_HOLD_MAX_MPS))
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
                    if amp > SAFETY_AMP_RAD:
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

        ts, amps = np.array(ts), np.array(amps)
        if aborted == 'tension':
            return None, aborted
        if aborted == 'amp_cap':
            return SAFETY_AMP_RAD, aborted
        if aborted == 'drift' or len(amps) < MIN_SAMPLES:
            return None, aborted
        late = amps[ts > ts[-1] - MEASURE_PERIODS * period]
        residual = float(np.mean(late)) if len(late) else float(np.mean(amps))
        return residual, aborted

    async def calibrate_swing_latency(self, fine_pass=False, progress_range=None):
        """Tune config.swing_latency by finding the value that damps the swing best.

        A good latency drives the swing to nothing; a bad one leaves a steady
        residual swing. So we try a range of latencies, measure the leftover swing at
        each, and keep the one that leaves the least. Every candidate stays close
        enough to the ideal that it damps (rather than pumps), so nothing gets
        thrown around.

        The coarse pass deliberately spreads its candidates wide (0, 0.6, 0.3) rather
        than sweeping a narrow range: the ideal latency depends on host event-loop
        contention and can land as high as ~0.6s. A spread this wide means some
        candidates pump hard rather than damp, but the safety amplitude cap stops those
        early, and whichever candidate is nearest the ideal still yields a clean, low
        residual to lock onto. Set fine_pass=True to add a second pass that refines
        around the coarse best.
        """
        COARSE_CANDS = (0.0, 0.6, 0.3)  # seconds; spread wide enough to bracket the ideal even under heavy loop contention
        FINE_HALF_WIDTH = 0.15       # fine pass spans +/- this around the coarse best (covers the gap between coarse samples)
        FINE_COUNT = 7
        FINE_CLIP = (0.0, 0.75)      # keep refined candidates within a sane latency range
        DRIFT_LIMIT_M = 0.6          # recenter between trials once drift exceeds this
        MIN_TRIALS = 3               # need at least this many good trials to choose
        TENSION_BACKOFF_S = 1.1      # wait this long after a tension trip before retrying a trial
        MAX_TENSION_RETRIES = 3      # give up (and abort) if a single trial keeps tripping tension

        if not isinstance(self.gripper_client, ArpeggioGripperClient):
            logger.warning('Swing latency calibration is only supported on the Arpeggio gripper')
            return None

        original_latency = self.config.swing_latency
        center_pos = np.array(self.pe.gant_pos, dtype=float)
        all_results = []      # (latency, residual) from every reliable trial

        async def sweep(cands):
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
                else:
                    logger.info(f'swing_latency {lat:.3f}s -> unreliable, excluded{tag}')
                await asyncio.sleep(0.3)
            return out

        self.tension_over_limit = False  # clear any stale trip so the first trial isn't cut short
        self.swing_cal_in_progress = True  # let passive_safety recover (not abort) on a tension trip here
        try:
            coarse = await sweep(COARSE_CANDS)
            if coarse and fine_pass:
                best_coarse = min(coarse, key=lambda r: r[1])[0]
                # Recenter before the fine pass so the trials we care about start with
                # full drift headroom and don't get cut short.
                await self._recenter_gantry(center_pos)
                fine = np.clip(np.linspace(best_coarse - FINE_HALF_WIDTH, best_coarse + FINE_HALF_WIDTH, FINE_COUNT), *FINE_CLIP)
                await sweep(sorted(set(np.round(fine, 3))))
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

        best = self._select_min_residual(all_results)
        self._broadcast_swing_latency(best)
        save_config(self.config, self.config_path)
        logger.info(f'Calibrated swing_latency = {best:.3f}s')
        return best

    @staticmethod
    def _select_min_residual(results):
        """Pick the center of the range of latencies that all damp the swing fully.

        The swing measurement can't read below a small floor (~20 mrad), so every
        latency that fully damps ties near that floor -- the best isn't a single
        point but a range. Any latency in that range works; we return its midpoint,
        which sits farthest from the edges where damping starts to fail and is more
        repeatable than picking an edge.

        results is a list of (latency, residual). Duplicate latencies keep their
        best reading so one bad settle doesn't reject an otherwise-good latency.
        """
        FLOOR_MARGIN = 0.010   # "as good as the best" = within this (or 50%) of the smallest residual

        groups = defaultdict(list)
        for lat, r in results:
            groups[round(lat, 3)].append(r)
        lats = np.array(sorted(groups))
        resid = np.array([min(groups[l]) for l in lats])

        rmin = float(resid.min())
        at_floor = resid <= rmin + max(0.5 * rmin, FLOOR_MARGIN)

        i0 = int(np.argmin(resid))
        lo = hi = i0
        while lo - 1 >= 0 and at_floor[lo - 1]:
            lo -= 1
        while hi + 1 < len(lats) and at_floor[hi + 1]:
            hi += 1
        best = float((lats[lo] + lats[hi]) / 2)
        logger.info(f'Fully-damped latency range {lats[lo]:.3f}-{lats[hi]:.3f}s; picking center {best:.3f}s')
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
            upper_z = np.mean(self.pe.anchor_points[:, 2]) # top of work area
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
            # Run the fine pass and emit progress so the debug-triggered run refines around
            # the coarse best and reports status just like the in-calibration invocation.
            r = await self.invoke_motion_task(self.calibrate_swing_latency(fine_pass=True, progress_range=(0.0, 100.0)))
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

    async def set_tension_reg(self, enabled: bool):
        """Enable or disable onboard tension regulation (the floor + soft mute) on both
        spools of every anchor."""
        logger.info(f'setting tension reg {"on" if enabled else "off"} for all anchors')
        await asyncio.gather(*[
            anchor.send_commands({'set_tension_reg': (enabled, spool_no)})
            for anchor in self.anchors.values()
            for spool_no in (0, 1)
        ])

    async def sync_timezone_to_bots(self):
        tz = subprocess.check_output(['timedatectl', 'show', '--property=Timezone', '--value']).decode().strip()
        await asyncio.gather(*[
            client.send_commands({'set_timezone': tz})
            for client in self.bot_clients.values()
        ])

    async def chase_tag(self, name):
        """Keep the gripper at the named location"""
        try:
            chase_task = None
            while self.run_command_loop:
                await asyncio.sleep(0.1)
                if not name in self.config.named_positions:
                    continue
                goal = tonp(self.config.named_positions[name]) + POLE
                self.gantry_goal_pos = goal
                if chase_task is None or chase_task.done():
                    chase_task = asyncio.create_task(self.seek_gantry_goal())
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
                goal = tonp(self.config.named_positions[source]) + POLE + GRIPPER_HEIGHT_OVER_TARGET
                self.gantry_goal_pos = goal
                await self.seek_gantry_goal()

                # auto grasp
                # await self.gripper_client.send_commands({'set_finger_angle': 30})
                # await asyncio.sleep(1)
                await self.execute_grasp()

                # wait for destination position to be seen
                while not dest in self.config.named_positions:
                    await asyncio.sleep(0.5)
                # go to position
                goal = tonp(self.config.named_positions[dest]) + POLE + GRIPPER_HEIGHT_OVER_TARGET
                self.gantry_goal_pos = goal
                await self.seek_gantry_goal()

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

    def _handle_delete_target(self, item: control.DeleteTarget):
        if item.target_id is not None:
            self.target_queue.remove_target(item.target_id);

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

    def submitTargets(self):
        """snapshot any active cameras at 1920x1080 and save images in the raw dir"""
        images = []
        for anchor in self.anchors.values():
            if anchor.frame is not None:
                images.append(anchor.frame.copy())

        def save_data(images):
            directory_path = Path("target_heatmap_data_unlabeled")
            directory_path.mkdir(exist_ok=True, parents=True)
            
            for img in images:
                img_filename = f"{str(uuid.uuid4())}.jpg"
                # write the image
                rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_full_path = directory_path / img_filename
                cv2.imwrite(str(img_full_path), rgb_image)

        threading.Thread(target=save_data, args=(images,)).start()

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
            case control.Command.ZERO_WINCH:
                asyncio.create_task(self._handle_zero_winch_line())
            case control.Command.HALF_CAL:
                r = await self.invoke_motion_task(self.half_auto_calibration())
            case control.Command.FULL_CAL:
                r = await self.invoke_motion_task(self.full_auto_calibration())
            case control.Command.PICK_AND_DROP:
                r = await self.invoke_motion_task(self.pick_and_place_loop())
            case control.Command.HORIZONTAL_CHECK:
                r = await self.invoke_motion_task(self.linear_height_check_task())
            case control.Command.COLLECT_GRIPPER_IMAGES:
                self._handle_collect_images()
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
            case control.Command.SUBMIT_TARGETS_TO_DATASET:
                self.submitTargets()
            case control.Command.UPDATE_FIRMWARE:
                r = await self._handle_update_firmware()
            case control.Command.DISABLE_TORQUE:
                await self._handle_disable_torque()
            case control.Command.ENABLE_TORQUE:
                await self._handle_enable_torque()
            case control.Command.DEBUG_LOG_OVER_T:
                self._enable_debug_log_over_telemetry()
            case control.Command.ENABLE_TENSION_REG:
                r = await self.set_tension_reg(True)
            case control.Command.DISABLE_TENSION_REG:
                r = await self.set_tension_reg(False)

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

    async def _handle_disable_torque(self):
        if self.config.anchor_type != common.AnchorType.ARPEGGIO:
            return
        await asyncio.gather(*[
            client.send_commands({'disable_torque': None})
            for client in self.anchors.values()
        ])

    async def _handle_enable_torque(self):
        if self.config.anchor_type != common.AnchorType.ARPEGGIO:
            return
        await asyncio.gather(*[
            client.send_commands({'enable_torque': None})
            for client in self.anchors.values()
        ])

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
        """Handles moving the gantry to a specific goal position."""
        self.gantry_goal_pos = goal_pos
        await self.invoke_motion_task(self.seek_gantry_goal())

    async def _handle_slow_stop_one(self, stop_data: dict):
        """Handles stopping a single spool motor."""
        if stop_data.get('id') == 'gripper' and self.gripper_client:
            r = await self.gripper_client.slow_stop_spool()
        else:
            for client in self.anchors.values():
                if client.anchor_num == stop_data.get('id'):
                    r = await client.slow_stop_spool()

    async def _handle_zero_winch_line(self):
        if self.gripper_client is not None and isinstance(self.gripper_client, RaspiGripperClient):
            await self.gripper_client.zero_winch()

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

            if self.gripper_client is not None and isinstance(self.gripper_client, ArpeggioGripperClient):
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
                await self._handle_disable_torque()
                await asyncio.sleep(1)
                await self._handle_enable_torque()
                await asyncio.sleep(1)
            await asyncio.sleep(0.2)

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
            if isinstance(client, RaspiAnchorClient):
                sends.append(client.send_commands({'tighten': None}))
            elif isinstance(client, ArpeggioAnchorClient):
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
            if isinstance(client, RaspiAnchorClient):
                sends.append(client.send_commands({'stow': None}))
            elif isinstance(client, ArpeggioAnchorClient):
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
        if self.config.anchor_type == common.AnchorType.PILOT:
            # any anchor that receives this and is slack would ignore it
            # If only some anchors are connected, this would still send reference lengths to those
            for client in self.anchors.values():
                asyncio.create_task(client.send_commands({'reference_length': lengths[client.anchor_num]}))
        elif self.config.anchor_type == common.AnchorType.ARPEGGIO:
            for client in self.anchors.values():
                # which two lines is this anchor responsible for?
                asyncio.create_task(client.send_commands({
                    'two_reference_lengths': (lengths[client.anchor_num*2], lengths[client.anchor_num*2+1])
                }))

        # use swing to estimate winch line length in pilot gripper
        if self.gripper_client is not None and isinstance(self.gripper_client, RaspiGripperClient):
            winch_length = self.pe.get_pendulum_length()
            if winch_length is not None:
                asyncio.create_task(self.gripper_client.send_commands({'reference_length': winch_length}))

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

    def snapshot_tag_observations(self):
        """Recent origin detections and cal_assist marker detections

        returns a dict of raw observations of various markers
        the shape of a pose is (2,3) with rotation coming first
        the first dimension is anchor number, the next is observation
        # for the arp anchor, the shape would be (2,12,2,3)

        'marker_name': array(n_anchors, n_observations, 2, 3)
        """
        markers = ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3', 'gantry']
        raw_obs = defaultdict(lambda: [[]]*N_ANCHORS[self.config.anchor_type])
        for client in self.anchors.values():
            # copy each list of detections, but leave them in the camera's reference frame.
            for marker in markers:
                if marker == 'gantry':
                    raw_obs[marker][client.anchor_num] = list(client.raw_gant_poses)
                else:
                    raw_obs[marker][client.anchor_num] = list(client.origin_poses[marker])
                # print(f'anchor {client.anchor_num} has {len(raw_obs[marker][client.anchor_num])} observations of {marker}')
        return dict(raw_obs)

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
            eyelets=[fromnp(e) for e in eyelet_positions]
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
            _, half_w = DIAMOND_SIZE
            # how far below the top of the work area (upper_z) the gantry's top point should stay.
            TOP_MARGIN_M = 1.0

            results = {}
            line_deltas = {}

            def get_eyelet_lengths():
                l1 = self.datastore.anchor_line_record[1].getLast()[1]
                l3 = self.datastore.anchor_line_record[3].getLast()[1]
                return l1, l3

            async def wait_for_lines_to_stop(deadband=0.05, timeout=30):
                await asyncio.sleep(2)
                deadline = asyncio.get_event_loop().time() + timeout
                while asyncio.get_event_loop().time() < deadline:
                    speeds = [abs(self.datastore.anchor_line_record[i].getLast()[2]) for i in range(N_LINES)]
                    if all(s < deadband for s in speeds):
                        await asyncio.sleep(2)
                        return
                    await asyncio.sleep(1/30)
                logger.warning('wait_for_lines_to_stop timed out; proceeding with current line lengths')

            async def move_to_diamond_point(jog1=0.0, jog3=0.0):
                """Reposition the gantry to a diamond point by jogging the two eyelet
                (indirect) lines. The two anchor (direct) lines are held at
                DIAMOND_DIRECT_TENSION_N by the onboard tension loop (set up below), so we
                only have to wait until every line has stopped moving before measuring."""
                if jog1:
                    await self.send_line_speed(1, jog1, jog=True)
                if jog3:
                    await self.send_line_speed(3, jog3, jog=True)
                await wait_for_lines_to_stop()
                await self.send_line_speed(1, 0)
                await self.send_line_speed(3, 0)

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
            logger.info('This position is the bottom of the diamond. Observe gantry for 2 seconds')
            # regulate the anchor lines to the target tension and wait for everything to settle before measuring
            await move_to_diamond_point()
            await asyncio.sleep(5)
            results['bottom'] = self.snapshot_tag_observations()['gantry']

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
            await asyncio.sleep(5)
            results['right'] = self.snapshot_tag_observations()['gantry'] # it is to the right from the perspective of camera 0

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
            await asyncio.sleep(5)
            results['top'] = self.snapshot_tag_observations()['gantry']

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
            await asyncio.sleep(5)
            results['left'] = self.snapshot_tag_observations()['gantry']

            # release the direct lines back to the normal tension floor
            await self.set_line_tension_target(0, None)
            await self.set_line_tension_target(2, None)

            logger.info('Return result')
            for a in self.anchors.values():
                a.save_raw = True

            analyze_diamond_data(results, anchor_poses, tilts)

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
        HOVER_CAMERA_HEIGHTS_M = [1.4, 1.0, 0.4]  # camera heights over each card to sample. Visiting a
                                                  # card from several altitudes gives the length-delta
                                                  # constraints a vertical baseline, which is what lets
                                                  # them begin to observe the far external eyelets (a
                                                  # single-height cluster leaves the eyelet radial
                                                  # direction free). The spread is deliberately wide -
                                                  # a wider baseline recovers more of a bad pass-2 - but
                                                  # each height is clamped under the work-area ceiling.
        SETTLE_S = 4.0                # let swing cancellation settle the gripper before measuring
        MEASURE_WINDOW_S = 3.0        # average camera + line readings over this window
        MEASURE_TIMEOUT_S = 6.0       # give up on a card if the gripper never sees it
        SEEK_TIMEOUT_S = 20.0         # cap the move to each hover altitude

        if not isinstance(self.gripper_client, ArpeggioGripperClient):
            logger.warning('collect_gripper_card_observations only supports the Arpeggio gripper')
            return {}

        # keep the gripper vertical and the lines taut throughout the survey
        self.set_swing_cancellation(True)

        card_positions = self.card_room_positions()
        if not card_positions:
            logger.warning('No calibration cards visible to the anchor cameras; cannot run gripper card survey')
            return {}

        # don't fly higher than just under the top of the work area
        upper_z = np.mean(self.pe.anchor_points[:, 2])

        survey_names = [n for n in ['origin', 'cal_assist_1', 'cal_assist_2', 'cal_assist_3'] if n in card_positions]

        async def measure_hover(name):
            """Center on the card and average the card-to-gantry offset and line lengths over a short
            window. Returns a sample dict, or None if the gripper never sees the card here."""
            # center the card in view so the measurement is taken on the camera's axis
            await self._center_card_in_view(name)
            await asyncio.sleep(1.5)
            gantry_offsets = []
            line_samples = []
            deadline = time.time() + MEASURE_TIMEOUT_S
            window_end = None
            while time.time() < deadline:
                pose_cam = self.gripper_client.route_tag_poses_relative_to_camera.get(name)
                if pose_cam is not None:
                    gantry_offsets.append(self.gripper_client.measure_gantry_minus_card(pose_cam))
                    line_samples.append([self.datastore.anchor_line_record[i].getLast()[1] for i in range(N_LINES)])
                    if window_end is None:
                        window_end = time.time() + MEASURE_WINDOW_S
                if window_end is not None and time.time() >= window_end:
                    break
                await asyncio.sleep(0.05)
            if not gantry_offsets:
                return None
            return {
                'gantry_minus_card': np.mean(gantry_offsets, axis=0),
                'line_lengths': np.mean(line_samples, axis=0),
                'n': len(gantry_offsets),
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
                gant_zs = sorted({min(upper_z - 0.1, cpos[2] + POLE[2] + h) for h in HOVER_CAMERA_HEIGHTS_M}, reverse=True)

                # Fly toward the anchor-camera estimate at the highest sampled height (widest view, so
                # the best chance to catch and center the card), but stop the moment the gripper camera
                # sees the card: its true spot can differ from the estimate, and continuing can carry it
                # back out of the narrow gripper FOV.
                approach_z = gant_zs[0]
                self.gantry_goal_pos = np.array([cpos[0], cpos[1], approach_z])
                logger.info(f'Gripper card survey: flying over {name} at goal {np.round(self.gantry_goal_pos, 3)} (card at {np.round(cpos, 3)})')
                seek_task = asyncio.create_task(self.seek_gantry_goal(head_turn=False))
                try:
                    while not seek_task.done():
                        if self.gripper_client.route_tag_poses_relative_to_camera.get(name) is not None:
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
                for gz in gant_zs:
                    self.gantry_goal_pos = np.array([cpos[0], cpos[1], gz])
                    # hold the exact target altitude (auto_altitude would cruise at a fixed height and
                    # defeat the point of sampling several).
                    seek_task = asyncio.create_task(self.seek_gantry_goal(head_turn=False, auto_altitude=False))
                    try:
                        await asyncio.wait_for(seek_task, timeout=SEEK_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        logger.warning(f'Gripper card survey: did not reach z={gz:.2f} over {name} within {SEEK_TIMEOUT_S:.0f}s; measuring anyway')
                    self.slow_stop_all_spools()
                    await asyncio.sleep(SETTLE_S)

                    sample = await measure_hover(name)
                    if sample is None:
                        logger.warning(f'Gripper card survey: never saw {name} at gantry z={gz:.2f}; skipping this height')
                        continue
                    samples.append(sample)
                    logger.info(
                        f'Gripper card survey: {name} z={gz:.2f} n={sample["n"]} '
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

    async def _nudge_gantry_xy(self, delta_xy, speed=0.04):
        """Move the gantry a small horizontal step of (approximately) delta_xy meters, then stop.
        Uses a bias-free velocity command (the onboard tension floor keeps lines taut), so the
        move stays in the horizontal plane rather than drifting down like a normal biased move."""
        dist = float(np.linalg.norm(delta_xy))
        if dist < 0.005:
            return
        dist = min(dist, 0.15)  # cap a single nudge for safety
        uvec = np.array([delta_xy[0], delta_xy[1], 0.0])
        uvec = uvec / (np.linalg.norm(uvec) + 1e-9)
        await self.move_direction_speed(uvec * speed, None, self.pe.gant_pos)
        await asyncio.sleep(dist / speed)
        self.slow_stop_all_spools()
        await asyncio.sleep(0.5)  # let it settle before the next observation

    async def _center_card_in_view(self, name, tol_m=0.05, gain=0.6, max_steps=6):
        """Bounded visual-centering: nudge the gantry so the named card sits under the gripper
        camera. measure_gantry_minus_card gives the room offset from card to gantry; moving the
        gantry by the negative of its horizontal part drives that toward zero (gantry over card,
        card centered). If a nudge grows the error (a wrong-sign heading convention) the nudge
        direction is flipped once; if it still grows, centering gives up. Stops when centered,
        when the card is lost, or after max_steps."""
        prev = None
        for step in range(max_steps):
            pose_cam = self.gripper_client.route_tag_poses_relative_to_camera.get(name)
            if pose_cam is None:
                # brief reacquire attempt before giving up
                deadline = time.time() + 1.0
                while time.time() < deadline and pose_cam is None:
                    await asyncio.sleep(0.05)
                    pose_cam = self.gripper_client.route_tag_poses_relative_to_camera.get(name)
                if pose_cam is None:
                    logger.info(f'Centering {name}: lost from view at step {step}; measuring as-is')
                    return
            err_xy = self.gripper_client.measure_gantry_minus_card(pose_cam)[:2]
            err = float(np.linalg.norm(err_xy))
            if err < tol_m:
                logger.info(f'Centering {name}: within {err*100:.1f}cm after {step} steps')
                return
            if prev is not None and err > prev + 0.02:
                logger.info(f'Centering {name}: error grew ({prev*100:.1f}->{err*100:.1f}cm); flipping nudge direction')
                return
            prev = err
            await self._nudge_gantry_xy(gain * err_xy)
        logger.info(f'Centering {name}: reached max steps')

    async def half_auto_calibration(self):
        """
        Set line lengths from observation
        tighten, wait for obs, estimate line lengths, move up slightly, estimate line lengths, move down slightly
        This is a motion task
        """
        NUM_SAMPLE_POINTS = 3
        OPTIMIZER_TIMEOUT_S = 60  # seconds
        
        try:
            if len(self.anchors) < N_ANCHORS[self.config.anchor_type]:
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

        If the gripper never answers the query, its server is too old to support it;
        we return without doing anything rather than upgrading without permission."""
        VERTICAL_TOLERANCE_DEG = 10.0
        MAX_LIFT_M = 1.0
        MAX_LIFT_S = 10.0
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
            if (np.linalg.norm(self.pe.gant_pos - vertical_start_pos) >= MAX_LIFT_M
                    or time.time() - vertical_start_time >= MAX_LIFT_S):
                self.slow_stop_all_spools()
                self.send_ui(pop_message=telemetry.Popup(
                    message='Could not achive a vertical pose to begin calibration. manually position the gripper in the center of the room hovering just over the floor and restart calibration.'
                ))
                raise RuntimeError('Could not achieve a vertical gripper pose to begin calibration')
            await self.move_direction_speed([0, 0, 1], 0.1, downward_bias=0)
            await asyncio.sleep(0.25)
        self.slow_stop_all_spools()

    async def full_auto_calibration(self):
        """Automatically determine anchor poses and zero angles
        This is a motion task"""
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=0.0,
            name="Calibration",
            current_action="Observing markers",
        ))
        finger_task = None
        DETECTION_WAIT_S = 1.0 # seconds
        FLOOR_CLEARANCE_M = 0.2 # how far above the floor to hold the gripper fingertips at the diamond's bottom point
        self.tension_over_limit = False  # clear any stale trip from a previous run
        try:
            if len(self.anchors) < N_ANCHORS[self.config.anchor_type]:
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=100.0,
                    name="Calibration",
                    current_action='Cannot run full calibration until all anchors are connected',
                ))
                return
            elif len(self.anchors) > N_ANCHORS[self.config.anchor_type]:
                logger.warning(f'Too many anchors found for type {self.config.anchor_type} \n{self.anchors}')
            await self._handle_enable_torque()
            # collect observations of origin card aruco marker to get initial guess of anchor poses.
            #   origin pose detections are actually always stored by all connected clients,
            #   it is only necessary to ensure enough have been collected from each client and average them.
            for a in self.anchors.values():
                a.save_raw = True
            num_o_dets = []
            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=0.0,
                name="Calibration",
                current_action="Observing markers",
            ))
            ORIGIN_VISIBILITY_TIMEOUT_S = 30.0 # give up if some anchor camera never sees the origin card
            detecting_start = time.time()
            while len(num_o_dets) == 0 or min(num_o_dets) < max_origin_detections:
                logger.debug(f'Waiting for enough origin card detections from every anchor camera {num_o_dets}')
                self.send_ui(visibility_states=telemetry.VisibilityStates(anchors_seeing_origin_card=list(
                    [anum for anum, count in enumerate(num_o_dets) if count > 0] # only anchor nums which see the origin card
                )))

                if time.time() - detecting_start >= ORIGIN_VISIBILITY_TIMEOUT_S:
                    self.slow_stop_all_spools()
                    self.send_ui(pop_message=telemetry.Popup(
                        message="The origin card must be placed at a location visible to both cameras. "
                                "If there is no overlap in the camera's views of the room. "
                                "either mount them closer, or install different camera tilt adapters."
                    ))
                    raise RuntimeError('Origin card not visible to all anchor cameras within timeout')

                await asyncio.sleep(DETECTION_WAIT_S)
                num_o_dets = [len(client.origin_poses['origin']) for client in self.anchors.values()]
            logger.info(f'Collected enough observations {num_o_dets}')
            self.send_ui(visibility_states=telemetry.VisibilityStates(anchors_seeing_origin_card=list(
                [anum for anum, count in enumerate(num_o_dets) if count > 0] # only anchor nums which see the origin card
            )))

            raw_obs = self.snapshot_tag_observations()

            self.send_ui(operation_progress=telemetry.OperationProgress(
                percent_complete=1.0,
                name="Calibration",
                current_action="Running 1st optimization pass",
            ))
            r = await self.flush_tele_buffer()

            if self.config.anchor_type == common.AnchorType.ARPEGGIO:
                tilts = (self.config.anchors[0].indirect_line.cam_tilt, self.config.anchors[1].indirect_line.cam_tilt)
                # determine position of two anchors visually and guess at external eyelets.
                async_result = self.pool.apply_async(optimize_arp_anchors, (raw_obs, None, None, None, tilts))
                anchor_poses, eyelet_positions, floor_z = async_result.get(timeout=30)
                logger.info(f'Obtained result from optimize_arp_anchors anchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')

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

                # top of work area
                upper_z = np.mean(self.pe.anchor_points[:, 2])

                # even without full calibration we should be able to make crude movements. go to the center
                # of the room just above the floor. This is the diamond's bottom point, so place the gantry
                # such that the gripper fingertips (POLE[2] + GRIPPER_FINGER_LEN_M below the gantry) sit
                # FLOOR_CLEARANCE_M above the floor.
                gant_z = min(
                    upper_z-0.1, # stay at least 0.1 under the top of the work area
                    POLE[2] + GRIPPER_FINGER_LEN_M + FLOOR_CLEARANCE_M - floor_z # mind that the origin card might be on a bed or a table, with the origin under the bed
                )
                self.gantry_goal_pos = np.array([0, 0, gant_z])
                await self.seek_gantry_goal()

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

                async_result = self.pool.apply_async(optimize_arp_anchors, (raw_obs, diamond_data, None, None, line_deltas, tilts))
                anchor_poses, eyelet_positions, floor_z = async_result.get(timeout=60)
                logger.info(f'Obtained result from optimize_arp_anchors anchor_poses=\n{anchor_poses}\neyelet_positions=\n{eyelet_positions}')

                self.save_poses_arp(anchor_poses, eyelet_positions)

            else:
                # Pilot anchors path
                for a in self.anchors.values():
                    a.save_raw = False

                # run optimization in pool
                async_result = self.pool.apply_async(optimize_anchor_poses, (raw_obs,))
                anchor_poses = async_result.get(timeout=30)
                logger.info(f'Obtained result from find_cal_params anchor_poses=\n{anchor_poses}')
                anchor_poses = np.array(anchor_poses)

                # Use the optimization output to update anchor poses and spool params
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
            gant_z = min(upper_z-0.1, POLE[2] + 0.8 - floor_z)
            self.gantry_goal_pos = np.array([0,0,gant_z])
            await self.seek_gantry_goal(head_turn=False)

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
            # them best. Only the Arpeggio gripper has the IMU-driven swing model.
            if and isinstance(self.gripper_client, ArpeggioGripperClient):
                self.send_ui(operation_progress=telemetry.OperationProgress(
                    percent_complete=34.0,
                    name="Calibration",
                    current_action="Tuning swing cancellation",
                ))
                # Perform swing cancellation measurements lower than the spin-measurement
                SWING_MEASURE_DROP_M = 0.4
                self.gantry_goal_pos = np.array([0, 0, gant_z - SWING_MEASURE_DROP_M])
                await self.seek_gantry_goal(head_turn=False)
                await self.calibrate_swing_latency(fine_pass=True, progress_range=(30.0, 61.0))

            # Refine the pull-point geometry with close-range gripper-camera views of the
            # calibration cards. The cards are still in place at this point (they are only
            # removed once calibration reports complete), and tension reg + swing cancellation
            # keep all four lines taut while hovering, so the measured (gantry, line-length)
            # pairs are a strong constraint on the anchors and eyelets.
            if (self.config.anchor_type == common.AnchorType.ARPEGGIO
                    and isinstance(self.gripper_client, ArpeggioGripperClient)
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
                self.gantry_goal_pos = np.array([0,0,gant_z])
                await self.seek_gantry_goal(head_turn=False)

                # Require a reading from all four cards (origin + 3 cal_assist). With fewer hovers the
                # gripper term has too few length-delta pairs to pin the two far eyelets, and the
                # under-constrained refinement distorts a good rectangular layout into a diamond.
                REQUIRED_GRIPPER_CARDS = 4
                if len(gripper_obs) >= REQUIRED_GRIPPER_CARDS:
                    # Final safety measure: turn swing cancellation off before applying the refined
                    # geometry, and turn it back on afterwards only if it still damps. Even with the
                    # anchors frozen (so the room frame cannot rotate), the new eyelets change the
                    # velocity->line-speed mapping swing cancellation depends on, so a bad refinement
                    # could make it pump and throw the gripper around.
                    self.set_swing_cancellation(False)
                    self.slow_stop_all_spools()

                    # Freeze the anchors during the gripper refinement so it can only move the
                    # eyelets. The room's absolute rotation about z is unobservable to the
                    # distance-based constraints, so with anchors free the gripper term (whose
                    # measured vectors live in the real room frame) can spin the whole solution
                    # about z. That silently invalidates the room-spin constant from the spin
                    # step and can flip swing cancellation from damping to pumping. Fixing the
                    # anchors pins the room frame; the well-determined anchors don't need this
                    # refinement anyway, while the weakly-constrained eyelets still get it.
                    args = (raw_obs, diamond_data, eyelet_positions, anchor_poses, line_deltas, tilts, gripper_obs)
                    async_result = self.pool.apply_async(optimize_arp_anchors, args)
                    refined_anchors, refined_eyelets, refined_floor_z = async_result.get(timeout=60)
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
            if self.tension_over_limit:
                self.tension_over_limit = False
                current_action = "Aborted: line tension exceeded the safe limit"
            else:
                current_action = "Cancelled by user"
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
        if self.gripper_client.last_frame_resized is None:
            logger.warning('Cannot calibrate the relationship between gripper zero angle and camera if gripper camera is offline!')
            return None

        # record the z rotation of the gantry card from the perspective of the gripper camera's stabilized frame
        # when the stabilization is done without any existing z rotation term
        self.gripper_client.calibrating_room_spin = True

        if isinstance(self.gripper_client, ArpeggioGripperClient):
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
                        # a pose of the origin card in the frame of reference of the stabilized gripper cam.
                        origin_card_pose[0] = d['p']
            end_time = time.time() + 10
            logger.info('Collecting observations of origin card from gripper cam')
            while origin_card_pose[0] is None and time.time() < end_time:
                async_result = self.pool.apply_async(
                    locate_markers_gripper,
                    (self.gripper_client.last_frame_resized, self.config.camera_cal_wide),
                    callback=partial(special_handle_det, time.time()))
                detections = async_result.get(timeout=5)
        except Exception as e:
            logger.exception(e)
            raise
        if origin_card_pose[0] is None:
            raise RuntimeError("Gripper camera was unable to make any observations of the origin card.")
        
        euler_rot = Rotation.from_rotvec(origin_card_pose[0][0]).as_euler('zyx')
        logger.info(f'Euler rotation of origin card relative to stabilized gripper camera {euler_rot}')
        roomspin = euler_rot[0]
        self.config.gripper.frame_room_spin = roomspin
        self.config.has_been_calibrated = True
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
        destination. Through an ideal move the laser should read (1.5 - POLE - laser_offset)
        the whole way. Aborts if the laser altitude drops below 0.2m or if the gantry comes
        within 0.4m of the ceiling (the z position of anchor 0).
        """
        TEST_ALTITUDE_M = 1.5
        MIN_LASER_ALTITUDE_M = 0.2
        CEILING_MARGIN_M = 0.4
        SAMPLE_INTERVAL_S = 0.02
        ideal_laser_range = TEST_ALTITUDE_M - POLE[2] - model_constants.laser_offset

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
        self.gantry_goal_pos = point_a
        await self.seek_gantry_goal(auto_altitude=True)
        await asyncio.sleep(2.0)

        # Traverse to the route destination, sampling the laser the whole way.
        # disable altitude cruise during test
        deviations = []
        aborted = None
        self.gantry_goal_pos = point_b
        move_task = asyncio.create_task(self.seek_gantry_goal(auto_altitude=False))
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
        Measure how accurately seek_gantry_goal parks the gripper over a route-point tag.
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
            samples = []
            deadline = time.time() + MEASURE_TIMEOUT_S
            window_end = None
            while time.time() < deadline:
                pose = self.gripper_client.route_tag_poses_relative_to_camera.get(tag_name)
                if pose is not None:
                    samples.append(np.array(pose[1]))
                    if window_end is None:
                        window_end = time.time() + MEASURE_WINDOW_S
                if window_end is not None and time.time() >= window_end:
                    break
                await asyncio.sleep(0.1)
            if not samples:
                return None
            return np.mean(samples, axis=0)

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
            self.gantry_goal_pos = goal_pos
            await self.seek_gantry_goal(auto_altitude=True)
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

        if isinstance(self.gripper_client, RaspiGripperClient):
            logger.warning("Self park unsupported in pilot gripper")
            return

        try:
            # TODO check if holding something, if so warn user and do not proceed.

            # perform half cal.

            # open gripper
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': FINGER_ANGLE_FOR_CLEAR_VIEW}))

            # move to position above and in front of saddle,
            parkpos = tonp(self.config.park_data.pos)
            away = get_inward_wall_normal(parkpos, self.pe.anchor_points) * STAGING_HOR_OFFSET_M
            self.gantry_goal_pos = parkpos + np.array([away[0], away[1], STAGING_VER_OFFSET_M])
            await self.seek_gantry_goal()

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
                    # since the gripper's camera is stabilized and rotated into the room frame of reference
                    # a vector pointing from the desired position of the marker to the current position in image space
                    # is the same direction we'd need to move the gantry in the room.
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
            await self.clear_gantry_goal()


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
            self.gantry_goal_pos = np.array([0,0,1])
            task = asyncio.create_task(self.seek_gantry_goal())
            # but don't go all the way, just stop after a bit
            await asyncio.sleep(5.0)
            await self.clear_gantry_goal()
            await self.half_auto_calibration()
        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

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

        is_power_anchor = kind == anchor_power_service_name
        is_standard_anchor = kind == anchor_service_name
        is_standard_gripper = kind == gripper_service_name
        is_arp_gripper = kind == arp_gripper_service_name
        is_arp_anchor = kind == arp_anchor_service_name

        # -- BEFORE --
        # the number of anchors is decided ahead of time (in main.py)
        # but they are assigned numbers as we find them on the network
        # and the chosen numbers are persisted in configuration.json

        # -- AFTER --
        # the number of lines is always four.
        # the number of anchors may be four pilot anchors controlling one line each,
        # or two arpeggio anchors controlling two lines each.
        # they cannot be mixed. As soon as one type is discovered, this config will be locked to that type.
        # when the anchor type is arpeggio, anchor_num is 0 or 1.
        # refrerences to anchor num that referred to a service, a camera or its pose can still reference anchor num.
        # references to anchor num that were referring to grommet positions or line lengths and speeds,
        # must now refer line numbers 0-3. sending a command to jog a spool or set a line speed must be abstracted through
        # a class that will send the message to the connected server that manages that line.

        if is_power_anchor or is_standard_anchor or is_arp_anchor:
            found_type = common.AnchorType.ARPEGGIO if is_arp_anchor else common.AnchorType.PILOT
            
            if self.config.anchor_type == common.AnchorType.UNSPECIFIED:
                # the first discovered anchor locks the config to an anchor type
                self.config.anchor_type = found_type
                if is_arp_anchor:
                    # replace the four default pilot anchors in the config with two default arp anchors having unset addresses and service names
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
                if anchor_num >= N_ANCHORS[self.config.anchor_type]:
                    # Discovering more that four anchors could be a sign that another robot in the same network is turned on.
                    # We need a way to know that, but for now, you'll have to make sure only one is one at a time while discovering.
                    # After discovery, it should be ok to have more than one on at a time.
                    logger.warning(f"Discovered another {found_type} server on the network, but we already know of {N_ANCHORS[self.config.anchor_type]} {key} {address}")
                    return None
            if self.config.anchors[anchor_num].address != address or self.config.anchors[anchor_num].port != info.port:
                self.config.anchors[anchor_num].num = anchor_num
                self.config.anchors[anchor_num].service_name = key
                self.config.anchors[anchor_num].address = address
                self.config.anchors[anchor_num].port = info.port
                save_config(self.config, self.config_path)

        elif is_standard_gripper or is_arp_gripper:
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
            if kind == anchor_service_name or kind == anchor_power_service_name or kind == arp_anchor_service_name:
                del self.anchors[client.anchor_num]
            elif kind == gripper_service_name or kind == arp_gripper_service_name:
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
            # is everything up the way we want it to be?
            if len([b for b in self.bot_clients.values() if b.connected])==5:
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

        is_power_anchor = name_component == anchor_power_service_name
        is_standard_anchor = name_component == anchor_service_name
        is_standard_gripper = name_component == gripper_service_name
        is_arp_gripper = name_component == arp_gripper_service_name
        is_arp_anchor = name_component == arp_anchor_service_name

        if is_standard_gripper:
            client = RaspiGripperClient(self.config.gripper.address, self.config.gripper.port, self.datastore, self, self.pool, self.stat, self.pe, self.telemetry_env)
            self.gripper_client_connected.clear()
            client.connection_established_event = self.gripper_client_connected
            self.gripper_client = client
            self.pe.set_gripper_type('pilot')
        if is_arp_gripper:
            client = ArpeggioGripperClient(self.config.gripper.address, self.config.gripper.port, self.datastore, self, self.pool, self.stat, self.pe, self.telemetry_env)
            self.gripper_client_connected.clear()
            client.connection_established_event = self.gripper_client_connected
            self.gripper_client = client
            self.pe.set_gripper_type('arp')
        elif is_power_anchor or is_standard_anchor:
            for a in self.config.anchors:
                if a.service_name != service_name:
                    continue
                client = RaspiAnchorClient(a.address, a.port, a.num, self.datastore, self, self.pool, self.stat, self.telemetry_env)
                client.connection_established_event = self.any_anchor_connected
                self.anchors[a.num] = client
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
            if is_power_anchor or is_standard_anchor or is_arp_anchor:
                display_name = f'Anchor {client.anchor_num}'
            elif is_standard_gripper or is_arp_gripper:
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

    async def connect_cloud_telemetry(self):
        ws_protocol_and_host = CONTROL_PLANE_LOCAL
        if self.telemetry_env == 'staging':
            ws_protocol_and_host = CONTROL_PLANE_STAGING
        if self.telemetry_env == 'production':
            ws_protocol_and_host = CONTROL_PLANE_PRODUCTION

        while self.run_command_loop:
            try:
                use_id = self.config.robot_id
                ws_path = f"{ws_protocol_and_host}/telemetry/{use_id}"
                async with websockets.connect(ws_path, max_size=None, open_timeout=10) as websocket:
                    self.cloud_telem_websocket = websocket
                    logger.info(f'Connected to control plane {ws_path}')
                    # send anything that it would need up-front
                    await self.send_setup_telemetry()
                    try:
                        async for message in websocket:
                            r = await self.handle_command(message)
                            if not self.run_command_loop:
                                r = await websocket.close()
                    except ConnectionClosedOK as e:
                        logger.info(f'ConnectionClosedOK from {ws_path}')
                    except ConnectionClosedError as e:
                        logger.error(e)
                    finally:
                        logger.info(f'Disconnected from control plane {ws_path}')
                        self.cloud_telem_websocket = None
                        self.zero_input_velocities()
            except (asyncio.exceptions.CancelledError, websockets.exceptions.ConnectionClosedOK):
                pass # normal close
            except websockets.exceptions.InvalidStatus as e:
                if e.response.status_code == 409:
                    logger.warning(
                        f'Control plane rejected connection (HTTP 409): another robot is '
                        f'already connected with id "{self.config.robot_id}".'
                    )
                else:
                    logger.warning(
                        f'Control plane rejected connection: HTTP {e.response.status_code}'
                    )
                await asyncio.sleep(10) # still could be considered a transient error, but probably not.
            except ConnectionRefusedError:
                logger.warning(f'Connection to control plane refused')
            except websockets.exceptions.InvalidMessage:
                logger.warning('Connection to control plane ended due to invalid message')
            await asyncio.sleep(2)

    def send_ui(self, **kwargs):
        """
        Ensure that the given telemetry item is sent to every connected UI
        keyword args are passed directly to telemetry item, so you can construct one like this

        self.send_ui(pop_message=telemetry.Popup('hello'))
        """
        if len(kwargs.keys()) != 1:
            raise ValueError
        key, msg = list(kwargs.items())[0]

        # mark certain messages with a retain key. the server will resend them to new UIs
        item = telemetry.TelemetryItem(**kwargs)
        if key == 'new_anchor_poses':
            item.retain_key = 'new_anchor_poses'
        if key == 'component_conn_status':
            if msg.is_gripper:
                item.retain_key = f'component_conn_status_g'
            else:
                item.retain_key = f'component_conn_status_{msg.anchor_num}'
        if key == 'video_ready':
            item.retain_key = f'video_ready_{msg.feed_number}'
        if key == 'episode_control' and item.episode_control.status is not None:
            self.last_ep_ctrl_status = item.episode_control.status
            item.retain_key = f'lerobot_status'
        if key == 'swing_cancellation_state':
            item.retain_key = 'swing_cancellation_state'
        if key == 'task_status':
            item.retain_key = 'task_status'

        # Add item to batch
        with self.telemetry_buffer_lock:
            self.telemetry_buffer.append(item)

    async def flush_tele_buffer(self):
        """
        Flush the teloperation buffer. sending all data to all UI clients.
        Normally called within position estimator's 60hz loop
        """
        with self.telemetry_buffer_lock:
            batch = telemetry.TelemetryBatchUpdate(
                robot_id=self.config.robot_id,
                updates=list(self.telemetry_buffer)
            )
            self.telemetry_buffer.clear()
        to_send = bytes(batch)
        # copy list to prevent RuntimeError: Set changed size during iteration
        connected_clients = self.connected_local_clients.copy()
        if self.cloud_telem_websocket:
            connected_clients.add(self.cloud_telem_websocket) # will only be connected when self.telemetry_env is not None
        for ui_websocket in connected_clients:
            try:
                r = await ui_websocket.send(to_send)
            except (ConnectionClosedOK, ConnectionClosedError) as e:
                pass # stale connection

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

        if self.telemetry_env is not None:
            self.cloud_telem = asyncio.create_task(self.connect_cloud_telemetry())

        # statistic counter - measures things like average camera frame latency
        asyncio.create_task(self.stat.stat_main())

        # A task that continuously estimates the position of the gantry
        # remains asleep until at least one anchor connects.
        self.pe_task = asyncio.create_task(self.start_pe_when_ready())

        # main process must own pool, and there's only one. multiple subprocesses may submit work.
        with Pool(processes=3) as pool:
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

            # start a websocket server to accept incoming connections from either a local UI or local Lerobot session
            async with websockets.serve(self.handle_local_client, "127.0.0.1", self.port):
                # await something that will end when the program closes to keep serving and
                # keep zeroconf alive and discovering services.
                try:
                    self.startup_complete.set()

                    if self.telemetry_env == None:
                        message = f'To control visit https://neufangled.com/playroom?robotid=lan on this machine'
                    elif self.telemetry_env == 'local':
                        message = f'To control visit http://localhost:5173/playroom?robotid={self.config.robot_id}'
                    elif self.telemetry_env == 'production':
                        message = f'To control visit https://neufangled.com/playroom?robotid={self.config.robot_id}'
                    elif self.telemetry_env == 'staging':
                        message = f'To control visit https://nf-site-monolith-staging-690802609278.us-east1.run.app/playroom?robotid={self.config.robot_id}'
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
        self.run_command_loop = False
        self.stat.run = False
        self.pe.run = False
        self.pe_task.cancel()
        tasks = [self.pe_task, self.keeper]
        tasks.extend([client.shutdown() for client in self.bot_clients.values()])
        if self.cloud_telem:
            self.cloud_telem.cancel()
            tasks.append(self.cloud_telem)
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

        if isinstance(self.gripper_client, ArpeggioGripperClient):

            # arpeggio gripper. Update finger and wrist speed
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

        elif isinstance(self.gripper_client, RaspiGripperClient):

            # pilot gripper, update winch speed and finger angle
            if line_speed is not None:
                update['aim_speed'] = line_speed # winch
            if finger_speed is not None and abs(finger_speed) > 1.0:
                finger_speed = clamp(finger_speed, -90, 90)
                await self.gripper_client.set_finger_speed(finger_speed)

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

    async def clear_gantry_goal(self):
        self.gantry_goal_pos = None
        self.send_ui(named_position=telemetry.NamedObjectPosition(name='gantry_goal_marker')) # not setting position causes it to be hidden

    async def seek_gantry_goal(self, head_turn=False, auto_altitude=True):
        """
        Move towards a goal position, using the constantly updating gantry position provided by the position estimator
        This is a motion task
        when head_turn, turn gripper to face direction of motion.
        when auto_altitude, room traversal is performed at an ideal gantry altitude
        """
        GOAL_PROXIMITY_M = 0.08
        MAX_SPEED = 0.3 # GANTRY_SPEED_MPS
        ACCEL = 0.15     # m/s^2
        LOOP_SLEEP_S = 0.1
        IDEAL_GANTRY_ALTITUDE = 1.3 # meters. ideal gantry height for room traversal
        CLIMB_RATE = 0.15 # m/s, constant rate of altitude change for auto_altitude
        ALTITUDE_DEADBAND_M = 0.05 # meters, tolerance to avoid hunting around target altitude

        if self.gantry_goal_pos is None:
            return

        # Calculate the distance needed to stop from MAX_SPEED: d = v^2 / (2a)
        braking_distance = (MAX_SPEED**2) / (2 * ACCEL)
        start_pos = self.pe.gant_pos
        current_speed = 0.0
        final_approach = False # latches once True so the altitude target doesn't flip back to cruise
        
        try:
            self.send_ui(named_position=telemetry.NamedObjectPosition(position=fromnp(self.gantry_goal_pos), name='gantry_goal_marker'))
            dist_to_goal = 10
            while self.gantry_goal_pos is not None:
                vector = self.gantry_goal_pos - self.pe.gant_pos
                dist_to_goal = np.linalg.norm(vector)
                dist_from_start = np.linalg.norm(self.pe.gant_pos - start_pos)

                if dist_to_goal < GOAL_PROXIMITY_M:
                    break

                # Calculate target speed based on distance from start (ramp up)
                # and distance to goal (ramp down)
                # v = sqrt(2 * a * d)
                ramp_dist_to_goal = np.linalg.norm(vector[:2]) if auto_altitude else dist_to_goal
                speed_ramp_up = np.sqrt(2 * ACCEL * max(dist_from_start, 0.01))
                speed_ramp_down = np.sqrt(2 * ACCEL * ramp_dist_to_goal)

                # Target speed is the lowest of the ramps or the max allowable speed
                target_speed = min(speed_ramp_up, speed_ramp_down, MAX_SPEED)

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
                    goal_altitude = self.gantry_goal_pos[2]
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

            logger.info(f'Goal reached {tuple(self.gantry_goal_pos)}')
        except asyncio.CancelledError:
            logger.debug('Goal move cancelled')
            raise
        finally:
            self.slow_stop_all_spools()
            await self.clear_gantry_goal()

    async def send_line_speed(self, line_no, speed, jog=False):
        # send the line speed to the client that controls that line
        # when jog==True, speed is interpreted as a length in meters by which to lengthen the line
        command = 'jog' if jog else 'aim_speed'
        if self.config.anchor_type == common.AnchorType.PILOT:
            if line_no in self.anchors:
                r = await self.anchors[line_no].send_commands({command: speed})
        elif self.config.anchor_type == common.AnchorType.ARPEGGIO:
            if line_no//2 in self.anchors:
                spool_no = line_no%2
                # we consider the lower line number to be the direct line
                r = await self.anchors[line_no//2].send_commands({command: (speed, spool_no)})

    async def set_line_tension_target(self, line_no, value):
        """Set (or clear, with None) the onboard two-sided tension hold target in newtons
        for one arpeggio line. The onboard loop then holds that line at the target."""
        if self.config.anchor_type == common.AnchorType.ARPEGGIO and line_no//2 in self.anchors:
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
        speed_limit = 0.35
        if self.config.anchor_type == common.AnchorType.PILOT:

            # On pilot stringman, also enforce a height dependent speed limit on the total combined velocity.
            # the reason being that as gantry height approaches anchor height, the line tension increases exponentially,
            # and a slower speed is need to maintain enough torque from the stepper motors.
            # The speed limit is proportional to how far the gantry hangs below a level 10cm below the average anchor.
            # This makes the behavior consistent across installations of different heights.
            hang_distance = np.mean(self.pe.anchor_points[:, 2]) - starting_pos[2]
            speed_limit = clamp(0.28 * (hang_distance - 0.1), 0.01, 0.25)
            # If the combined total speed exceeds the limit, scale the vector down
        elif self.config.anchor_type == common.AnchorType.ARPEGGIO:
            speed_limit = 0.35
            if self.feature_supported("speed_0.45"):
                speed_limit = 0.45

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

    def _ortho_worker(self, ortho_floor_vs, heatmap_floor_vs):
        """
        Sync thread driven by self.ortho_event, which anchor frame_resizer_loops set on every
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
                    if c.last_frame_resized is not None and c.anchor_num in self.config.preferred_cameras
                ]
                if not valid_clients:
                    continue

                # Pass heatmaps=None when the target model isn't producing them so
                # generate_orthographic_floor_maps skips all heatmap warp work and
                # only computes the ortho BGR floor projection.
                heatmaps = self.last_heatmaps_np
                if heatmaps is not None and len(heatmaps) != len(valid_clients):
                    heatmaps = None  # stale/mismatched batch; skip the heatmap channel

                ortho_heatmap, ortho_bgr = generate_orthographic_floor_maps(
                    valid_clients, heatmaps, self.config.camera_cal,
                    map_size_px=1000, map_extent_meters=EXTENT,
                )
                self.last_ortho_bgr = ortho_bgr
                if ortho_heatmap is not None:
                    self.last_ortho_heatmap = ortho_heatmap

                if ortho_floor_vs is not None:
                    ortho_floor_vs.send_frame(cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2RGB))
                if heatmap_floor_vs is not None and ortho_heatmap is not None:
                    heatmap_floor_vs.send_frame(
                        cv2.applyColorMap((ortho_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    )
            except Exception:
                logger.exception('_ortho_worker iteration failed')

    async def run_perception(self):
        """
        Orthographic floor projection and target heatmap inference.
        run_ortho and target model inference are independent; the target model is loaded at
        runtime via SetTargetModel control messages.
        """
        LOOP_DELAY = 0.1
        FIND_TARGETS_EVERY = 5
        EXTENT = 5.0

        # wait until at least one preferred camera is producing frames
        logging.info('waiting for camera frames')
        while True:
            await asyncio.sleep(1)
            have_frames = (
                (self.gripper_client is not None and self.gripper_client.last_frame_resized is not None)
                or any(
                    anum in self.config.preferred_cameras and c.last_frame_resized is not None
                    for anum, c in self.anchors.items()
                )
            )
            if have_frames:
                break

        ortho_floor_vs = None
        heatmap_floor_vs = None
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
            )
            ortho_floor_vs.start()
            self.ortho_streamers = [(ortho_floor_vs, 3)]
            if self.stream_heatmap:
                heatmap_floor_vs = NfVideoStreamer(
                    width=1000, height=1000, fps=10,
                    mjpeg_port=8748,
                    stream_path=f'stringman/{self.config.robot_id}/4',
                    telemetry_env=self.telemetry_env,
                    on_ready=_make_on_ready(4),
                )
                heatmap_floor_vs.start()
                self.ortho_streamers.append((heatmap_floor_vs, 4))

        ortho_thread = threading.Thread(
            target=self._ortho_worker,
            args=(ortho_floor_vs, heatmap_floor_vs),
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

            # Lazy imports: only reached once a target model is loaded, so torch
            # stays off the startup path. Both are import-cached after first use.
            import torch
            from nf_robot.ml.target_heatmap import extract_targets_from_heatmap, HM_IMAGE_RES

            valid_anchor_clients = [
                c for c in self.anchors.values()
                if c.last_frame_resized is not None and c.anchor_num in self.config.preferred_cameras
            ]
            if not valid_anchor_clients:
                continue

            img_tensors = [
                torch.from_numpy(cv2.resize(c.last_frame_resized, HM_IMAGE_RES, interpolation=cv2.INTER_AREA))
                     .permute(2, 0, 1).float() / 255.0
                for c in valid_anchor_clients
            ]
            batch = torch.stack(img_tensors).to(self._device)

            def infer_sync():
                with torch.no_grad():
                    return self.target_model(batch).squeeze(1).cpu().numpy()

            heatmaps_np = await asyncio.to_thread(infer_sync)
            self.last_heatmaps_np = heatmaps_np

            ortho_heatmap = self.last_ortho_heatmap
            if ortho_heatmap is None:
                continue

            results = extract_targets_from_heatmap(ortho_heatmap)
            if len(results) > 0:
                targets2d = (results[:, :2] + np.array([-0.5, -0.5])) * EXTENT
                floor_targets = [
                    {'position': np.array([p[0], p[1], 0]), 'dropoff': 'hamper'}
                    for p in targets2d
                    if self.pe.point_inside_work_area_2d(p)
                ]
            else:
                floor_targets = []
            self.target_queue.add_ai_targets(floor_targets)
            self.send_tq_to_ui()

        if self.run_ortho:
            ortho_floor_vs.stop()
            if heatmap_floor_vs is not None:
                heatmap_floor_vs.stop()

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

        # TODO if no lerobot session is connected with an appropriate model or --arp_grasp is false, this cannot work.
        # check prereqs and warn user with popup message or just start necessary model.

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
                        self.gantry_goal_pos = None
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

                self.gantry_goal_pos = goal_pos
                if gtask is None or gtask.done():
                    gtask = asyncio.create_task(self.seek_gantry_goal())
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
                self.gantry_goal_pos = drop_point + GANTRY_HEIGHT_OVER_DROPOFF
                await self.seek_gantry_goal()
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
            await self.clear_gantry_goal()

    async def execute_grasp(self):
        """Try to grasp whatever is directly below the gripper"""
        if isinstance(self.gripper_client, ArpeggioGripperClient):
            # A lerobot session may be driving from our own subprocess or connected remotely
            # through the prod telemetry relay, so we can't tell locally if one is present.
            # lerobot_grasp broadcasts the eval-start and returns None if no session answers,
            # in which case we fall back to the older centering model.
            result = await self.lerobot_grasp()
            if result is not None:
                return result
            if self.centering_model is None:
                await self._load_centering_model()
            return await self.arp_execute_grasp()
        else:
            return await self.pilot_execute_grasp()

    async def pilot_execute_grasp(self):
        FINGER_LENGTH = 0.1 # length between rangefinder and floor when fingers touch in meters
        HALF_VIRTUAL_FOV = model_constants.rpi_cam_3_fov * SF_SCALE_FACTOR / 2 * (np.pi/180)
        DOWNWARD_SPEED = -0.06
        VISUAL_CONF_THRESHOLD = 0.1 # level below which we give up on the target
        COMMIT_HEIGHT = 0.3 # height below which giving up due to visual disconfidence is not allowed.
        LAT_TRAVEL_FRACTION = 0.75 # try to finish lateral travel by this fraction of the time spent travelling downwards
        LAT_SPEED_ADJUSTMENT = 5.00 # final adjustment to lateral speed
        LOOP_DELAY = 0.1
        PRESSURE_SENSE_WAIT = 2.0

        smooth_grip_angle = self.grip_angle

        try:
            asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
            attempts = 3
            while not self.pe.holding and attempts > 0 and self.run_command_loop:
                attempts -= 1
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))

                # move laterally until target is centered
                # at the same time, move downward until tip is detected.

                nothing_seen_countdown = 15
                self.pe.tip_over.clear()
                while (self.predicted_lateral_vector is not None and not self.pe.tip_over.is_set()):
                    distance_to_floor = self.datastore.range_record.getLast()[1]
                    if distance_to_floor < FINGER_LENGTH:
                        logger.debug(f'Stop going down, distance to floor is {distance_to_floor}')
                        break

                    if self.gripper_sees_target < VISUAL_CONF_THRESHOLD and distance_to_floor > COMMIT_HEIGHT:
                        nothing_seen_countdown -= 1
                        if nothing_seen_countdown == 0:
                            logger.debug('Nothing seen during centering loop')
                            break
                    else:
                        nothing_seen_countdown = 15

                    # calculate eta to the floor using laser range, we want to finish lateral travel at 0.75 of that eta
                    lat_travel_seconds = (distance_to_floor-FINGER_LENGTH)/(-DOWNWARD_SPEED)*LAT_TRAVEL_FRACTION
                    lateral_vector = np.zeros(3)
                    if lat_travel_seconds > 0:
                        # determine which direction we'd have to move laterally to center the object
                        # you get a normalized u,v coordinate in the [-1,1] range
                        # for now assume that the up direction in the gripper image is -Y in world space 
                        # stabilize_frame produced this direction and I think it depends on the compass.
                        # the direction in world space depends on how the user placed the origin card on the ground
                        # we need to capture a number during calibration to relate these two.
                        # +1 is the edge of the image. how far laterally that would be depends on how far from the ground the gripper is.
                        pred_vector = self.predicted_lateral_vector
                        pred_vector[1] *= -1
                        # lateral distance to object
                        lateral_vector = np.sin(pred_vector * HALF_VIRTUAL_FOV) * distance_to_floor
                        # lateral distance in meters
                        lateral_distance = np.linalg.norm(lateral_vector)
                        # speed to travel that lateral distance in lat_travel_seconds
                        lateral_speed = lateral_distance / lat_travel_seconds * LAT_SPEED_ADJUSTMENT
                    else:
                        # once we get too close, go straight down, stop relying on the camera
                        lateral_speed = 0
                    lateral_vector *= lateral_speed

                    logger.debug(f'Moving {[lateral_vector[0],lateral_vector[1],DOWNWARD_SPEED]}')
                    await self.move_direction_speed([lateral_vector[0],lateral_vector[1],DOWNWARD_SPEED])

                    try:
                        # the normal sleep on this loop would be LOOP_DELAY s, but if tip is detected
                        # we want to stop immediately.
                        await asyncio.wait_for(self.pe.tip_over.wait(), LOOP_DELAY)
                        logger.debug('Detected tip over, must be floor')
                        break
                    except TimeoutError:
                        pass

                self.slow_stop_all_spools()
                self.pe.tip_over.clear()

                if nothing_seen_countdown == 0:
                    logger.debug('Nothing seen')
                    continue # find new target?

                logger.info('Close gripper')
                await self.gripper_client.send_commands({'set_finger_angle': CLOSED})
                logger.debug(f'Wait up to {PRESSURE_SENSE_WAIT} seconds for pad to sense object.')
                try:
                    await asyncio.wait_for(self.pe.finger_pressure_rising.wait(), PRESSURE_SENSE_WAIT)
                    self.pe.finger_pressure_rising.clear()
                except TimeoutError:
                    pressure = self.datastore.finger.getLast()[2]
                    logger.debug(f'Did not detect a successful hold. pressure=({pressure}) open and go back up high enough to get a view of the object')
                    # move up slowly at first, till fingers just touch ground and we are veritical. this keeps unwanted swinging to a minimum
                    await self.move_direction_speed([0,0,0.06])
                    await asyncio.sleep(1.0)
                    # now move up a little faster in a slightly random direction
                    direction = np.concatenate([np.random.uniform(-0.025, 0.025, (2)), [0.12]])
                    await self.move_direction_speed(direction)
                    asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
                    await asyncio.sleep(2.0)
                    self.slow_stop_all_spools()
                    continue
                logger.info('Successful grasp')
                return True
            logger.info(f'Gave up on grasp after {attempts} attempts. self.pe.holding={self.pe.holding}')
            return False

        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()

    async def arp_execute_grasp(self):
        """Try to grasp whatever is directly below the gripper"""
        FINGER_LENGTH = 0.1 # length between rangefinder and floor when fingers touch in meters
        FLOOR_GRIPPER_HEIGHT = 0.11 # distance above floor (gripper origin) when grasp should be started
        RANGE_ITEM = 0.04 # range to item below which grip should be started
        HALF_VIRTUAL_FOV = model_constants.rpi_cam_3_wide_fov * SF_SCALE_FACTOR / 2 * (np.pi/180)
        DOWNWARD_SPEED = -0.07
        VISUAL_CONF_THRESHOLD = 0.1 # level below which we give up on the target
        COMMIT_HEIGHT = 0.3 # height below which giving up due to visual disconfidence is not allowed.
        LAT_TRAVEL_FRACTION = 0.75 # try to finish lateral travel by this fraction of the time spent travelling downwards
        LAT_SPEED_ADJUSTMENT = 5.00 # final adjustment to lateral speed. so huge because network outputs small values (why?)
        LOOP_DELAY = 0.1
        PRESSURE_SENSE_WAIT = 10.0
        NUM_ATTEMPTS = 3
        CLOSING_FINGER_SPEED = 30
        WRIST_SMOOTH_FACTOR = 0.9

        smooth_grip_angle = self.grip_angle

        try:
            attempts = NUM_ATTEMPTS
            while not self.pe.holding and attempts > 0 and self.run_command_loop:
                attempts -= 1
                logger.debug(f'Open fingers to {OPEN} to clear camera')
                asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))

                # move laterally until target is centered
                # at the same time, move downward until tip is detected.

                nothing_seen_countdown = 15
                approach_timeout = time.time()+10
                self.pe.tip_over.clear()
                while (self.predicted_lateral_vector is not None and not self.pe.tip_over.is_set() and time.time() < approach_timeout):
                    range_to_target = self.datastore.range_record.getLast()[1]
                    # compare this rangefinder distance to the distance estimated from other methods
                    gripper_height = self.pe.grip_pose[1][2]

                    # for bulky objects, we want to close range_to_target to about zero to get the fingers all the way around
                    # for small objects, we don't want to, we can't get that low, the fingers would touch the floor and the object
                    # would still be a few cm away from the rangefinder. 

                    logger.debug(f'range_to_target {range_to_target} gripper_height = {gripper_height}')
                    if range_to_target < RANGE_ITEM or gripper_height < FLOOR_GRIPPER_HEIGHT:
                        logger.debug(f'Reached target at height {gripper_height} and range {range_to_target}')
                        break

                    if self.gripper_sees_target < VISUAL_CONF_THRESHOLD and range_to_target > COMMIT_HEIGHT:
                        nothing_seen_countdown -= 1
                        if nothing_seen_countdown == 0:
                            logger.debug('Nothing seen during centering loop')
                            break
                    else:
                        nothing_seen_countdown = 15

                    # calculate eta to the floor using laser range, we want to finish lateral travel at 0.75 of that eta
                    lat_travel_seconds = (range_to_target-FINGER_LENGTH)/(-DOWNWARD_SPEED)*LAT_TRAVEL_FRACTION
                    lateral_vector = np.zeros(2)
                    if lat_travel_seconds > 0:
                        # determine which direction we'd have to move laterally to center the object
                        # you get a normalized u,v coordinate in the [-1,1] range
                        # for now assume that the up direction in the gripper image is -Y in world space 
                        # stabilize_frame produced this direction and I think it depends on the compass.
                        # the direction in world space depends on how the user placed the origin card on the ground
                        # we need to capture a number during calibration to relate these two.
                        # +1 is the edge of the image. how far laterally that would be depends on how far from the ground the gripper is.
                        pred_vector = self.predicted_lateral_vector
                        pred_vector[1] *= -1
                        # lateral distance to object
                        lateral_vector = np.sin(pred_vector * HALF_VIRTUAL_FOV) * range_to_target
                        # lateral distance in meters
                        lateral_distance = np.linalg.norm(lateral_vector)
                        # speed to travel that lateral distance in lat_travel_seconds
                        lateral_speed = lateral_distance / lat_travel_seconds * LAT_SPEED_ADJUSTMENT
                    else:
                        # once we get too close, go straight down, stop relying on the camera
                        lateral_speed = 0
                    lateral_vector *= lateral_speed

                    # rotate later component of direction from gripper frame into room frame
                    lateral_vector = rotate_vector(lateral_vector, -self.gripper_client.get_spin())

                    await self.move_direction_speed([lateral_vector[0],lateral_vector[1],DOWNWARD_SPEED])

                    # move wrist to predicted grip angle with smoothing
                    smooth_grip_angle = smooth_grip_angle*WRIST_SMOOTH_FACTOR + self.grip_angle*(1-WRIST_SMOOTH_FACTOR)
                    await self.gripper_client.send_commands({'set_wrist_angle': smooth_grip_angle/np.pi*180})

                    try:
                        # the normal sleep on this loop would be LOOP_DELAY s, but if tip is detected
                        # we want to stop immediately.
                        await asyncio.wait_for(self.pe.tip_over.wait(), LOOP_DELAY)
                        logger.debug('Detected tip over, must be floor')
                        break
                    except TimeoutError:
                        pass

                self.slow_stop_all_spools()
                self.pe.tip_over.clear()

                if nothing_seen_countdown == 0:
                    logger.debug('Nothing seen')
                    continue # find new target?

                logger.info('Close gripper')
                end_time = time.time() + PRESSURE_SENSE_WAIT
                self.pe.finger_pressure_rising.clear()

                await self.gripper_client.send_commands({'set_finger_speed': CLOSING_FINGER_SPEED})
                # finger speed commands take effect for 200ms only. they must be sent repeatedly.
                t, angle, pressure = self.datastore.finger.getLast()
                while time.time() < end_time and not self.pe.finger_pressure_rising.is_set() and angle < CLOSED:
                    await asyncio.sleep(0.03)
                    await self.gripper_client.send_commands({'set_finger_speed': CLOSING_FINGER_SPEED})
                    t, angle, pressure = self.datastore.finger.getLast()
                logger.debug(f'End grip finger_pressure_rising={self.pe.finger_pressure_rising.is_set()} angle={self.datastore.finger.getLast()[1]}')
                await self.gripper_client.send_commands({'set_finger_speed': 0})

                if not self.pe.finger_pressure_rising.is_set():
                    pressure = self.datastore.finger.getLast()[2]
                    logger.debug(f'Did not detect a successful hold, pressure=({pressure}) open and go back up high enough to get a view of the object')
                    # move up slowly at first, till fingers just touch ground and we are veritical. this keeps unwanted swinging to a minimum
                    await self.move_direction_speed([0,0,0.06])
                    await asyncio.sleep(1.0)
                    asyncio.create_task(self.gripper_client.send_commands({'set_finger_angle': OPEN}))
                    # now move up a little faster in a slightly random direction
                    direction = np.concatenate([np.random.uniform(-0.025, 0.025, (2)), [0.12]])
                    await self.move_direction_speed(direction)
                    await asyncio.sleep(2.0)
                    self.slow_stop_all_spools()
                    continue

                self.pe.finger_pressure_rising.clear()
                logger.info('Successful grasp')
                # slowly at first
                await self.move_direction_speed(np.array([0,0,0.05]))
                await asyncio.sleep(1.0)
                # and then all at once
                await self.move_direction_speed(np.array([0,0,0.15]))
                await asyncio.sleep(2.0)
                logger.info('Stop moving')
                self.slow_stop_all_spools()
                return True
            logger.info(f'Gave up on grasp after {NUM_ATTEMPTS-attempts} attempts. self.pe.holding={self.pe.holding}')
            return False

        except asyncio.CancelledError:
            raise
        finally:
            self.slow_stop_all_spools()

    async def _load_centering_model(self):
        import torch
        from huggingface_hub import hf_hub_download
        from nf_robot.ml.centering import CenteringNet
        DEVICE = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = DEVICE
        CENTERING_MODEL_REPOID = "naavox/centering"

        if DEVICE == "cpu":
            logger.warning("Refusing to load centering model on CPU; hardware acceleration required.")
            self.centering_model = None
            self.send_ui(pop_message=telemetry.Popup(
                message="The arp grasp (centering model) cannot be used without some kind of "
                        "hardware acceleration. Loading was aborted because the torch device is CPU."
            ))
            return

        def load_sync():
            if self.local_models:
                center_path = "models/square_centering.pth"
            else:
                center_path = hf_hub_download(repo_id=CENTERING_MODEL_REPOID, filename="square_centering.pth")
            logger.info(f"Loading centering model from {center_path}...")
            c_model = CenteringNet().to(DEVICE)
            c_model.load_state_dict(torch.load(center_path, map_location=DEVICE))
            c_model.eval()
            return c_model

        self.centering_model = await asyncio.to_thread(load_sync)

    async def _load_target_model(self):
        import torch
        from huggingface_hub import hf_hub_download
        from nf_robot.ml.target_heatmap import TargetHeatmapNet
        DEVICE = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = DEVICE
        TARGETING_MODEL_REPOID = "naavox/targeting"

        if DEVICE == "cpu":
            logger.warning("Refusing to load targeting model on CPU; hardware acceleration required.")
            self.target_model = None
            self.send_ui(pop_message=telemetry.Popup(
                message="Automatic target identification (targeting model) cannot be used without "
                        "some kind of hardware acceleration. Loading was aborted because the torch "
                        "device is CPU."
            ))
            return

        def load_sync():
            if self.local_models:
                target_path = "models/target_heatmap.pth"
            else:
                target_path = hf_hub_download(repo_id=TARGETING_MODEL_REPOID, filename="target_heatmap.pth")
            logger.info(f"Loading targeting model from {target_path}...")
            t_model = TargetHeatmapNet().to(DEVICE)
            t_model.load_state_dict(torch.load(target_path, map_location=DEVICE))
            t_model.eval()
            return t_model

        self.target_model = await asyncio.to_thread(load_sync)

    async def _handle_set_target_model(self, item: control.SetTargetModel):
        if item.action == control.TargetModelAction.TARGET_MODEL_DISABLE:
            self.target_model = None
            logger.info('Target model disabled')
        elif item.action == control.TargetModelAction.TARGET_MODEL_ENABLE_DEFAULT:
            logger.info('Loading default target model...')
            await self._load_target_model()
            logger.info('Target model ready')

    async def lerobot_grasp(self):
        """
        Execute a grasp on an arp gripper using a lerobot ACT policy.
        End the episode either when a timeout is reached, when motion ceases for some time, or when a grasp condition is reached.
        A grasp condition is a certain amount of force being exerted by the fingers while being at a certain altitude off the floor.

        Returns True/False for grasp success once a session takes over, or None if no session
        answered the eval-start (so the caller can fall back to the centering model).

        A seperate process must be connected to the telemetry stream to manage the act policy at this time. It can be started with

        python -m nf_robot.ml.stringman_lerobot eval   --robot_id=lan   --server_address=ws://localhost:4245   --policy_id=outputs/train/grasp_remote_act_eggs_2/checkpoints/last/pretrained_model/   --dataset_id=naavox/grasping_dataset_eggs_fix
        """
        SESSION_PING_TIMEOUT_S = 10  # how long to wait for any session to answer the eval-start

        self.pe.finger_pressure_rising.clear()
        self.lerobot_session_status_event.clear()
        try:
            # Broadcast the eval-start: any listening lerobot session (local subprocess or one
            # connected remotely through the relay) starts controlling and answers with a status.
            self.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.EVAL_START))
            try:
                await asyncio.wait_for(self.lerobot_session_status_event.wait(), timeout=SESSION_PING_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.debug(f'No lerobot session answered the eval-start within {SESSION_PING_TIMEOUT_S}s; no session active.')
                return None

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

    def _handle_collect_images(self):
        if self.run_collect_images:
            self.run_collect_images = False # ends the task
        else:
            self.run_collect_images = True
            self.gip_task = asyncio.create_task(self.collect_images())

    async def collect_images(self):
        """Collects data for the centering network"""
        while self.run_command_loop and self.run_collect_images:
            if self.gripper_client.last_frame_resized is not None:
                logger.debug(f'Gripper frame shape: {self.gripper_client.last_frame_resized.shape}')
                rgb_image = cv2.cvtColor(self.gripper_client.last_frame_resized, cv2.COLOR_BGR2RGB)
                capture_gripper_image(rgb_image, gripper_occupied=self.pe.holding)
            else:
                logger.debug('No resized frame available from gripper')
            await asyncio.sleep(1)

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
    parser.add_argument("--no_ortho", action="store_true", help="Disable orthographic floor projection and its video streams")
    parser.add_argument("--stream_heatmap", action="store_true", help="Generate and stream the target heatmap video feed (off by default)")
    parser.add_argument("--auto_start", action="store_true", help="Automatically unpark and start cleaning when all components connect")
    parser.add_argument("--local_models", action="store_true", help="Use local models from models/ rather than downloading the production models from huggingface")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s %(message)s')
        logging.getLogger('nf_robot').setLevel(logging.DEBUG)

    async def run_async():
        runner = AsyncObserver(
            False,
            args.config,
            telemetry_env=args.telemetry_env,
            run_ortho=(not args.no_ortho),
            stream_heatmap=args.stream_heatmap,
            auto_start=args.auto_start,
            local_models=args.local_models,
            debug=args.debug
        )

        # Idempotent stop trigger. Runs as a signal-handler callback on the event
        # loop thread, so it must not block: schedule the telemetry-socket abort
        # for later instead of time.sleep()-ing on the loop.
        def stop():
            runner.run_command_loop = False
            def _abort_telem():
                if runner.cloud_telem_websocket is not None:
                    runner.cloud_telem_websocket.transport.abort()
            asyncio.get_running_loop().call_later(0.5, _abort_telem)

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
