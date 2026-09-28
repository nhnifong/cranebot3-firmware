"""Lerobot: running a recording or eval session as a subprocess, and grasping with a policy."""

import asyncio
import logging
import re
import sys
import time

from nf_robot.generated.nf import common, control
from nf_robot.host.maneuver import Maneuver, control_item

logger = logging.getLogger(__name__)


class Lerobot(Maneuver):
    name = 'lerobot'
    title = 'Lerobot'

    def __init__(self, ob, use_for_grasp=False):
        super().__init__(ob)
        # --lerobot_grasp: grasp with a connected policy session rather than the visual
        # servoing model, falling back to servoing when no session answers
        self.use_for_grasp = use_for_grasp
        self.process_task = None
        self.process_pid = None
        # the latest status any session reported, replayed to peers that connect later
        self.last_status = common.LerobotStatus.NA
        # fires whenever any lerobot session (our own subprocess or one connected remotely
        # through the telemetry relay) reports a status. Used to detect whether a session is
        # actually listening after we broadcast an eval-start.
        self.session_status_event = asyncio.Event()

    def send_setup_telemetry(self):
        config = self.ob.config
        if self.process_task is None or self.process_task.done():
            self.last_status = common.LerobotStatus.NA
        if isinstance(self.last_status, common.LerobotSessionStatus):
            status = self.last_status
        else:
            status = common.LerobotSessionStatus(
                status=self.last_status,
                policy_repo_id=config.last_lerobot_policy,
                dataset_repo_id=config.last_lerobot_dataset_repo_id,
            )
        self.ob.send_ui(episode_control=common.EpisodeControl(
            status=status,
            prompt=config.last_lerobot_prompt,
        ))

    def on_stop_all(self):
        # If lerobot scripts are connected this must also stop them
        self.ob.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.ABANDON))

    @control_item('episode_control')
    def on_episode_control(self, data: common.EpisodeControl):
        if data.prompt:
            self.ob.config.last_lerobot_prompt = data.prompt
        # A status here means some lerobot session is alive and answering, wherever it's connected.
        if data.status is not None:
            self.last_status = data.status
            self.session_status_event.set()
        # forward episode control events back to all telemetry listeners
        self.ob.send_ui(episode_control=data)
        asyncio.create_task(self.ob.flush_tele_buffer())

    @control_item('manage_lerobot_session')
    def start_session(self, item: control.ManageLerobotSession):
        self.process_task = self.spawn(self.run_process(item), name='lerobot_process')

    def start_eval_session(self, repo_id):
        self.start_session(control.ManageLerobotSession(
            action=control.LerobotSessionAction.START_EVAL, repo_id=repo_id))

    async def run_process(self, item: control.ManageLerobotSession):
        if self.process_pid is not None:
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
            self.ob.config.last_lerobot_dataset_repo_id = repo_id
        elif action == control.LerobotSessionAction.START_EVAL:
            func_name = 'eval_until_disconnected'
            self.ob.config.last_lerobot_policy = repo_id

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
            f"from nf_robot.ml.lerobot.stringman import {func_name}; "
            f"{func_name}('{tele_addr}', '{repo_id}', '{self.ob.robot_id()}'{up})"
        ]

        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        logger.info(f"Lerobot process started with PID: {process.pid}")
        self.process_pid = process.pid

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
            self.process_pid = None

    async def session_connected(self, timeout=2) -> bool:
        """
        Broadcast a ping and see whether any lerobot session (local subprocess or one
        connected remotely through the relay) answers with a status within `timeout` seconds.
        """
        self.session_status_event.clear()
        self.ob.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.PING))
        try:
            await asyncio.wait_for(self.session_status_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.debug(f'No lerobot session answered the ping within {timeout}s; no session active.')
            return False

    async def grasp(self):
        """
        Execute a grasp on an arp gripper using a lerobot ACT policy.
        End the episode either when a timeout is reached, when motion ceases for some time, or when a grasp condition is reached.
        A grasp condition is a certain amount of force being exerted by the fingers while being at a certain altitude off the floor.

        Returns True/False for grasp success once a session takes over, or None if no session
        answered the ping (so the caller can fall back to the visual servoing model).

        A seperate process must be connected to the telemetry stream to manage the act policy at this time. It can be started with

        python -m nf_robot.ml.lerobot.stringman eval   --robot_id=lan   --server_address=ws://localhost:4245   --policy_id=outputs/train/grasp_remote_act_eggs_2/checkpoints/last/pretrained_model/   --dataset_id=naavox/grasping_dataset_eggs_fix
        """
        self.ob.reset_finger_pressure_rising()
        try:
            if not await self.session_connected():
                return None

            # A session is listening; tell it to start controlling.
            self.ob.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.EVAL_START))

            timeout = time.time() + 30
            lifted = False
            applying_force = False
            while not (lifted and applying_force) and time.time() < timeout:
                await asyncio.sleep(0.2)
                applying_force = self.ob.finger_pressure_rose()
                gripper_height = self.ob.gripper_position()[2]
                lifted = gripper_height > 0.4
            logger.debug(f'Ended grasp lifted={lifted} applying_force={applying_force} time_rem={timeout - time.time():.1f}s')
            # return value indicates whether grasp was successful
            # todo future models will predict grasp success on their own
            return lifted # and applying_force
        except asyncio.CancelledError:
            raise
        finally:
            self.ob.send_ui(episode_control=common.EpisodeControl(command=common.EpCommand.EVAL_STOP))
            await asyncio.sleep(0.01)
            self.ob.slow_stop_all_spools()
