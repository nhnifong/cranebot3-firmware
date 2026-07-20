import asyncio
from asyncio.subprocess import PIPE, STDOUT
import os
import subprocess
import signal
import websockets
from websockets.exceptions import (
    ConnectionClosedOK,
    ConnectionClosedError,
)
import json
import importlib.metadata
import zeroconf
from zeroconf.asyncio import (
    AsyncZeroconf,
)
import time
import re
import logging

from nf_robot.robot.forget_wifi import forget_all_wifi_networks
from nf_robot.common.util import get_local_ip

# using libav makes it possible to send a containerized stream with pts
# hardware h264 encoding is still used as long as resolution is below 1080
# this requires rpicam-apps (not present in lite OS image)
# the --framerate value here is just the default; build_stream_command() rewrites it
# from self.stream_framerate_conf_key's config var each time rpicam-vid is (re)launched.
stream_command = [
    "/usr/bin/rpicam-vid", "-t", "0", "-n",
    "--width=1920", "--height=1080",
    "--framerate=20",
    "-o", "tcp://0.0.0.0:8888?listen=1",
    "--codec", "libav",
    "--libav-format", "mpegts",
    "--vflip", "--hflip",
    "--autofocus-mode", "manual",
    "--lens-position", "0.1",
    "--low-latency",
    "--bitrate", "520kbps"
]

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])') # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
# the line we are looking for looks like this
#Output #0, mpegts, to 'tcp://0.0.0.0:8888?listen=1':
ready_line_re = re.compile(r"Output #0, mpegts, to 'tcp://([^:]+):(\d+)\?listen=1':")

# offset in seconds between the appearance of the ready line and the zero point of the DTS times in the stream container.
# determined experimentally by running experiments/measure_dts_zero_point.py on the rpi
dts_zero_offset = 0.719379

# values that can be overridden by the controller
default_conf = {
    # delay in seconds between updates sent on websocket during normal operation
    'RUNNING_WS_DELAY': 1/25,
}

class RobotComponentServer:
    def __init__(self):
        self.conf = default_conf.copy()
        self.run_server = True
        # a dict of update to be flushed periodically to the websocket
        self.update = {}
        self.ws_delay = self.conf['RUNNING_WS_DELAY']
        self.rpicam_process = None
        # Name of the conf var that controls this server's rpicam-vid --framerate, or None if
        # this server's stream_command doesn't expose one / shouldn't be runtime-adjustable.
        # Subclasses opt in so different component types (anchor vs gripper) can be broadcast
        # framerate changes independently instead of sharing one var.
        self.stream_framerate_conf_key = None
        # framerate value baked into the currently running rpicam_process, if any.
        # compared against self.conf to decide whether a config change needs a restart.
        self.running_framerate = None
        # the currently running stream_video task, so it can be stopped before a firmware update
        self.stream_video_task = None
        self.zc = None # zerconf instance.
        self.mock_camera_port = None
        self.extra_tasks = []
        self.stream_command = stream_command # subclasses may override
        self.have_client = False
        self.reset_wifi_event = asyncio.Event()
        self.wait_reset_task = None

    async def stream_measurements(self, ws):
        """
        stream line length measurements to the provided websocket connection
        as long as it exists
        """
        logging.info('start streaming measurements')
        while True:
            if self.spooler is not None:
                # add line lengths
                meas = self.spooler.popMeasurements()
                if len(meas) > 0:
                    if len(meas) > 50:
                        meas = meas[:50]
                    self.update['line_record'] = meas

            self.readOtherSensors()

            # send on websocket
            if self.update != {}:
                await ws.send(json.dumps(self.update))
            self.update = {}

            # chill
            await asyncio.sleep(self.ws_delay)

    async def stream_video(self, websocket):
        # keep rpicam-vid running until this task is cancelled by the client disconnecting
        while True:

            # make sure websocket is alive by running ping before starting rpicam-vid again.
            # if connected, this returns the connection latency.
            # if not connected, trying to get the result of the future throws a connection closed exception
            if not await websocket.ping():
                return

            if self.mock_camera_port is not None:
                # in a unit test, use mock camera. it's already running, just tell the client to connect to it
                print(f'Anchor server is configured to use mock camera on localhost:{self.mock_camera_port}')
                self.update['video_ready'] = (self.mock_camera_port, time.time())
                # normally the only other thing this task needs to do is watch the output of rpicam-vid and collect information
                # indiciating the wall time of the DTS zero point
                # this behavior is not at part of the test and the client will receive the default of now
                result = await asyncio.Future()
            else:
                try:
                    logging.info('Restarting rpi-cam_vid')
                    result = await self.run_rpicam_vid()
                    # if it stops, we'll have to wait a second or two for the OS to free the port it listens on
                    await asyncio.sleep(5)
                except FileNotFoundError:
                    # we may be running in a test. In this case stop attempting to run rpicam-vid.
                    # the client will never receive the message indicating it should connect to video.
                    logging.warning('/usr/bin/rpicam-vid does not exist on this system')
                    return
                except asyncio.CancelledError as e:
                    logging.info("Killing rpicam-vid subprocess the task is being cancelled")
                    try:
                        self.rpicam_process.kill()
                    except (ProcessLookupError, AttributeError):
                        pass
                    if self.rpicam_process is None:
                        return
                    return await self.rpicam_process.wait()

    def _framerate_arg_index(self):
        """Index of the '--framerate=...' entry in self.stream_command, or None if this
        particular stream_command doesn't expose one."""
        for i, arg in enumerate(self.stream_command):
            if arg.startswith('--framerate'):
                return i
        return None

    def build_stream_command(self):
        """self.stream_command with --framerate rewritten from self.stream_framerate_conf_key,
        if this server has opted into a live-configurable framerate."""
        idx = self._framerate_arg_index()
        if idx is None or self.stream_framerate_conf_key is None:
            return self.stream_command
        cmd = list(self.stream_command)
        cmd[idx] = f"--framerate={self.conf[self.stream_framerate_conf_key]}"
        return cmd

    async def run_rpicam_vid(self):
        """
        Start the rpicam-vid stream process
        rpicam-vid listens for a single connection on 8888 and streams video to it, then terminates when the client disconnects.
        It prints a few setup lines and we need to record the time of one of them and inform the client of it.
        the client uses that to compute wall times from PTS times.
        it prints one more line after that then stops printing stuff until a few lines when the client disconnects.
        """
        command = self.build_stream_command()
        if self.stream_framerate_conf_key is not None:
            self.running_framerate = self.conf[self.stream_framerate_conf_key]
        self.rpicam_process = await asyncio.create_subprocess_exec(
            command[0], *command[1:], stdout=PIPE, stderr=STDOUT)
        # read all the lines of output
        while True:
            # during normal streaming, it is normal for this to block a long time because rpicam-vid isn't writing lines
            line = await self.rpicam_process.stdout.readline()
            if not line: # EOF.
                print('rpicam-vid exited')
                break
            line = line.decode()
            # remove color codes
            line = ansi_escape.sub('', line)
            print(line[:-1])

            # Look for the line indicating the stream is ready
            match = ready_line_re.match(line)
            if match:
                ready_wall_time = time.time()
                await asyncio.sleep(1.5) # it's not ready quite yet
                logging.info('rpicam-vid appears to be ready')
                # tell the websocket client to connect to the video stream. it will do so in another thread.
                self.update['video_ready'] = (8888, ready_wall_time + dts_zero_offset)
            else:
                # catch a few different kinds of errors that mean rpi-cam will have to be restarted
                # some of these can only happen after we have asked the client to try connecting to video.
                # and they don't result in rpicam-vid terminating on it's own.
                # ERROR: *** failed to allocate buffers
                # ERROR: *** failed to acquire camera
                # ERROR: *** failed to bind listen socket
                if line.startswith("ERROR: ***"):
                    logging.info('Killing rpicam-vid subprocess')
                    self.rpicam_process.kill()
                    break    
            # nothing wrong keep waiting for output lines

        # wait for the subprocess to exit, whether because we killed it, or it stopped normally
        return await self.rpicam_process.wait()

    async def read_temperature(self):
        while True:
            try:
                with open('/sys/class/thermal/thermal_zone0/temp') as f:
                    self.update['temp'] = int(f.read()) / 1000.0
            except OSError:
                pass
            await asyncio.sleep(1)

    async def process_imu(self, ws):
        pass

    async def log_subprocess_output(self, stream, logger_func):
        """Logs each line as it streams in (as before), and also returns all of them, so a
        caller that wants the full output after the fact (e.g. to report a failure back to
        the host) doesn't have to re-read the local log file."""
        lines = []
        async for line in stream:
            text = line.decode('utf-8').rstrip()
            logger_func(text)
            lines.append(text)
        return lines

    def restart_stream_if_framerate_changed(self):
        """Kill the running rpicam-vid process so stream_video's loop relaunches it with the
        newly configured framerate. A no-op if this server hasn't opted into a live framerate
        var, no stream is running, or the framerate didn't actually change."""
        if self.stream_framerate_conf_key is None or self._framerate_arg_index() is None:
            return
        if self.rpicam_process is None or self.conf[self.stream_framerate_conf_key] == self.running_framerate:
            return
        try:
            self.rpicam_process.kill()
        except (ProcessLookupError, AttributeError):
            pass

    async def stop_camera_stream(self):
        """Stop the camera stream and kill rpicam-vid. Cancelling stream_video makes
        it kill the subprocess and return without restarting it (it swallows the
        cancellation), so the task completes normally and its TaskGroup siblings —
        notably stream_measurements, which still needs to flush update progress —
        keep running."""
        if self.stream_video_task is not None and not self.stream_video_task.done():
            self.stream_video_task.cancel()
            try:
                await self.stream_video_task
            except asyncio.CancelledError:
                pass
        if self.rpicam_process is not None:
            try:
                self.rpicam_process.kill()
            except (ProcessLookupError, AttributeError):
                pass

    # TODO(remove ~2026-09): drop this and its call in run_update() once all fielded Pis
    # have picked up the fix (a month or two after this ships).
    CAN_SETUP_SERVICE_PATH = '/etc/systemd/system/can-setup.service'

    async def _fix_can_setup_service_wantedby(self):
        """One-off migration for Pis imaged before commit eca5db9 ("fix 90s boot delay on
        gripper"). That commit changed stringman-pilot-rpi-image/can-setup.service's
        WantedBy= from multi-user.target to sys-subsystem-net-devices-can0.device, so the
        service is only queued when a can0 device actually appears (pulled in by udev) rather
        than on every boot. Grippers have no MCP2515/can0 device, so with the old WantedBy=
        they queued this service every boot and its BindsTo/After on can0.device forced a
        full 90s DefaultDeviceTimeoutSec wait, which also delayed cranebot.service.

        Nothing in the normal update process rewrites files outside the Python package (this
        one lives in the rpi-image repo layout, not the pip package), so a Pi imaged before
        that commit will keep the old unit file and the boot delay forever unless patched by
        hand. This does that patch in-place, on whatever update cycle happens to run after
        this method exists. Anchors have a real can0 device and are unaffected either way, so
        this only actually changes anything on grippers, but it's harmless to run everywhere.
        """
        try:
            with open(self.CAN_SETUP_SERVICE_PATH) as f:
                content = f.read()
        except FileNotFoundError:
            return  # not a pilot anchor/gripper image, or already removed
        except OSError as e:
            logging.warning(f'Could not read {self.CAN_SETUP_SERVICE_PATH} to check for the can-setup.service fix: {e}')
            return

        if 'WantedBy=sys-subsystem-net-devices-can0.device' in content:
            return  # fresh image, or already patched by a previous update cycle

        fixed = content.replace('WantedBy=multi-user.target', 'WantedBy=sys-subsystem-net-devices-can0.device')
        if fixed == content:
            logging.warning(f'{self.CAN_SETUP_SERVICE_PATH} has an unexpected WantedBy= line; leaving it alone.')
            return

        logging.info('Patching can-setup.service WantedBy= to fix the 90s gripper boot delay (see commit eca5db9)')
        tmp_path = '/tmp/can-setup.service'
        try:
            with open(tmp_path, 'w') as f:
                f.write(fixed)
            for command in (
                ['sudo', 'install', '-m', '644', tmp_path, self.CAN_SETUP_SERVICE_PATH],
                # disable removes any symlink to this unit regardless of what WantedBy= used
                # to say; enable (run after the file is patched) creates the correct new one.
                ['sudo', 'systemctl', 'disable', 'can-setup.service'],
                ['sudo', 'systemctl', 'enable', 'can-setup.service'],
                ['sudo', 'systemctl', 'daemon-reload'],
            ):
                proc = await asyncio.create_subprocess_exec(*command, stdout=PIPE, stderr=STDOUT)
                output = (await proc.stdout.read()).decode('utf-8', errors='ignore')
                if await proc.wait() != 0:
                    logging.error(f"'{' '.join(command)}' failed: {output}")
                    return
            logging.info('can-setup.service patched successfully.')
        except OSError as e:
            logging.warning(f'Failed to patch {self.CAN_SETUP_SERVICE_PATH}: {e}')
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def run_update(self):
        logging.info('Performing Update')
        # Best-effort and unrelated to the pip upgrade below; must not block or fail it.
        try:
            await self._fix_can_setup_service_wantedby()
        except Exception:
            logging.exception('can-setup.service migration check failed; continuing with update anyway.')
        # Stop the camera first so rpicam-vid isn't holding the camera or burning CPU
        # while pip upgrades and we restart onto the new version.
        await self.stop_camera_stream()
        self.update['firmware_update_complete'] = {'pending': None}
        pip_subprocess = await asyncio.create_subprocess_exec(
            '/opt/robot/env/bin/pip', 'install', '--upgrade', 'nf_robot[pi]', '-q',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        stdout_task = asyncio.create_task(self.log_subprocess_output(pip_subprocess.stdout, logging.info))
        stderr_task = asyncio.create_task(self.log_subprocess_output(pip_subprocess.stderr, logging.error))
        returncode = await pip_subprocess.wait()
        stdout_lines = await stdout_task
        stderr_lines = await stderr_task
        if returncode == 0:
            self.update['firmware_update_complete'] = {'returncode': returncode}
            logging.info('Self update complete. Restarting.')
            # give stream_measurements a chance to flush firmware_update_complete before we exit.
            await asyncio.sleep(0.2)
            self.shutdown()
            # Exit the whole process so systemd restarts us on the new one.
            os._exit(0)
        else:
            # send pip output back so the host can put it
            error_output = '\n'.join(stdout_lines + stderr_lines)
            self.update['firmware_update_complete'] = {'returncode': returncode, 'error': error_output}
            logging.error(f'Self update failed with returncode {returncode}. Not restarting.')

    async def read_updates_from_client(self,websocket,tg):
        while True:
            message = await websocket.recv()
            update = json.loads(message)

            if 'set_config_vars' in update:
                self.conf.update(update['set_config_vars'])
                if self.stream_framerate_conf_key in update['set_config_vars']:
                    self.restart_stream_if_framerate_changed()
            if 'host_time' in update:
                logging.debug(f'measured latency = {time.time() - float(update["host_time"])}')
            if 'run_update' in update:
                self.extra_tasks.append(asyncio.create_task(self.run_update()))

            if self.spooler is not None:
                if 'length_set' in update:
                    self.spooler.setTargetLength(update['length_set'])
                if 'aim_speed' in update:
                    self.spooler.setAimSpeed(update['aim_speed'])
                if 'jog' in update:
                    self.spooler.jogRelativeLen(float(update['jog']))
                if 'reference_length' in update:
                    self.spooler.setReferenceLength(float(update['reference_length']))
                if 'set_zero_angle' in update:
                    self.spooler.sc.set_zero_angle(float(update['set_zero_angle']))

            if 'set_timezone' in update:
                tz = update['set_timezone']
                subprocess.run(['sudo', 'timedatectl', 'set-timezone', tz], check=True)
                logging.info(f'Timezone set to {tz}')

            # defer to specific server subclass
            result = await self.processOtherUpdates(update,tg)

    async def handler(self,websocket):
        logging.info('Websocket connected')

        # This features requires Python3.11
        # The first time any of the tasks belonging to the group fails with an exception other than asyncio.CancelledError,
        # the remaining tasks in the group are cancelled.
        # For normal client disconnects either the streaming or reading task will throw a ConnectionClosedOK
        # and the taskgroup context manager will cancel the other tasks, and re-raise it in an ExceptionGroup
        # except* matches errors within an ExceptionGroup
        # If the thrown exception is not one of the type caught here, the server stops.
        try:
            self.have_client = True
            # tell the client which version of nf_robot this component server is running
            await websocket.send(json.dumps({'nf_robot_v': importlib.metadata.version('nf_robot')}))
            async with asyncio.TaskGroup() as tg:
                read_updates = tg.create_task(self.read_updates_from_client(websocket, tg))
                stream = tg.create_task(self.stream_measurements(websocket))
                self.stream_video_task = tg.create_task(self.stream_video(websocket))
                stabil = tg.create_task(self.process_imu(websocket))
                temp = tg.create_task(self.read_temperature())
        except* (ConnectionClosedOK, ConnectionClosedError):
            logging.info("Client disconnected")
            self.have_client = False
        logging.info("All tasks in handler task group completed")
        # stop spool motors just in case some task left it running
        if self.spooler is not None:
            self.spooler.setAimSpeed(0)
        # the arp anchor drives two spools instead of a single spooler; stop both of them too
        for spool in getattr(self, 'spools', None) or []:
            spool.setAimSpeed(0)

    async def main(self, port=8765, name=None):
        logging.info('Starting cranebot server')
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), self.shutdown)

        # used in testing when running multiple servers on the same machine
        if name is not None:
            self.service_name = name

        self.run_server = True
        asyncio.create_task(self.register_mdns_service(f"123.{self.service_name}", "_http._tcp.local.", port))

        # thread for controlling stepper motor
        if self.spooler is not None:
            self.extra_tasks.append(asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop)))
            self.update['torque'] = True

        self.wait_reset_task = asyncio.create_task(self.watch_for_reset())
        self.extra_tasks.append(self.wait_reset_task)

        # Call a function which subclasses implement to start tasks at startup that should remain running even if clients disconnect.
        # tasks started this way should run only while self.run_server is true
        # should return a list of any tasks it started
        self.extra_tasks.extend(self.startOtherTasks())

        async with websockets.serve(self.handler, "0.0.0.0", port):
            logging.info("Websocket server started")
            while self.run_server:
                await asyncio.sleep(0.5)
            # if those tasks finish, exiting this context will cause the server's close() method to be called.
            logging.info("Closing websocket server")


        await self.zc.async_unregister_all_services()
        logging.info("Service unregistered")
        if len(self.extra_tasks) > 0:
            result = await asyncio.gather(*self.extra_tasks)


    def shutdown(self):
        # this might get called twice
        if self.run_server:
            logging.info('\nStopping detection listener task')
            self.run_server = False
            if self.wait_reset_task is not None:
                self.wait_reset_task.cancel()
            if self.spooler is not None:
                logging.info('Stopping Spool Motor')
                self.spooler.fastStop()

    async def register_mdns_service(self, name, service_type, port, properties={}):
        """Registers an mDNS service on the network."""

        ip = "127.0.0.1" # if ip remains unchanged, we are in a unit test
        if self.zc is None:
            self.zc = AsyncZeroconf(ip_version=zeroconf.IPVersion.V4Only)
            ip = get_local_ip()

        logging.info(f'zeroconf instance advertising on {ip}')
        await asyncio.sleep(1)
        info = zeroconf.ServiceInfo(
            service_type,
            name + "." + service_type,
            port=port,
            properties=properties,
            addresses=[ip],
            server=name,
        )

        await self.zc.async_register_service(info)
        logging.info(f"Registered service: {name} ({service_type}) on port {port}")

    async def watch_for_reset(self):
        """
        Forget all wifi networks if the switch is clicked five times in two seconds with no client.
        """
        try:
            await self.reset_wifi_event.wait()
            if not self.have_client:
                forget_all_wifi_networks()
                self.shutdown()
                os._exit(0) # systemctl will bring us back up. User must show new wifi share code.
        except asyncio.exceptions.CancelledError:
            pass
