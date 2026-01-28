import asyncio
from asyncio.subprocess import PIPE, STDOUT
import os
import signal
import websockets
from websockets.exceptions import (
    ConnectionClosedOK,
    ConnectionClosedError,
)
import json
import threading
import zeroconf
from zeroconf.asyncio import (
    AsyncZeroconf,
)
import uuid
import socket
import time
import re
from getmac import get_mac_address
import argparse
import logging

import nf_robot.common.definitions as model_constants
from nf_robot.robot.spools import SpoolController
from nf_robot.robot.mks42c_motor import MKSSERVO42C

# using libav makes it possible to send a containerized stream with pts
# hardware h264 encoding is still used as long as resolution is below 1080
# this requires rpicam-apps (not present in lite OS image)
stream_command = [
    "/usr/bin/rpicam-vid", "-t", "0", "-n",
    "--width=1920", "--height=1080",
    "-o", "tcp://0.0.0.0:8888?listen=1",
    "--codec", "libav",
    "--libav-format", "mpegts",
    "--vflip", "--hflip",
    "--autofocus-mode", "manual",
    "--lens-position", "0.1",
    "--low-latency",
    "--bitrate", "1200kbps"
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
    'RUNNING_WS_DELAY': 0.1,
    # delay in seconds between updates sent on websocket during calibration
    'CALIBRATING_WS_DELAY': 0.05
}

class RobotComponentServer:
    def __init__(self):
        self.conf = default_conf.copy()
        self.run_server = True
        # a dict of update to be flushed periodically to the websocket
        self.update = {}
        self.ws_delay = self.conf['RUNNING_WS_DELAY']
        self.rpicam_process = None
        self.zc = None # zerconf instance.
        self.mock_camera_port = None
        self.extra_tasks = []

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
                    except ProcessLookupError:
                        pass
                    return await self.rpicam_process.wait()

    async def run_rpicam_vid(self):
        """
        Start the rpicam-vid stream process
        rpicam-vid listens for a single connection on 8888 and streams video to it, then terminates when the client disconnects.
        It prints a few setup lines and we need to record the time of one of them and inform the client of it.
        the client uses that to compute wall times from PTS times.
        it prints one more line after that then stops printing stuff until a few lines when the client disconnects.
        """
        self.rpicam_process = await asyncio.create_subprocess_exec(stream_command[0], *stream_command[1:], stdout=PIPE, stderr=STDOUT)
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
                logging.info('rpicam-vid appears to be ready')
                await asyncio.sleep(1) # it's not ready quite yet
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

    async def read_updates_from_client(self,websocket,tg):
        while True:
            message = await websocket.recv()
            update = json.loads(message)

            if 'set_config_vars' in update:
                self.conf.update(update['set_config_vars'])
            if 'host_time' in update:
                logging.debug(f'measured latency = {time.time() - float(update["host_time"])}')

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
            async with asyncio.TaskGroup() as tg:
                read_updates = tg.create_task(self.read_updates_from_client(websocket, tg))
                stream = tg.create_task(self.stream_measurements(websocket))
                mjpeg = tg.create_task(self.stream_video(websocket))
        except* (ConnectionClosedOK, ConnectionClosedError):
            logging.info("Client disconnected")
        logging.info("All tasks in handler task group completed")
        # stop spool motors just in case some task left it running
        if self.spooler is not None:
            self.spooler.setAimSpeed(0)


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
            if self.spooler is not None:
                logging.info('Stopping Spool Motor')
                self.spooler.fastStop()

    def get_wifi_ip(self):
        """Gets the Raspberry Pi's IP address on the Wi-Fi interface."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception as e:
            logging.error(f"Error getting IP address: {e}")
            # todo just wait 10 seconds and try again
            return None

    async def register_mdns_service(self, name, service_type, port, properties={}):
        """Registers an mDNS service on the network."""

        ip = "127.0.0.1" # if ip remains unchanged, we are in a unit test
        if self.zc is None:
            self.zc = AsyncZeroconf(ip_version=zeroconf.IPVersion.V4Only)
            ip = self.get_wifi_ip()

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

default_anchor_conf = {
    # 0 or 1. provides a method of configuring that the switch is wired up backwards.
    # should be set to the value the pin will read when the line is pulled tight and the switch closes.
    'switch_tight_val': 0,

    # speed to reel in when the 'tighten' command is received. Meters of line per second
    'tightening_speed': -0.12,

    # if set, the switch is disabled and the line is always assumed to be tight
    'disable_switch': False,

    # Number of buffers to use when streaming mjpeg. Use as many as possible for high framerate without running out of ram
    'buffers': 3
}

try:
    import RPi.GPIO as GPIO
    gpio_ready = True
except RuntimeError:
    # we can only run that on an actual pi, not in a unit test.
    gpio_ready = False

SWITCH_PIN = 18

class RaspiAnchorServer(RobotComponentServer):
    def __init__(self, power_anchor=False, mock_motor=None):
        super().__init__()
        self.conf.update(default_anchor_conf)
        ratio = 20/51 # 20 drive gear teeth, 51 spool teeth.
        if mock_motor is not None:
            motor = mock_motor
        else:
            motor = MKSSERVO42C()
            motor.runConstantSpeed(0) # just in case
        if power_anchor:
            # A power anchor spool has a thicker line
            self.spooler = SpoolController(
                motor,
                empty_diameter=model_constants.empty_spool_diameter,
                full_diameter=model_constants.full_spool_diameter_power_line,
                full_length=model_constants.assumed_full_line_length,
                conf=self.conf, gear_ratio=ratio, tight_check_fn=self.tight_check)
        else:
            # other spools are wound with 50lb test braided fishing line with a thickness of 0.35mm
            self.spooler = SpoolController(
                motor,
                empty_diameter=model_constants.empty_spool_diameter,
                full_diameter=model_constants.full_spool_diameter_fishing_line,
                full_length=model_constants.assumed_full_line_length,
                conf=self.conf, gear_ratio=ratio, tight_check_fn=self.tight_check)
        unique = ''.join(get_mac_address().split(':'))

        if power_anchor:
            self.service_name = 'cranebot-anchor-power-service.' + unique
        else:
            self.service_name = 'cranebot-anchor-service.' + unique

        if gpio_ready:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        self.report_raw = False

    def tight_check(self):
        """Return whether the line is tight according to the lever switch"""
        if (not gpio_ready) or (self.conf['disable_switch']):
            return True
        return GPIO.input(SWITCH_PIN) == self.conf['switch_tight_val']

    async def processOtherUpdates(self, updates, tg):
        if 'tighten' in updates:
            tg.create_task(self.tighten())
        if 'report_raw' in updates:
            self.report_raw = bool(updates['report_raw'])

    def readOtherSensors(self):
        # when in calibration mode send a message containing the last raw encoder angle
        if self.report_raw:
            self.update['last_raw_encoder'] = self.spooler.last_angle

    def startOtherTasks(self):
        return []

    async def tighten(self):
        """
        Pulls in the line until tight. If the line slips within 3 seconds,
        it reduces the speed by 30% and retries, up to 5 times.
        """
        max_retries = 5
        monitoring_duration_s = 3
        check_interval_s = 0.05
        
        current_speed = self.conf['tightening_speed']

        for attempt in range(1, max_retries + 1):
            # Pull in the line until the switch clicks
            while not self.tight_check():
                self.spooler.setAimSpeed(current_speed)
                await asyncio.sleep(check_interval_s)
            self.spooler.setAimSpeed(0)

            # Monitor for re-loosening over the next 3 seconds
            loosened = False
            # Calculate when the monitoring window should end
            end_time = time.monotonic() + monitoring_duration_s
            while time.monotonic() < end_time:
                if not self.tight_check():
                    loosened = True
                    break  # Exit monitoring loop immediately on slip
                await asyncio.sleep(check_interval_s)

            # Check the outcome
            if not loosened:
                print(f"Tightening successful on attempt {attempt}.")
                return # Success!

            # If it slipped, reduce speed and the loop will try again
            print(f"Line re-loosened on attempt {attempt}. Reducing speed and reeling in.")
            current_speed *= 0.7

        # If the loop finishes, all retries have failed
        self.spooler.setAimSpeed(0)
        logging.error(f"Failed to tighten line after {max_retries} attempts.")
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--power", action="store_true",
                        help="Configures this anchor as the one which has the power line")
    args = parser.parse_args()

    ras = RaspiAnchorServer(args.power)
    asyncio.run(ras.main())
