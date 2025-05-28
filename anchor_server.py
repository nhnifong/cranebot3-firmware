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
from spools import SpoolController
from motor_control import MKSSERVO42C
import argparse
import logging

# video framerate and latency are limited by the memory on the raspberry pi zero.
# with more memory, we could increase buffer-count to 10 or 20 and get great performance.
# 6 is is the highest I have ever seen it work at full resolution, but not very reliably.
# 5 works about 80% of the time and 4 works about 95%
# higher framerate is possible with lower resolution, but it is not useful because at fullres
# I can still barely detect a 9cm aruco marker at 5 meters.
# There is an --roi arg to crop the image but it doesnt lessen the amount of memory rpicam-vid attempts to allocate.
stream_command = """
/usr/bin/rpicam-vid -t 0
  --width=4608 --height=2592
  --listen -o tcp://0.0.0.0:8888
  --codec mjpeg
  --vflip --hflip
  --buffer-count=4
  --autofocus-mode continuous""".split()
frame_line_re = re.compile(r"#(\d+) \((\d+\.\d+)\s+fps\) exp (\d+\.\d+)\s+ag (\d+\.\d+)\s+dg (\d+\.\d+)")
ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])') # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python
rpicam_ready_re = re.compile(r"\[(.*?)\]\s+\[(.*?)\]\s+INFO\s+RPI\s+pipeline_base\.cpp:\d+\s+Using\s+configuration\s+file\s+'(.*?)'")
busy_re = re.compile(r"\[\d+:\d+:\d+\.\d+\] \[\d+\] ERROR V4L2 v4l2_device\.cpp:\d+ '[\w-]+': Unable to set controls: Device or resource busy")


# values that can be overridden by the controller
default_conf = {
    # delay in seconds between updates sent on websocket during normal operation
    'RUNNING_WS_DELAY': 0.1,
    # delay in seconds between updates sent on websocket during calibration
    'CALIBRATING_WS_DELAY': 0.05,
}

class RobotComponentServer:
    def __init__(self):
        self.conf = default_conf.copy()
        self.run_server = True
        # a dict of update to be flushed periodically to the websocket
        self.update = {}
        self.frames = []
        self.ws_delay = self.conf['RUNNING_WS_DELAY']
        self.stream_command = stream_command
        self.rpicam_process = None
        self.line_timeout = 60

    async def stream_measurements(self, ws):
        """
        stream line length measurements to the provided websocket connection
        as long as it exists
        """
        logging.info('start streaming measurements')
        while True:
            # add line lengths
            meas = self.spooler.popMeasurements()
            if len(meas) > 0:
                if len(meas) > 50:
                    meas = meas[:50]
                self.update['line_record'] = meas

            self.readOtherSensors()

            if len(self.frames) > 0:
                self.update['frames'] = self.frames
                self.frames = []

            # send on websocket
            if self.update != {}:
                await ws.send(json.dumps(self.update))
            self.update = {}

            # chill
            await asyncio.sleep(self.ws_delay)

    async def stream_mjpeg(self, websocket):
        # keep rpicam-vid running until this task is cancelled by the client disconnecting
        while True:

            # if connected, this returns the connection latency.
            # of not connected, trying to get the result of the future throws a connection closed exception
            if not await websocket.ping():
                return

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
                self.rpicam_process.kill()
                return await self.rpicam_process.wait()

    async def run_rpicam_vid(self):
        """
        Start the rpicam-vid stream process with a timeout.
        If no line is printed for timeout seconds, kill the process.
        rpicam-vid listens for a single connection on 8888 and streams video to it, then terminates when the client disconnects.
        it prints a line for every frame. Every time it does that, record the time and frame number.
        spit that down the provided websocket connection, if there is one.
        rpicam-vid has no way of sending timestamps on its own as of Feb 2025
        """

        start_time = time.time()
        self.rpicam_process = await asyncio.create_subprocess_exec(self.stream_command[0], *self.stream_command[1:], stdout=PIPE, stderr=STDOUT)
        # read all the lines of output
        while True:
            try:
                line = await asyncio.wait_for(self.rpicam_process.stdout.readline(), self.line_timeout)
            except asyncio.TimeoutError:
                logging.warning(f'rpicam-vid wrote no lines for {self.line_timeout} seconds')
                self.rpicam_process.kill()
                break
            t = time.time()
            if not line: # EOF.
                break
            line = line.decode()

            # these are the types of lines we expect under normal operation after a client has started streaming.
            match = frame_line_re.match(line)
            if match:
                self.frames.append({
                    'time': t,
                    'fnum': int(match.group(1)),
                    # 'fps': match.group(2),
                    # 'exposure': match.group(3),
                    # 'analog_gain': match.group(4),
                    # 'digital_gain': match.group(5),
                })
            else:
                # log all other lines
                logging.info(line[:-1])
                # remove color codes
                line = ansi_escape.sub('', line)

                # check for "Unable to set controls: Device or resource busy
                if busy_re.match(line):
                    logging.info('Killing rpicam-vid subprocess because it cannot communicate with the device.')
                    self.rpicam_process.kill()
                    break

                # check if line is indicative that we started and are waiting for a connection.
                match = rpicam_ready_re.match(line)
                if match:
                    logging.info('rpicam-vid appears to be ready')
                    # tell the websocket client to connect to the video stream. it will do so in another thread.
                    self.update['video_ready'] = True

                # catch a few different kinds of errors that mean rpi-cam will have to be restarted
                # some of these can only happen after we have asked the client to try connecting to video.
                # ERROR: *** failed to allocate buffers
                # ERROR: *** failed to acquire camera
                # ERROR: *** failed to bind listen socket
                if line.startswith("ERROR: ***"):
                    logging.info('Killing rpicam-vid subprocess')
                    self.rpicam_process.kill()
                    break    
            # nothing wrong keep going

        # wait for the subprocess to exit, whether because we killed it, or it stopped normally
        return await self.rpicam_process.wait()

    async def read_updates_from_client(self,websocket,tg):
        while True:
            message = await websocket.recv()
            update = json.loads(message)

            if 'length_plan' in update:
                self.spooler.setPlan(update['length_plan'])
            if 'aim_speed' in update:
                self.spooler.setAimSpeed(update['aim_speed'])
            if 'host_time' in update:
                logging.debug(f'measured latency = {time.time() - float(update["host_time"])}')
            if 'jog' in update:
                self.spooler.jogRelativeLen(float(update['jog']))
            if 'reference_length' in update:
                self.spooler.setReferenceLength(float(update['reference_length']))
            if 'set_config_vars' in update:
                self.conf.update(update['set_config_vars'])
                pass

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
                mjpeg = tg.create_task(self.stream_mjpeg(websocket))
        except* (ConnectionClosedOK, ConnectionClosedError):
            logging.info("Client disconnected")
        logging.info("All tasks in handler task group completed")
        # stop motor just in case some task left it running
        self.spooler.setAimSpeed(0)


    async def main(self, port=8765):
        logging.info('Starting cranebot server')
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), self.shutdown)

        self.run_server = True
        asyncio.create_task(self.register_mdns_service(f"123.{self.service_name}", "_http._tcp.local.", port))

        # thread for controlling stepper motor
        self.spooler.setReferenceLength(0.5)
        spool_task = asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop))

        self.startOtherTasks()

        async with websockets.serve(self.handler, "0.0.0.0", port):
            logging.info("Websocket server started")
            # cause the server to serve only as long as these other tasks are running
            # note that you must always get the result from something run with asyncio.to_thread or it will silently pass exceptions.
            result = await spool_task
            # if those tasks finish, exiting this context will cause the server's close() method to be called.
            logging.info("Closing websocket server")


        await self.zc.async_unregister_all_services()
        logging.info("Service unregistered")
        result = await spool_task


    def shutdown(self):
        # this might get called twice
        if self.run_server:
            logging.info('\nStopping detection listener task')
            self.run_server = False
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

        self.zc = AsyncZeroconf(ip_version=zeroconf.IPVersion.All)
        info = zeroconf.ServiceInfo(
            service_type,
            name + "." + service_type,
            port=port,
            properties=properties,
            addresses=[self.get_wifi_ip()],
            server=name,
        )

        await self.zc.async_register_service(info)
        logging.info(f"Registered service: {name} ({service_type}) on port {port}")

default_anchor_conf = {
    # 0 or 1. provides a method of configuring that the switch is wired up backwards.
    # should be set to the value the pin will read when the line is pulled tight and the switch closes.
    'switch_tight_val': 0,

    # speed to reel in when the 'tighten' command is received. Meters of line per second
    'tightening_speed': -0.12
}

try:
    import RPi.GPIO as GPIO
    gpio_ready = True
except RuntimeError:
    # we can only run that on an actual pi, not in a unit test.
    gpio_ready = False

SWITCH_PIN = 18

class RaspiAnchorServer(RobotComponentServer):
    def __init__(self, power_anchor=False, flat=False, mock_motor=None):
        super().__init__()
        self.conf.update(default_anchor_conf)
        ratio = 20/51 # 20 drive gear teeth, 51 spool teeth.
        if mock_motor is not None:
            motor = mock_motor
        else:
            motor = MKSSERVO42C()
        if power_anchor:
            # A power anchor spool has a thicker line
            self.spooler = SpoolController(motor, empty_diameter=25, full_diameter=43.7, full_length=7.5, conf=self.conf, gear_ratio=ratio, tight_check_fn=self.tight_check)
        else:
            # other spools are wound with 50lb test braided fishing line with a thickness of 0.35mm
            self.spooler = SpoolController(motor, empty_diameter=25, full_diameter=27, full_length=7.5, conf=self.conf, gear_ratio=ratio, tight_check_fn=self.tight_check)
        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-anchor-service.' + unique

        if gpio_ready:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(SWITCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    def tight_check(self):
        """Return whether the line is tight according to the lever switch"""
        if not gpio_ready:
            return True
        return GPIO.input(SWITCH_PIN) == self.conf['switch_tight_val']

    async def processOtherUpdates(self, updates, tg):
        if 'tighten' in updates:
            tg.create_task(self.tighten())

    def readOtherSensors(self):
        pass

    def startOtherTasks(self):
        pass

    async def tighten(self):
        """
        Pull in the line slowly until the lever switch clicks.
        The client may check if this is completed by the tight value in the line_record updates
        """
        while not self.tight_check():
            self.spooler.setAimSpeed(self.conf['tightening_speed'])
            await asyncio.sleep(0.05)
        self.spooler.setAimSpeed(0)
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--power", action="store_true",
                        help="Configures this anchor as the one which has the power line")
    parser.add_argument("--flat", action="store_true",
                        help="Configures this anchor as one of the old direct drive type")
    args = parser.parse_args()

    ras = RaspiAnchorServer(args.power)
    asyncio.run(ras.main())
