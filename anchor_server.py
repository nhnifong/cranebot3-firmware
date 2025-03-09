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

class RobotComponentServer:
    def __init__(self):
        self.run_server = True
        self.ws_client_connected = False
        # a dict of update to be flushed periodically to the websocket
        self.update = {}

    async def stream_measurements(self, ws):
        """
        stream line length measurements to the provided websocket connection
        as long as it exists
        """
        while ws:
            try:
                update = self.update
                self.update = {'frames': []}

                # add line lengths
                meas = self.spooler.popMeasurements()
                if len(meas) > 0:
                    if len(meas) > 50:
                        meas = meas[:50]
                    update['line_record']= meas

                self.readOtherSensors()

                # send on websocket
                if update != {}:
                    await ws.send(json.dumps(update))

                # chill
                await asyncio.sleep(0.3)
            except (ConnectionClosedOK, ConnectionClosedError):
                logging.info("stopped streaming measurements")
                break

    async def stream_mjpeg(self):
        while self.ws_client_connected:
            await self.run_rpicam_vid()
            await asyncio.sleep(0.5)

    async def run_rpicam_vid(self, line_timeout=60):
        """
        Start the rpicam-vid stream process with a timeout.
        If no line is printed for timeout seconds, kill the process.
        rpicam-vid listens for a single connection on 8888 and streams video to it, then terminates when the client disconnects.
        it prints a line for every frame. Every time it does that, record the time and frame number.
        spit that down the provided websocket connection, if there is one.
        rpicam-vid has no way of sending timestamps on its own as of Feb 2025
        """
        process = await asyncio.create_subprocess_exec(stream_command[0], *stream_command[1:], stdout=PIPE, stderr=STDOUT)
        # read all the lines of output
        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), line_timeout)
                t = time.time()
                if not line: # EOF
                    break
                else:
                    line = line.decode()
                    match = frame_line_re.match(line)
                    if match:
                        self.update['frames'].append({
                            'time': t,
                            'fnum': int(match.group(1)),
                            # 'fps': match.group(2),
                            # 'exposure': match.group(3),
                            # 'analog_gain': match.group(4),
                            # 'digital_gain': match.group(5),
                        })
                    else:
                        logging.info(line)
                        if line.strip() == "ERROR: *** failed to allocate capture buffers for stream ***":
                            logging.warning(f'rpicam-vid failed to allocate buffers. restarting...')
                            break
                    continue # nothing wrong keep going
            except asyncio.TimeoutError:
                logging.warning(f'rpicam-vid wrote no lines for {line_timeout} seconds')
                process.kill()
                break
            except asyncio.CancelledError:
                logging.info("Killing rpicam-vid because the task has been cancelled")
                process.kill()
                break

            # unless continue is hit above because we got a line of output quick enough, we kill the process.
            process.kill()
            break
        # Wait for the child process to exit. whether normally as when the client disconnects,
        # or because we just killed it.
        return await process.wait()

    async def handler(self,websocket):
        logging.info('Websocket connected')
        self.ws_client_connected = True
        stream = asyncio.create_task(self.stream_measurements(websocket))
        mjpeg = asyncio.create_task(self.stream_mjpeg())
        while True:
            try:
                message = await websocket.recv()
                update = json.loads(message)

                if 'length_plan' in update:
                    self.spooler.setPlan(update['length_plan'])
                if 'jog' in update:
                    self.spooler.jogRelativeLen(float(update['jog']))
                if 'reference_length' in update:
                    self.spooler.setReferenceLength(float(update['reference_length']))

                # defer to specific server subclass
                self.processOtherUpdates(update)

            except ConnectionClosedOK:
                logging.info("Client disconnected")
                break
            except ConnectionClosedError as e:
                logging.info(f"Client disconnected with {e}")
                break
        self.ws_client_connected = False
        stream.cancel()
        mjpeg.cancel()


    async def main(self, port=8765):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), self.shutdown)

        self.run_server = True
        asyncio.create_task(self.register_mdns_service(f"123.{self.service_name}", "_http._tcp.local.", port))

        # thread for controlling stepper motor
        self.spooler.setReferenceLength(0.5)
        spool_task = asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop))

        async with websockets.serve(self.handler, "0.0.0.0", port):
            logging.info("Websocket server started")
            # cause the server to serve only as long as these other tasks are running
            await spool_task
            # if those tasks finish, exiting this context will cause the server's close() method to be called.
            logging.info("Closing websocket server")


        await self.zc.async_unregister_all_services()
        logging.info("Service unregistered")


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
        except Exception:
            logging.error(f"Error getting IP address: {e}")
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

class RaspiAnchorServer(RobotComponentServer):
    def __init__(self, power_anchor=False):
        super().__init__()
        if power_anchor:
            # the large spool is wound with a 2 core pvc sheathed wire
            self.spooler = SpoolController(MKSSERVO42C(), empty_diameter=28, full_diameter=64, full_length=9)
        else:
            # the small spools are wound with 50lb test braided fishing line
            self.spooler = SpoolController(MKSSERVO42C(), empty_diameter=24, full_diameter=25, full_length=9)
        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-anchor-service.' + unique

    def processOtherUpdates(self, updates):
        pass

    def readOtherSensors(self):
        pass


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
