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

stream_command = """
rpicam-vid -t 0
  --width=4608 --height=2592
  --listen -o tcp://0.0.0.0:8888
  --codec mjpeg
  --vflip --hflip
  --buffer-count=5
  --autofocus-mode manual
  --lens-position 0.1""".split()
frame_line_re = re.compile(r"#(\d+) \((\d+\.\d+)\s+fps\) exp (\d+\.\d+)\s+ag (\d+\.\d+)\s+dg (\d+\.\d+)")

class RobotComponentServer:
    def __init__(self):
        self.run_client = True
        self.frametimes = []

    async def stream_measurements(self, ws):
        """
        stream line length measurements to the provided websocket connection
        as long as it exists
        """
        while ws:
            try:
                update = {}

                # add line lengths
                meas = self.spooler.popMeasurements()
                if len(meas) > 0:
                    if len(meas) > 50:
                        meas = meas[:50]
                    update['line_record']= meas

                # add frame times if we have any
                if len(self.frametimes) > 0:
                    update['frames'] = self.frametimes
                self.frametimes = []

                # send on websocket
                if update != {}
                    await ws.send(json.dumps(update))

                # chill
                await asyncio.sleep(0.3)
            except (ConnectionClosedOK, ConnectionClosedError):
                print("stopped streaming measurements")
                break

    async def stream_mjpeg(self, line_timeout=5):
        """
        Start the rpicam-vid stream process with a timeout.
        If no line is printed for timeout seconds, kill the process.
        rpicam-vid listens for a single connection on 8888 and streams video to it, then terminates when the client disconnects.
        it prints a line for every frame. Every time it does that, record the time and frame number.
        spit that down the provided websocket connection, if there is one.
        rpicam-vid has no way of sending timestamps on its own as of Feb 2025
        """
        process = await asyncio.create_subprocess_exec(stream_command,
                stdout=PIPE, stderr=STDOUT)
        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), line_timeout)
                t = time.time()
                if not line: # EOF
                    break
                else:
                    match = frame_line_re.match(line)
                    if match:
                        self.frametimes.append({
                            'time': t,
                            'fnum': int(match.group(1)),
                            # 'fps': match.group(2),
                            # 'exposure': match.group(3),
                            # 'analog_gain': match.group(4),
                            # 'digital_gain': match.group(5),
                        })
                    else:
                        print(line)
                    continue # nothing wrong keep going
            except asyncio.TimeoutError:
                print(f'rpicam-vid wrote no lines for {line_timeout} seconds')
            except asyncio.CancelledError:
                break

            # unless continue is hit above because we got a line of output quick enough, we kill the process.
            process.kill()
            break
        return await process.wait() # Wait for the child process to exit normally, such as when the client disconnects

    async def handler(self,websocket):
        print('Websocket connected')
        self.control_queue.put(f'MODE:True:False')
        stream = asyncio.create_task(self.stream_measurements(websocket))
        mjpeg = asyncio.create_task(self.stream_mjpeg())
        while True:
            try:
                message = await websocket.recv()
                update = json.loads(message)
                print(f"Received: {update}")

                if 'length_plan' in update:
                    self.spooler.setPlan(update['length_plan'])
                if 'reference_length' in update:
                    self.spooler.setReferenceLength(float(update['reference_length']))
                # command to kill or restart the camera task
                # sleep
                # slow stop
                # emergency stop

                response = {"status": "OK"}
                await websocket.send(json.dumps(response)) #Encode JSON

            except ConnectionClosedOK:
                print("Client disconnected")
                break
            except ConnectionClosedError as e:
                print(f"Client disconnected with {e}")
                break
        stream.cancel()


    async def main(self, port=8765):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), self.shutdown)

        self.run_client = True
        asyncio.create_task(self.register_mdns_service(f"123.{self.service_name}", "_http._tcp.local.", port))

        # thread for controlling stepper motor
        spool_task = asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop))

        async with websockets.serve(self.handler, "0.0.0.0", port):
            print("Websocket server started")
            # cause the server to serve only as long as these other tasks are running
            await asyncio.gather(listen_detector_task, spool_task)
            # if those tasks finish, exiting this context will cause the server's close() method to be called.
            print("Closing websocket server")


        await self.zc.async_unregister_all_services()
        print("Service unregistered")


    def shutdown(self):
        # this might get called twice
        if self.run_client:
            print('\nStopping detection listener task')
            self.run_client = False
            print('Stopping Spool Motor')
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
            print(f"Error getting IP address: {e}")
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
        print(f"Registered service: {name} ({service_type}) on port {port}")

class RaspiAnchorServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.spooler = SpoolController(MKSSERVO42C(), spool_diameter_mm=24)
        unique = ''.join(get_mac_address().split(':'))
        self.service_name = 'cranebot-anchor-service.' + unique


if __name__ == "__main__":
    ras = RaspiAnchorServer()
    asyncio.run(ras.main())
