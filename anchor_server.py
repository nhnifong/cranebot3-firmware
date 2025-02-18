import asyncio
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
from getmac import get_mac_address
import multiprocessing
from spools import SpoolController
from local_cam import local_aruco_detection
from motor_control import MKSSERVO42C

def dummyProcess(outq, control_queue):
    print("dummy process started")
    while True:
        if not control_queue.empty():
            if control_queue.get_nowait() == "STOP":
                break # exit loop, ending process
    print("dummy process ended")


class RobotComponentServer:
    def __init__(self):
        self.ws = None # active websocket connection if there is one
        self.run_client = True

    async def stream_measurements(self, ws):
        """
        stream line length measurements to the provided websocket connection
        as long as it exists
        """
        while ws:
            try:
                meas = self.spooler.popMeasurements()
                if len(meas) > 0:
                    if len(meas) > 50:
                        meas = meas[:50]
                    print(f"sending {len(meas)} line length data")
                    await ws.send(json.dumps({'line_record': meas}))
                await asyncio.sleep(0.5)
            except (ConnectionClosedOK, ConnectionClosedError):
                print("stopped streaming measurements")
                break

    async def handler(self,websocket):
        print('Websocket connected')
        self.ws = websocket
        stream = asyncio.create_task(self.stream_measurements(websocket))
        while True:
            try:
                message = await websocket.recv()
                update = json.loads(message)
                print(f"Received: {update}")

                if 'length_plan' in update:
                    self.spooler.setPlan(update['length_plan'])
                if 'reference_length' in update:
                    self.spooler.setReferenceLength(float(update['reference_length']))

                response = {"status": "OK"}
                await websocket.send(json.dumps(response)) #Encode JSON

            except ConnectionClosedOK:
                print("Client disconnected")
                break
            except ConnectionClosedError as e:
                print(f"Client disconnected with {e}")
                break
        stream.cancel()

    async def listen_detector(self, detection_queue):
        while self.run_client:
            # pull up to 20 things off the queue. This is to keep ws messages smaller
            dets = []
            for i in range(20):
                if detection_queue.empty():
                    break
                dets.append(detection_queue.get_nowait())
            if len(dets) > 0 and self.ws:
                print(f"sending {len(dets)} detections")
                try:
                    await self.ws.send(json.dumps({'detections': dets}))
                except (ConnectionClosedOK, ConnectionClosedError):
                    # no problem, client left. just throw these out
                    await asyncio.sleep(2.0)
                    continue
            else:
                await asyncio.sleep(0.1)


    async def main(self, port=8765):
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), self.shutdown)

        self.run_client = True
        asyncio.create_task(self.register_mdns_service(f"123.{self.service_type}", "_http._tcp.local.", port))

        # process for detecting fudicial markers
        print("starting video task")
        control_queue = multiprocessing.Queue()
        detection_queue = multiprocessing.Queue()
        control_queue.cancel_join_thread()
        detection_queue.cancel_join_thread()

        aruco_process = multiprocessing.Process(target=local_aruco_detection, args=(detection_queue, control_queue))
        aruco_process.daemon = True
        aruco_process.start()

        # thread for listening to aruco detector process
        listen_detector_task = asyncio.create_task(self.listen_detector(detection_queue))

        # thread for controlling stepper motor
        spool_task = asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop))

        async with websockets.serve(self.handler, "0.0.0.0", port):
            print("Websocket server started")
            # cause the server to serve only as long as these other tasks are running
            await asyncio.gather(listen_detector_task, spool_task)
            # if those tasks finish, exiting this context will cause the server's close() method to be called.
            print("Closing websocket server")

        # once that context has exited, stop and join the aruco process
        control_queue.put("STOP")
        aruco_process.join()

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
        unique = ''.join(get_mac_address().split(':'))
        info = zeroconf.ServiceInfo(
            service_type,
            name + "." + service_type,
            port=port,
            properties=properties,
            addresses=[self.get_wifi_ip()],
            server=self.name_prefix + unique,
        )

        await self.zc.async_register_service(info)
        print(f"Registered service: {name} ({service_type}) on port {port}")

class RaspiAnchorServer(RobotComponentServer):
    def __init__(self):
        super().__init__()
        self.spooler = SpoolController(MKSSERVO42C(), spool_diameter_mm=24)
        self.name_prefix = 'raspi-anchor-'
        self.service_type = 'cranebot-anchor-service'


if __name__ == "__main__":
    ras = RaspiAnchorServer()
    asyncio.run(ras.main())