import asyncio
import websockets
from websockets.exceptions import (
    ConnectionClosedOK,
    ConnectionClosedError,
)
import json
import threading
import zeroconf
import uuid
import socket
import argparse
import time
from getmac import get_mac_address
import multiprocessing
from spools import SpoolController
from cv_common import locate_markers
from picamera2 import Picamera2

def local_aruco_detection(outq, control_queue):
    """
    Open the camera and detect aruco markers
    put any detections on the provided queue
    """
    picam2 = Picamera2()
    # full res is 4608x2592
    # this is half res. seems it can still detect a 10cm aruco from about 2 meters at a rate of 30fps
    capture_config = picam2.create_preview_configuration(main={"size": (2304, 1296), "format": "RGB888"})
    picam2.configure(capture_config)
    picam2.start()
    while True:
        if not control_queue.empty():
            if control_queue.get_nowait() == "STOP":
                break # exit loop, ending process

        sec = time.time()
        im = picam2.capture_array()
        detections = locate_markers(im)
        if len(detections) > 0:
            #print(f'detected {list([d["n"] for d in detections])}')
            for det in detections:
                det['s'] = sec # add the time of capture to the detection
                outq.put(det)
        #await asyncio.sleep(0.5)

class RaspiAnchorServer:
    def __init__(self):
        self.spooler = SpoolController()
        self.ws = None # active websocket connection if there is one

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
                break

    async def handler(self,websocket):
        print('Websocket connected')
        self.ws = websocket
        asyncio.create_task(self.stream_measurements(websocket))
        while True:
            try:
                message = await websocket.recv()
                update = json.loads(message)
                print(f"Received: {update}")

                if 'length_plan' in update:
                    spooler.setPlan(update['length_plan'])
                if 'reference_length' in update:
                    spooler.setReferenceLength(float(update['reference_length']))

                response = {"status": "OK"}
                await websocket.send(json.dumps(response)) #Encode JSON

            except (ConnectionClosedOK, ConnectionClosedError):
                break

    async def listen_detector(self, detection_queue):
        while self.run_detection_listen_loop:
            # pull up to 20 things off the queue. This is to keep ws messages smaller
            dets = []
            for i in range(20):
                if detection_queue.empty():
                    break
                dets.append(detection_queue.get_nowait())
            if len(dets) > 0 and self.ws:
                print(f"sending {len(dets)} detections")
                await self.ws.send(json.dumps({'detections': dets}))
            else:
                await asyncio.sleep(0.1)


    async def main(self, port):
        # process for detecting fudicial markers
        print("starting video task")
        control_queue = multiprocessing.Queue()
        detection_queue = multiprocessing.Queue()
        aruco_process = multiprocessing.Process(target=local_aruco_detection, args=(detection_queue, control_queue))
        aruco_process.daemon = True
        aruco_process.start()
        #ttask = asyncio.create_task(local_aruco_detection(detection_queue, control_queue))

        # thread for listening to aruco detector process
        self.run_detection_listen_loop = True
        listen_detector_task = asyncio.create_task(self.listen_detector(detection_queue))

        # thread for controlling stepper motor
        spool_task = asyncio.to_thread(self.spooler.trackingLoop)

        try:
            # todo catch that exception that happens when the client on the other end crashes and doesnt send the close frame and ignore it.
            async with websockets.serve(self.handler, "0.0.0.0", port):
                # cause the server to serve forever, because these tasks don't finish
                await asyncio.gather(listen_detector_task, spool_task)

        except KeyboardInterrupt:
            self.spooler.fastStop()
            control_queue.put("STOP")
            aruco_process.join()
            self.run_detection_listen_loop = False
            await asyncio.gather(listen_detector_task, spool_task)

def get_wifi_ip():
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

def register_mdns_service(name, service_type, port, properties={}):
    """Registers an mDNS service on the network."""

    zc = zeroconf.Zeroconf()
    unique = ''.join(get_mac_address().split(':'))
    info = zeroconf.ServiceInfo(
        service_type,
        name + "." + service_type,
        port=port,
        properties=properties,
        addresses=[get_wifi_ip()],
        server=f'raspi-anchor-{unique}',
    )

    zc.register_service(info)
    print(f"Registered service: {name} ({service_type}) on port {port}")
    while True:
        time.sleep(1)
    zc.unregister_service(info)
    print("Service unregistered")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTTP test server for streaming images and acceleration data.")
    
    parser.add_argument("-p", "--port", type=int, default=8765, help="Port number to listen on.")
    parser.add_argument("-m", "--mdns", type=bool, default=True, help="Advertise the service with MDNS")

    args = parser.parse_args()
    PORT = args.port

    if args.mdns:
        # Start mdns advertisement in a separate thread
        mdns_thread = threading.Thread(target=register_mdns_service,
            args=("123.cranebot-anchor-service", "_http._tcp.local.", PORT), daemon=True)
        mdns_thread.start()
    ras = RaspiAnchorServer()
    asyncio.run(ras.main(PORT))
