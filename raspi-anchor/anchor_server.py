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
import subprocess
from spools import SpoolController

class RaspiAnchorServer:
    def __init__(self):
        self.spooler = SpoolController()

    async def stream_measurements(self, ws):
        """
        stream line length measurements to the provided websocket connection
        as long as it exists
        """
        while ws:
            measurements = self.spooler.popMeasurements()
            if len(measurements) > 0:
                await ws.send(json.dumps({'line_record': measurements}))
            await asyncio.sleep(0.5)

    async def handler(self,websocket):
        print('Websoocket connected')
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

    async def serve_video(self):
        while True:
            # keep restarting this forever.
            result = subprocess.run(['./start_stream.sh'], shell=True, capture_output=False, text=True)

    async def main(self, port):
        video_task = asyncio.create_task(asyncio.to_thread(self.serve_video))
        spool_task = asyncio.create_task(asyncio.to_thread(self.spooler.trackingLoop))
        async with websockets.serve(self.handler, "0.0.0.0", port): #Listen on all interfaces, port 8765
            await asyncio.gather[video_task, spool_task] # I guess that means run until they both crash lol
        video_task.cancel()

def get_wifi_ip():
    """Gets the Raspberry Pi's IP address on the Wi-Fi interface.

    Returns:
        The IP address as a string, or None if the Wi-Fi interface
        is not found or has no IP address.
    """
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
        # there is supposed to be some way to make zeroconf play nice with asyncio but I couldn't make it work
        mdns_thread = threading.Thread(target=register_mdns_service,
            args=("123.cranebot-anchor-service", "_http._tcp.local.", PORT), daemon=True)
        mdns_thread.start()
    ras = RaspiAnchorServer()
    asyncio.run(ras.main(PORT))
