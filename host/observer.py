import sys
import json
import requests
import threading
import time
import socket
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from cv_common import locate_board
import cv2
import numpy as np

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']

# global that will point to a DataStore passed to this process.
datastore = None

class PartHandler:
    def __init__(self):
        pass

    def handle_image(self, headers, buf):
        """
        handle a single image from the stream
        """
        # Decode image
        timestamp = float(headers['X-Timestamp'])
        if buf[:2] != b'\xff\xd8': # start of image marker
            # discard broken frames. they have no header, and a premature end of image marker.
            # I have no idea what causes these but the browser can display them.
            # they must be some kind of intermediate frame, but I have no information of what the encoding is.
            # the bad news is that when the lights are on, almost all frames are of this type.
            print("broken frame")
            return

        frame = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            retval, rvec, tvec = locate_board(frame, 'origin')
            print(f"Found board: {retval}")
            print(f"Timestamp: {timestamp}")
            print(f"Rotation Vector: {rvec}")
            print(f"Translation Vector: {tvec}")
            sys.stdout.flush()

                # using the board id, figure out which object it is
                # rotate and translate to where that object's origin would be
                # store the time and that position in the appropriate measurement array in observer.
                
                # datastore.some_part.insert(np.concatenate([[timestamp], [part_position]]))

    def handle_json(self, headers, buf):
        """
        handle a single json blob from the stream
        """
        dec = buf.decode()
        accel = json.loads(dec)
        # datastore.imu_accel.insert(np.array([timestamp, accel['x'], accel['y'], accel['z']]))

def parse_mixed_replace_stream(url):
    """
    Parses a multipart/x-mixed-replace stream using the requests library.

    Args:
        url: The URL of the stream.
        part_cb: function accepting a dict of headers and a bytes
            called for every part that is received from the stream

    Prints the content type of each part in the stream.
    """
    ph = PartHandler()
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        print(response.headers)
        content_type = response.headers.get('Content-Type', '')
        if "multipart/x-mixed-replace" not in content_type:
            print("Error: Not a multipart/x-mixed-replace stream.")
            sys.stdout.flush()
            return
        boundary = content_type[content_type.find('boundary=')+9:]
        boundary_with_newlines = f"\r\n--{boundary}\r\n"
        print(repr(boundary_with_newlines))

        for part in response.iter_lines(chunk_size=2**20, decode_unicode=False, delimiter=boundary_with_newlines.encode()):
            headers = {}
            lines = part.split(b'\r\n')
            for line in lines:
                line = line.decode()
                s = line.split(': ')
                if len(s) == 2:
                    headers[s[0]] = s[1]
                else:
                    break
            if len(headers) == 0:
                continue
            # the data begins after the first double newline
            data_start = part.find(b'\r\n\r\n')+4
            if headers['Content-Length'] == '0' or data_start == 3:
                print('skipping part with zero data size')
                continue

            if headers['Content-Type'] == 'image/jpeg':
                ph.handle_image(headers, part[data_start:])
            elif headers['Content-Type'] == 'application/json':
                ph.handle_json(headers, part[data_start:])
            else:
                print(f"Got an unexpected content type {headers['Content-Type']}")
                sys.stdout.flush()
        print("consumed stream")

cranebot_service_name = 'cranebot-service'
# listener for updating the list of available robot component servers
class CranebotListener(ServiceListener):
    def __init__(self):
        super().__init__()
        self.available_bots = {}
        self.bot_threads = {}

    def start_stream(self, info):
        address = socket.inet_ntoa(info.addresses[0])
        hostport = f"{address}:{info.port}"
        self.available_bots[info.server] = hostport
        # self.bot_threads[info.server].terminate()
        self.bot_threads[info.server] = threading.Thread(
            target=parse_mixed_replace_stream, args=(f"http://{hostport}/stream",), daemon=True)
        self.bot_threads[info.server].start()

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} updated")
        sys.stdout.flush()
        info = zc.get_service_info(type_, name)
        if name.split(".")[1] == cranebot_service_name:
            if info.server is not None and info.server != '':
                self.start_stream(info)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} removed")
        sys.stdout.flush()
        info = zc.get_service_info(type_, name)
        # the thread probably already stopped when the pipe broke, but just in case
        self.bot_threads[info.server].terminate()
        del self.bot_threads[info.server]
        del self.available_bots[info.server]

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        print(f"Service {name} added, service info: {info}")
        sys.stdout.flush()
        if name.split(".")[1] == cranebot_service_name:
            if info.server is not None and info.server != '':
                self.start_stream(info)



def start_observation(shared_array, to_ui_q):
    # set the global 
    datastore = shared_array
    to_ui_q = to_ui_q

    # start discovery
    run_discovery_task = True
    def service_discovery_task():
        print('Started service discovery task')
        zeroconf = Zeroconf()
        print('initialized Zeroconf')
        listener = CranebotListener()
        browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
        sys.stdout.flush()
        while run_discovery_task:
            time.sleep(0.1)
        zeroconf.close()

    service_discovery_task()

    # discovery_thread = threading.Thread(target=service_discovery_task, daemon=True)
    # discovery_thread.start()

# start_observation(None)