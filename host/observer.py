import sys
import json
import requests
import threading
import time
import socket
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from cv_common import cranebot_boards, cranebot_detectors
import cv2
import cv2.aruco as aruco
import numpy as np

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']

# global that will point to a DataStore passed to this process.
datastore = None

# Intrinsic Matrix: 
camera_matrix = np.array(
[[1.55802968e+03, 0.00000000e+00, 8.58167917e+02],
 [0.00000000e+00, 1.56026885e+03, 6.28095370e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
 
# Distortion Coefficients: 
dist_coeffs = np.array(
[[ 3.40916628e-01, -2.38650897e+00, -8.85125582e-04, 3.34240054e-03, 4.69525036e+00]])

def handle_image(headers, buf):
    """
    handle a single image from the stream
    """
    # Decode image
    timestamp = float(headers['X-Timestamp-Sec']) + float(headers['X-Timestamp-Usec'])*0.000001
    frame = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), -1)
    # Detect ArUco markers
    charuco_corners, charuco_ids, marker_corners, marker_ids = cranebot_detectors["origin"].detectBoard(frame)
    if charuco_corners is not None and len(charuco_corners) > 0:
        #estimate charuco board pose
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, cranebot_boards["origin"], camera_matrix, dist_coeffs, None, None)
        print(f"Found board: {marker_ids}")
        print(f"Timestamp: {timestamp}")
        print(f"Rotation Vector: {rvec}")
        print(f"Translation Vector: {tvec}")
        sys.stdout.flush()

        # using the board id, figure out which object it is
        # rotate and translate to where that object's origin would be
        # store the time and that position in the appropriate measurement array in observer.
        
        # datastore.some_part.insert(np.concatenate([[timestamp], [part_position]]))

def handle_json(headers, buf):
    """
    handle a single json blob from the stream
    """
    timestamp = float(headers['X-Timestamp-Sec']) + float(headers['X-Timestamp-Usec'])*0.000001
    accel = json.loads(buf.decode())
    print(json)
    sys.stdout.flush()
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
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if "multipart/x-mixed-replace" not in content_type:
            print("Error: Not a multipart/x-mixed-replace stream.")
            sys.stdout.flush()
            return
        boundary = content_type[content_type.find('boundary=')+9:]
        boundary_with_newlines = f"\r\n{boundary}\r\n"

        for part in response.iter_lines(chunk_size=2**10, decode_unicode=False, delimiter=boundary_with_newlines.encode()):
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
            print(headers)
            sys.stdout.flush()
            if headers['Content-Type'] == 'image/jpeg':
                handle_image(headers, lines[-1])
            elif headers['Content-Type'] == 'application/json':
                handle_json(headers, lines[-1])
            else:
                print(f"Got an unexpected content type {headers['Content-Type']}")
                sys.stdout.flush()

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



def start_observation(shared_array):
    # set the global 
    datastore = shared_array

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