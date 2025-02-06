import sys
import json
import requests
import threading
import time
import socket
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from cv_common import locate_markers
import cv2
import numpy as np
from calibration import compose_poses, invert_pose, average_pose
import model_constants

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']

# global that will point to a DataStore passed to this process.
datastore = None
# queue for sending info to user interface
to_ui_q = None
# queue for sending info to position estimator
to_pe_q = None
# maximum number of origin detections to keep
max_origin_detections = 40
calibration_mode = False

class AnchorClient:
    def __init__(self, anchor_num):
        self.origin_detections = []
        self.anchor_num = anchor_num # which anchor are we connected to
        try:
            # read calibration data from file
            saved_info = np.load('anchor_pose_%i' % self.anchor_num)
            self.anchor_pose = tuple(saved_info['pose'])
        except FileNotFoundError:
            self.anchor_pose = (np.array([0,0,0]), np.array([0,0,0]))

        # to help with a loop that does the same thing four times in handle_image
        # name, offset, datastore
        self.arucos = [
            ('gripper_front', model_constants.gripper_aruco_front_inv, datastore.gripper_pose),
            ('gripper_back', model_constants.gripper_aruco_back_inv, datastore.gripper_pose),
            ('gantry_front', model_constants.gantry_aruco_front_inv, datastore.gantry_pose),
            ('gantry_back', model_constants.gantry_aruco_back_inv, datastore.gantry_pose),
        ]

    def calibrate_pose(self):
        global to_ui_q
        global to_pe_q
        # recalculate the pose of the connected anchor from recent origin detections
        anchor_cam_pose = [invert_pose(*average_pose(det)) for det in self.origin_detections]
        self.anchor_pose = compose_poses([anchor_cam_pose, invert_pose(model_constants.gripper_camera)])
        np.savez('anchor_pose_%i' % self.anchor_num, pose = pose)
        to_ui_q.put({'anchor_pose': (self.anchor_num, pose)})
        to_pe_q.put({'anchor_pose': (self.anchor_num, pose)})


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
            # within this scope, interpret the symbol calibration_mode as referring to the global calibration_mode
            global calibration_mode
            if calibration_mode:
                for detection in locate_markers(frame):
                    print(f"Found board: {detection.name}")
                    print(f"Timestamp: {timestamp}")
                    print(f"Rotation Vector: {detection.rvec}")
                    print(f"Translation Vector: {detection.tvec}")
                    # sys.stdout.flush()

                    if detection.name == "origin":
                        origin_detections.append(detection)
                        if len(origin_detections) > max_origin_detections:
                            origin_detections.pop(0)
            else:
                for detection in locate_markers(frame):
                    # rotate and translate to where that object's origin would be
                    # given the position and rotation of the camera that made this observation (relative to the origin)
                    # store the time and that position in the appropriate measurement array in observer.

                    for name, offset, dest  in self.arucos:
                        if detection.name == name:
                            # you have the pose of gripper_front relative to a particular anchor camera
                            # Anchor is relative to the origin
                            # anchor camera is relative to anchor
                            # gripper_front is relative to anchor camera
                            # gripper is relative to gripper_front
                            # gripper_grommet is relative to gripper
                            gripper_global_pose = np.array(compose_poses([
                                self.anchor_pose, # obtained from calibration
                                model_constants.anchor_camera, # constant
                                (detection.rotation, detection.translation), # the pose obtained just now
                                offset, # constant
                            ]))
                            dest.insert(np.concatenate([[timestamp], gripper_global_pose.reshape(6)]))


    def handle_json(self, headers, buf):
        """
        handle a single json blob from the stream
        """
        accel = json.loads(buf.decode())
        # datastore.imu_accel.insert(np.array([timestamp, accel['x'], accel['y'], accel['z']]))

    def parse_mixed_replace_stream(self, url):
        """
        Parses a multipart/x-mixed-replace stream using the requests library.
        Blocks until the stream closes.

        Args:
            url: The URL of the stream.

        Prints the content type of each part in the stream.
        """
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
                    self.handle_image(headers, part[data_start:])
                elif headers['Content-Type'] == 'application/json':
                    self.handle_json(headers, part[data_start:])
                else:
                    print(f"Got an unexpected content type {headers['Content-Type']}")
                    sys.stdout.flush()
            print("consumed stream")

def connect_anchor_client(url, anchor_num):
    ac = AnchorClient(anchor_num)
    ac.parse_mixed_replace_stream(url)


cranebot_service_name = 'cranebot-service'
# listener for updating the list of available robot component servers
class CranebotListener(ServiceListener):
    def __init__(self):
        super().__init__()
        self.available_bots = {}
        self.bot_threads = {}

        # read a mapping of server names to anchor numbers from a file
        self.next_available_anchor_num = 0;
        self.anchor_num_map = {}
        self.load_anchor_num_map()

    def load_anchor_num_map(self):
        try:
            with open('anchor_mapping.txt', 'r') as f:
                for line in f:
                    s = line.split(':')
                    self.anchor_num_map[s[0]] = int(s[1])
            self.next_available_anchor_num = max(self.anchor_num_map.values())+1
        except FileNotFoundError:
            pass

    def save_anchor_num_map(self):
        with open('anchor_mapping.txt', 'w') as f:
            for k,v in self.anchor_num_map:
                f.write(f'{k}:{v}\n')

    def start_stream(self, info):
        address = socket.inet_ntoa(info.addresses[0])
        hostport = f"{address}:{info.port}"
        self.available_bots[info.server] = hostport
        # self.bot_threads[info.server].terminate()

        # the number of anchors is decided ahead of time (in main.py)
        # but they are assigned numbers as we find them on the network
        # and the chosen numbers are persisted on disk
        if info.server in self.anchor_num_map:
            anchor_num = self.anchor_num_map[info.server]
        else:
            anchor_num = self.next_available_anchor_num
            self.anchor_num_map[info.server] = anchor_num
            self.save_anchor_num_map()

        self.bot_threads[info.server] = threading.Thread(
            target=connect_anchor_client, args=(f"http://{hostport}/stream", anchor_num), daemon=True)
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



def start_observation(shared_array, to_ui_q, to_pe_q):
    # set the global 
    datastore = shared_array
    to_ui_q = to_ui_q
    to_pe_q = to_pe_q

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

    # run discovery in main thread
    service_discovery_task()

    # run discovery in it's own thread
    # discovery_thread = threading.Thread(target=service_discovery_task, daemon=True)
    # discovery_thread.start()

# start_observation(None, None)