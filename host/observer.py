import sys
import threading
import time
import socket
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
from raspi_anchor_client import RaspiAnchorClient

fields = ['Content-Type', 'Content-Length', 'X-Timestamp-Sec', 'X-Timestamp-Usec']

# global that will point to a DataStore passed to this process.
datastore = None
# queue for sending info to user interface
to_ui_q = None
# queue for sending info to position estimator
to_pe_q = None
# global calibration mode
calibration_mode = True

def connect_raspi_anchor(url, anchor_num):
    ac = RaspiAnchorClient(anchor_num, datastore, to_ui_q, to_pe_q)


    # todo theres a different address for the video and the websocket
    # no need to advertise both right?

    raspberry_pi_ip = "192.168.1.151"  # Replace with your Pi's IP
    port_number = 8888  # Replace with your port
    stream_url = f"tcp://{raspberry_pi_ip}:{port_number}"  # Construct the URL
    ac.connect_video_stream(stream_url)
    video_thread = threading.Thread(target=ac.connect_video_stream, args=(stream_url,), daemon=True)
    video_thread.start()

    # a websocket connection for the control
    ac.connect_websocket()


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
            target=connect_raspi_anchor, args=(f"http://{hostport}/stream", anchor_num), daemon=True)
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