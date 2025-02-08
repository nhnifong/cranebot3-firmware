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
# queue for receiving info meant for this process
to_ob_q = None
# global calibration mode
calibration_mode = True


cranebot_anchor_service_name = 'cranebot-anchor-service'
cranebot_gripper_service_name = 'cranebot-gripper-service'
# Manager of multiple threads running clients connected to each robot component
class CranebotListener(ServiceListener):
    def __init__(self):
        super().__init__()

        # all keyed by server name
        self.bot_threads = {}
        self.bot_clients = {}

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

    def listen_position_updates(url):
        while True:     
            updates = to_ob_q.get()
            if 'future_anchor_lines' in updates:
                updates['future_anchor_lines']
            if 'future_winch_line' in updates:
                pass

    def start_client(self, info, type):
        address = socket.inet_ntoa(info.addresses[0])
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
        if type == 'anchor':
            ac = RaspiAnchorClient(address, anchor_num, datastore, to_ui_q, to_pe_q)
        elif type == 'gripper':
            return
        else:
            print(f"wtf is a {type}")
            return
        self.bot_clients[info.server] = ac
        asyncio.create_task(ac.connect_all)
        # self.bot_threads[info.server] = threading.Thread(
        #     target=ac.connect, args=(address,), daemon=True)
        # self.bot_threads[info.server].start()

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} updated")
        sys.stdout.flush()
        info = zc.get_service_info(type_, name)
        print(info)
        # if name.split(".")[1] == cranebot_service_name:
        #     if info.server is not None and info.server != '':
        #         self.start_client(info)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        print(f"Service {name} removed")
        sys.stdout.flush()
        info = zc.get_service_info(type_, name)
        # the thread probably already stopped when the pipe broke, but just in case
        self.bot_threads[info.server].terminate()
        del self.bot_threads[info.server]

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        print(f"Service {name} added, service info: {info}")
        sys.stdout.flush()
        if name.split(".")[1] == cranebot_anchor_service_name:
            if info.server is not None and info.server != '':
                self.start_client(info, type='anchor')



def start_observation(shared_array, to_ui_q, to_pe_q, to_ob_q):
    # set the global 
    datastore = shared_array
    to_ui_q = to_ui_q # for sending to the UI
    to_pe_q = to_pe_q # for sending to the position estimator
    to_ob_q = to_ob_q # queue where other processes send to us

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

if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    start_observation(datastore, to_ui_q, to_pe_q, to_ob_q)