
import asyncio
import os
import signal
import websockets
import time
import json
from cv_common import locate_markers, compose_poses, invert_pose, average_pose
import cv2
import numpy as np
import model_constants
from functools import partial
import threading

# number of origin detections to average
video_port = 8888
websocket_port = 8765

def pose_from_det(det):
    return (np.array(det['r'], dtype=float), np.array(det['t'], dtype=float))

# this client is designed for the raspberri pi based gripper
class RaspiGripperClient:
    def __init__(self, address, datastore, to_ui_q, to_pe_q, pool, stat):
        self.address = address
        self.datastore = datastore
        self.to_ui_q = to_ui_q
        self.to_pe_q = to_pe_q
        self.websocket = None
        self.connected = False  # status of connection to websocket
        self.receive_task = None  # Task for receiving messages from websocket
        self.video_task = None  # Task for streaming video
        self.frame_times = {}
        self.pool = pool
        self.stat

    def receive_video(self):
        # don't connect too early or you will be rejected
        time.sleep(6)
        video_uri = f'tcp://{self.address}:{video_port}'
        print(f'Connecting to {video_uri}')
        cap = cv2.VideoCapture(video_uri)
        print(cap)
        self.to_ui_q.put({'connection_status': {
            'gripper': True,
            'websocket': 2,
            'video': True,
        }})
        while self.connected:
            last_time = time.time()
            ret, frame = cap.read()
            if ret:
                fnum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                try:
                    timestamp = self.frame_times[fnum]
                    del self.frame_times[fnum]
                    now = time.time()
                    self.stat.latency.append(now - timestamp)
                    self.stat.framerate.append(1/(now - last_time))
                    last_time = now
                except KeyError:
                    print('received a frame without knowing when it was captured')
                    continue
                self.pool.apply_async(locate_markers, (frame,), callback=partial(self.handle_detections, timestamp=timestamp))

    def handle_frame_times(self, frame_time_list):
        for ft in frame_time_list:
            # this item represents the time that rpicam-vid captured the frame with the given number.
            # we need to know this for when we get frames from the stream
            if len(self.frame_times) > 500:
                print('How did we miss 500 frames? Video task crashed?')
                self.shutdown()
            self.frame_times[int(ft['fnum'])] = float(ft['time'])

    def handle_detections(self, detections, timestamp):
        """
        handle a list of aruco detections from the server
        """
        self.stat.detection_count += len(detections)
        for detection in detections:
            pass

    async def connect_websocket(self):
        # main client loop
        ws_uri = f"ws://{self.address}:{websocket_port}"
        print(f"Connecting to gripper at {ws_uri}...")
        try:
            # connect() can be used as an infinite asynchronous iterator to reconnect automatically on errors
            async for websocket in websockets.connect(ws_uri, max_size=None, open_timeout=10):
                # try:
                self.connected = True
                print(f"Connected to gripper at {ws_uri}.")
                await self.receive_loop(websocket)
        except asyncio.exceptions.CancelledError:
            print("Cancelling connection")
            return

    async def receive_loop(self, websocket):
        self.to_ui_q.put({'connection_status': {
            'gripper': True,
            'websocket': 2,
            'video': True,
        }})
        # loop of a single websocket connection.
        self.websocket = websocket
        # just could not make asyncio deal with this, so I used threading. hey it works, go figure
        vid_thread = threading.Thread(target=self.receive_video)
        vid_thread.start()
        # Loop until disconnected
        while self.connected:
            try:
                message = await websocket.recv()
                # print(f'received message of length {len(message)}')
                update = json.loads(message)
                if 'line_record' in update:
                    self.datastore.winch_line_record.insertList(update['line_record'])
                if 'frames' in update:
                    self.handle_frame_times(update['frames'])

            except Exception as e:
                # don't catch websockets.exceptions.ConnectionClosedOK because we want it to trip the infinite generator in websockets.connect
                # so it will stop retrying.
                print("Connection to gripper closed.")
                self.connected = False
                self.websocket = None
                self.to_ui_q.put({'connection_status': {
                    'gripper': True,
                    'websocket': 0,
                    'video': False,
                }})
                raise e
                break
        vid_thread.join()

    async def send_commands(self, update):
        if self.connected:
            await self.websocket.send(json.dumps(update))
        # just discard the update if not connected.

    async def slow_stop_spool(self):
        await self.send_commands({'length_plan' : []})

    async def startup(self):
        self.ct = asyncio.create_task(self.connect_websocket())
        await self.ct

    def shutdown(self):
        # this might get called twice
        print("\nWait for client shutdown")
        if self.connected:
            self.connected = False
            if self.websocket:
                asyncio.create_task(self.websocket.close())
        else:
            self.ct.cancel()

if __name__ == "__main__":
    from multiprocessing import Queue
    from data_store import DataStore
    datastore = DataStore(horizon_s=10, n_cables=4)
    to_ui_q = Queue()
    to_pe_q = Queue()
    to_ob_q = Queue()
    to_ui_q.cancel_join_thread()
    to_pe_q.cancel_join_thread()
    to_ob_q.cancel_join_thread()

    async def main():
        ac = RaspiGripperClient("127.0.0.1", datastore, to_ui_q, to_pe_q, None)
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(getattr(signal, 'SIGINT'), ac.shutdown)
        await ac.startup()
    asyncio.run(main())
