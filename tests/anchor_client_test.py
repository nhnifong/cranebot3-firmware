"""
Test that starts up gripper server and connects to it with a gripper client
"""
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, Mock, MagicMock, ANY
import asyncio
import numpy as np
import websockets
from websockets.exceptions import (
    ConnectionClosedOK,
    ConnectionClosedError,
)
from multiprocessing import Pool, Queue
import json

from data_store import DataStore
from raspi_anchor_client import RaspiAnchorClient
from stats import StatCounter
from config import Config

class TestAnchorClient(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.datastore = DataStore(horizon_s=10, n_cables=4)
        self.to_ui_q = Queue()
        self.to_pe_q = Queue()
        self.to_ob_q = Queue()
        self.to_ui_q.cancel_join_thread()
        self.to_pe_q.cancel_join_thread()
        self.to_ob_q.cancel_join_thread()

        self.mock_pool_class = Mock(spec=Pool)
        self.pool = self.mock_pool_class.return_value

        self.stat = StatCounter(self.to_ui_q)

        self.shape_tracker = MagicMock()

        self.server_ws = None
        self.receiver = MagicMock()

        self.got_connection = asyncio.Event()
        self.close_test_server = asyncio.Event()
        asyncio.create_task(self.runTestServer())
        await asyncio.sleep(0.5)

    async def asyncTearDown(self):
        self.close_test_server.set()

    async def runTestServer(self):
        async with websockets.serve(self.serverHandler, "127.0.0.1", 8765):
            result = await self.close_test_server.wait()

    async def serverHandler(self, ws):
        print('connected')
        self.got_connection.set()
        self.server_ws = ws
        while True:
            try:
                message = await ws.recv()
                update = json.loads(message)
                print(update)
                self.receiver.update(update)
            except ConnectionClosedOK:
                print("Client disconnected")
                break
            except ConnectionClosedError as e:
                print(f"Client disconnected with {e}")
                break

    async def test_shutdown_before_connect(self):
        ac = RaspiAnchorClient("127.0.0.1", 1, self.datastore, self.to_ui_q, self.to_pe_q, self.to_ob_q, self.pool, self.stat, self.shape_tracker)
        self.assertFalse(ac.connected)
        await ac.shutdown()

    async def clientSetup(self):
        self.ac = RaspiAnchorClient("127.0.0.1", 1, self.datastore, self.to_ui_q, self.to_pe_q, self.to_ob_q, self.pool, self.stat, self.shape_tracker)
        self.client_task = asyncio.create_task(self.ac.startup())
        result = await asyncio.wait_for(self.got_connection.wait(), 2)
        await asyncio.sleep(0.1) # client_task needs a chance to act

    async def clientTearDown(self):
        await self.ac.shutdown()
        result = await self.client_task

    async def test_connect_and_shutdown(self):
        await self.clientSetup()
        self.assertTrue(self.ac.connected)

        # Anchor pose was sent first
        ui_queue_message = self.to_ui_q.get(timeout=1)
        self.assertTrue('anchor_pose' in ui_queue_message)
        anchor_num, pose = ui_queue_message['anchor_pose']
        self.assertTrue(1, anchor_num)
        self.assertTrue(2, len(pose))
        self.assertTrue(3, len(pose[0]))
        self.assertTrue(3, len(pose[1]))
        # Connecting
        ui_queue_message = self.to_ui_q.get(timeout=1)
        self.assertEqual({'anchor_num': 1, 'websocket': 1, 'video': False}, ui_queue_message['connection_status'])
        # Online
        ui_queue_message = self.to_ui_q.get(timeout=1)
        self.assertEqual({'anchor_num': 1, 'websocket': 2, 'video': False}, ui_queue_message['connection_status'])
        await self.clientTearDown()
        self.assertFalse(self.ac.connected)
        # gone
        ui_queue_message = self.to_ui_q.get(timeout=1)
        self.assertEqual({'anchor_num': 1, 'websocket': 0, 'video': False}, ui_queue_message['connection_status'])

        # if gripper_client is going to use the configs in configuration.json, then so will we.
        config = Config()
        self.receiver.update.assert_called_with({'set_config_vars': config.vars_for_anchor(1)})

    async def test_line_record(self):
        """
        When an achor client receives "line_record" it should put it in the anchor line array in datastore corresponding to it's anchor number
        """
        await self.clientSetup()
        # expect time, length, tension
        linerecord = [(88.0,2.0,1,0), (89.0,2.1,1,0)]
        asyncio.create_task(self.server_ws.send(json.dumps({'line_record': linerecord})))
        await asyncio.sleep(0.1)
        np.testing.assert_array_almost_equal(linerecord[-1], self.datastore.anchor_line_record[1].getLast())
        await self.clientTearDown()

    async def test_handle_detections_gantry(self):
        """
        When the anchor client gets an aruco marker detection from the pool
        if it is a gantry or gripper aruco, and the client is not in pose cal mode
        it should use that info to add an entry to the gantry or gripper pose arrays in the datastore.
        """
        detection_list = [
            {
                'n': 'gantry_front',
                'r': [0.1,0.2,0.3],
                't': [0.4,0.5,0.6]
            },
        ]
        timestamp = 123456789.0
        await self.clientSetup()
        self.ac.handle_detections(detection_list, timestamp)
        last_pose = self.datastore.gantry_pose.getLast()
        self.assertEqual(timestamp, last_pose[0])
        await self.clientTearDown()