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
from raspi_gripper_client import RaspiGripperClient
from stats import StatCounter
from config import Config

class TestGripperClient(unittest.IsolatedAsyncioTestCase):

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
        gc = RaspiGripperClient("127.0.0.1", self.datastore, self.to_ui_q, self.to_pe_q, self.to_ob_q, self.pool, self.stat)
        self.assertFalse(gc.connected)
        await gc.shutdown()

    async def clientSetup(self):
        self.gc = RaspiGripperClient("127.0.0.1", self.datastore, self.to_ui_q, self.to_pe_q, self.to_ob_q, self.pool, self.stat)
        self.client_task = asyncio.create_task(self.gc.startup())
        result = await asyncio.wait_for(self.got_connection.wait(), 2)
        await asyncio.sleep(0.1) # client_task needs a chance to act

    async def clientTearDown(self):
        await self.gc.shutdown()
        result = await self.client_task

    async def test_connect_and_shutdown(self):
        await self.clientSetup()
        self.assertTrue(self.gc.connected)
        # Connecting
        ui_queue_message = self.to_ui_q.get(timeout=1)
        self.assertEqual({'gripper': True, 'websocket': 1, 'video': False}, ui_queue_message['connection_status'])
        # Online
        ui_queue_message = self.to_ui_q.get(timeout=1)
        self.assertEqual({'gripper': True, 'websocket': 2, 'video': False}, ui_queue_message['connection_status'])
        await self.clientTearDown()
        self.assertFalse(self.gc.connected)
        # gone
        ui_queue_message = self.to_ui_q.get(timeout=1)
        self.assertEqual({'gripper': True, 'websocket': 0, 'video': False}, ui_queue_message['connection_status'])

        # if gripper_client is going to use the configs in configuration.json, then so will we.
        config = Config()
        self.receiver.update.assert_called_with({'set_config_vars': config.gripper_vars})

    async def test_line_record(self):
        """
        When gripper client receives "line_record" it should put it in the winch line array in datastore
        """
        await self.clientSetup()
        linerecord = [(88.0,2.0,0.0), (89.0,2.1,0.0)]
        asyncio.create_task(self.server_ws.send(json.dumps({'line_record': linerecord})))
        await asyncio.sleep(0.1)
        np.testing.assert_array_almost_equal(linerecord[-1], self.datastore.winch_line_record.getLast())
        await self.clientTearDown()


    async def test_imu(self):
        """
        When gripper client receives "imu" it should put it in the imu array in datatore
        and send a rodruiges rotation vector to the ui via it's queue
        """
        await self.clientSetup()
        timestamp = 1.23
        imu_data = {
            'quat': [timestamp, 4,5,6,7],
        }
        asyncio.create_task(self.server_ws.send(json.dumps({'imu': imu_data})))
        await asyncio.sleep(0.1)
        expected_rotation_vector = np.array([ timestamp, 0.140709, -0.422127, -1.5478])
        np.testing.assert_array_almost_equal(expected_rotation_vector, self.datastore.imu_rotvec.getLast())
        # throw away two connection status messages
        ui_queue_message = self.to_ui_q.get(timeout=1)
        ui_queue_message = self.to_ui_q.get(timeout=1)
        # this is the message we want.
        ui_queue_message = self.to_ui_q.get(timeout=1)
        print(f'ui_queue_message = {ui_queue_message}')
        self.assertTrue('gripper_rvec' in ui_queue_message)
        await self.clientTearDown()

    async def test_range(self):
        """
        When gripper client receives "range" it should put it in the range array in datastore
        """
        await self.clientSetup()
        rangerecord = [88.0,30.0]
        asyncio.create_task(self.server_ws.send(json.dumps({'range': rangerecord})))
        await asyncio.sleep(0.1)
        np.testing.assert_array_almost_equal(rangerecord, self.datastore.range_record.getLast())
        await self.clientTearDown()

    async def test_holding(self):
        """
        When gripper client receives "holding" it should forward the update to the pe queue
        """
        await self.clientSetup()
        asyncio.create_task(self.server_ws.send(json.dumps({'holding': True})))
        await asyncio.sleep(0.1)
        pe_queue_message = self.to_pe_q.get(timeout=1)
        self.assertEqual({'holding': True}, pe_queue_message)
        await self.clientTearDown()

