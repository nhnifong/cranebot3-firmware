"""
Test that starts up gripper server and connects to it with a gripper client
"""
import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, Mock, MagicMock, ANY, call
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
from generated.nf import telemetry, control, common

ws_port = 8765

class TestGripperClient(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.datastore = DataStore()

        self.mock_pool_class = Mock(spec=Pool)
        self.pool = self.mock_pool_class.return_value
        self.ob_mock = MagicMock()
        self.stat = StatCounter(self.ob_mock)
        self.pe = MagicMock()

        self.server_ws = None
        self.receiver = MagicMock()

        self.got_connection = asyncio.Event()
        self.close_test_server = asyncio.Event()
        self.server_task = asyncio.create_task(self.runTestServer())
        await asyncio.sleep(0.5)

    async def asyncTearDown(self):
        self.close_test_server.set()
        await self.server_task

    async def runTestServer(self):
        async with websockets.serve(self.serverHandler, "127.0.0.1", 8765):
            result = await self.close_test_server.wait()

    async def serverHandler(self, ws):
        print("Mock server: Client connected")
        self.got_connection.set()
        self.server_ws = ws
        hb_task = asyncio.create_task(self.send_heartbeat())
        while True:
            try:
                message = await ws.recv()
                update = json.loads(message)
                print(f'Mock server: received {update}')
                self.receiver.update(update)
            except ConnectionClosedOK:
                print("Mock server: Client disconnected")
                break
            except ConnectionClosedError as e:
                print(f"Mock server: Client disconnected with {e}")
                break
        self.server_ws = None
        r = await hb_task

    async def send_heartbeat(self):
        while self.server_ws is not None:
            await self.server_ws.send(json.dumps({'line_record': [1,2,3]}))
            await asyncio.sleep(0.9)

    async def test_shutdown_before_connect(self):
        gc = RaspiGripperClient("127.0.0.1", ws_port, self.datastore, self.ob_mock, self.pool, self.stat, self.pe )
        self.assertFalse(gc.connected)
        await gc.shutdown()

    async def clientSetup(self):
        self.gc = RaspiGripperClient("127.0.0.1", ws_port, self.datastore, self.ob_mock, self.pool, self.stat, self.pe )
        self.client_task = asyncio.create_task(self.gc.startup())
        result = await asyncio.wait_for(self.got_connection.wait(), 2)
        await asyncio.sleep(0.1) # client_task needs a chance to act

    async def clientTearDown(self):
        await self.gc.shutdown()
        result = await self.client_task

    async def test_connect_and_shutdown(self):
        # test that the gripper sends the right connection status messages using the mock observer's send_ui method
        await self.clientSetup()
        self.assertTrue(self.gc.connected)
        # should report to UI: Connecting, connected
        self.ob_mock.assert_has_calls([
            call.send_ui(component_conn_status=telemetry.ComponentConnStatus(
                is_gripper=True, websocket_status=telemetry.ConnStatus.CONNECTING, ip_address='127.0.0.1')),
            call.send_ui(component_conn_status=telemetry.ComponentConnStatus(
                is_gripper=True, websocket_status=telemetry.ConnStatus.CONNECTED, ip_address='127.0.0.1')),
        ])
        self.ob_mock.reset_mock()

        # This test does not set up a video stream. just immediately disconnect the client
        await self.clientTearDown()
        self.assertFalse(self.gc.connected)
        # gone
        # self.assertEqual({'gripper': True, 'websocket': 'none', 'video': 'none', 'ip_address': '127.0.0.1'}, ui_queue_message['connection_status'])
        print(self.ob_mock.send_ui.call_args_list)
        # note that when making an assertion about generated proto dataclasses with enums, an enum set to it's 0th value will show up as an unset field
        # so here, we are asserting that both websocket and video status are CONNSTATUS_NOT_DETECTED
        self.ob_mock.send_ui.assert_called_once_with(component_conn_status=telemetry.ComponentConnStatus(
                is_gripper=True, ip_address='127.0.0.1'))

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


    async def test_grip_sensors(self):
        """
        When gripper client receives "grip_sensors" it should put put the readings in the datastore.
        expect the the imu array in datatore and a rodruiges rotation vector to the ui via it's queue
        """
        await self.clientSetup()
        self.ob_mock.reset_mock() # don't care about the connection status messages in this test
        timestamp = 1.23
        data = {
            'time': timestamp,
            'quat': [4,5,6,7],
            'fing_v': 1.0,
            'fing_a': 10.0,
            'range': 30.0,
        }
        asyncio.create_task(self.server_ws.send(json.dumps({'grip_sensors': data})))
        await asyncio.sleep(0.1)
        expected_rotation_vector = np.array([ timestamp, 4,5,6,7])
        np.testing.assert_array_almost_equal(expected_rotation_vector, self.datastore.imu_quat.getLast())
        expected_range_record = np.array([timestamp, 30.0])
        np.testing.assert_array_almost_equal(expected_range_record, self.datastore.range_record.getLast())
        expected_finger_record = np.array([timestamp, 10.0, 1.0]) # time, angle, voltage
        np.testing.assert_array_almost_equal(expected_finger_record, self.datastore.finger.getLast())

        # assert that the correct telemetry item was sent to the UI
        self.ob_mock.send_ui.assert_called_once_with(grip_sensors=telemetry.GripperSensors(
            range = expected_range_record[1],
            angle = expected_finger_record[1],
            pressure = expected_finger_record[2]
        ))

        await self.clientTearDown()

    async def test_holding(self):
        """
        When gripper client receives "holding" it should call notify_update on the position estimator
        """
        await self.clientSetup()
        asyncio.create_task(self.server_ws.send(json.dumps({'holding': True})))
        await asyncio.sleep(0.1)
        self.pe.notify_update.assert_called_with({'holding': True})
        await self.clientTearDown()


    async def test_zero_winch(self):
        # common client setup
        await self.clientSetup()
        config = Config()

        # code under test
        task = asyncio.create_task(self.gc.zero_winch())
        await asyncio.sleep(0.1)
        self.receiver.update.assert_called_with({'zero_winch_line': None})
        self.assertFalse(task.done())
        asyncio.create_task(self.server_ws.send(json.dumps({'winch_zero_success': True})))
        await asyncio.sleep(0.1)
        self.assertTrue(task.done())
        result = await task

