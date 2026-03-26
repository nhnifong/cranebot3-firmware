import pytest
pytestmark = pytest.mark.pi
pytest.importorskip("gpiodevice")

import asyncio
import unittest
import numpy as np
from unittest.mock import patch, Mock, MagicMock, ANY
from multiprocessing import Queue
from adafruit_vl53l1x import VL53L1X
from zeroconf import IPVersion
from zeroconf.asyncio import AsyncZeroconf
from math import pi
from adafruit_mpu6050 import MPU6050
from adafruit_ads1x15 import AnalogIn, ADS1015
import websockets

# from local tests folder
# from mock_rpicam_vid import RPiCamVidMock, convert_pose

from nf_robot.robot.anchor_server import RaspiAnchorServer
from nf_robot.robot.gripper_arp_server import GripperArpServer
from nf_robot.robot.debug_motor import DebugMotor
from nf_robot.robot.simple_st3215 import SimpleSTS3215
from nf_robot.host.observer import AsyncObserver
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, control, common

# this test starts four anchor servers and a gripper server and then starts up a full instances of the observer
# which is expoected to discover and connect to all of them.
# and we then test the auto calibration on the system'
# the UI is left out of this integration test (for now)

class TestSystemIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patchers = []

        self.setup_gripper_mocks()

        for p in self.patchers:
            p.start()

        self.anchor_servers = []
        self.server_tasks = []

        # before starting any service, set it's zeroconf instance to a special version that searches on localhost only
        self.zc = AsyncZeroconf(ip_version=IPVersion.All, interfaces=["127.0.0.1"])

        # # this class listens on four ports, showing views of the scene from four angles
        # # anchor servers will not need to start rpicam-vid, they only need to inform their
        # # client to connect on the right port.
        # self.mock_camera = RPiCamVidMock(
        #     width=4608, height=2592, framerate=5,
        #     gantry_initial_pose=(0.5, 0.5, 1, 0, 0, 0) # off center, 1m from floor
        # )
        # # anchor poses to use in simulated enviroment
        # self.mock_camera.set_camera_poses(np.array([
        #     (3, 3, 2.5, 135, -28, 0),
        #     (3, -3, 2.5, 45, -28, 0),
        #     (-3, 3, 2.5, 225, -28, 0),
        #     (-3, -3, 2.5, 315, -28, 0),
        # ]))

        # self.mock_cam_task = asyncio.create_task(self.mock_camera.start_server())

        # # make a local variable to contain a tight or not tight bool and a functiion that captures it.
        # local_tight_var = {'tight': True}
        # def local_t():
        #     return local_tight_var['tight']

        # # Make four of these on different ports
        # for i in range(4):
        #     server = RaspiAnchorServer(power_anchor=(i==0), mock_motor=DebugMotor())
        #     server.mock_camera_port = i+8888
        #     server.zc = self.zc
        #     server.tight_check = local_t
        #     self.anchor_servers.append(server)
        #     self.server_tasks.append(asyncio.create_task(server.main(port=i+8765, name=f'cranebot-anchor-service.test_{i}')))
        #     await asyncio.sleep(0.1)  # Give the server a moment to start

        self.gripper_server = GripperArpServer()
        self.gripper_server.zc = self.zc
        self.server_tasks.append(asyncio.create_task(self.gripper_server.main(port=8764, name=f'cranebot-gripper-arpeggio-service.test')))
        await asyncio.sleep(0.1)  # Give the server a moment to start

        # Start the observer
        # Zeroconf will attempt to discover services on localhost only.
        self.ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=4249)
        self.ob.aiozc = self.zc

    async def start_observer(self):
        self.ob_task = asyncio.create_task(self.ob.main())
        await asyncio.wait_for(self.ob.startup_complete.wait(), 20)

    def setup_gripper_mocks(self):
        # mock servos SimpleSTS3215
        self.mock_servo_controller_class = Mock(spec=SimpleSTS3215)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.SimpleSTS3215', self.mock_servo_controller_class))
        self.mock_servo_controller = self.mock_servo_controller_class.return_value
        self.motor_feedback = MagicMock()
        self.motor_feedback.return_value = {
            "position": 2000,
            "speed": 0,
            "load": 0,
            "voltage": 7.4,
            "temp": 20,
            "moving": 0,
        }
        self.mock_servo_controller.get_feedback = self.motor_feedback

        # prevent this call from failing         i2c = busio.I2C(board.SCL, board.SDA)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.board.SCL', None, create=True))
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.board.SDA', None, create=True))
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.busio.I2C', lambda a, b: None,))

        # mock accelerometer MPU6050
        self.mock_imu_class = Mock(spec=MPU6050)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.MPU6050', self.mock_imu_class))
        self.mock_imu = self.mock_imu_class.return_value
        self.mock_imu.gyro = (0,0,0)

        # mock analog to digital converter ADS1015
        self.mock_ads_class = Mock(spec=ADS1015)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.ADS1015', self.mock_ads_class))
        self.mock_ads = self.mock_ads_class.return_value

        self.mock_analog_class = Mock(spec=AnalogIn)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.AnalogIn', self.mock_analog_class))
        self.mock_analog = self.mock_analog_class.return_value
        self.mock_analog.voltage = (2.5)

        # mock Rangefinder
        self.mock_range_class = Mock(spec=VL53L1X)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.VL53L1X', self.mock_range_class))
        self.mock_range = self.mock_range_class.return_value
        self.mock_range.model_info = (1,2,3)
        self.mock_range.data_ready = True
        self.mock_range.distance = 30.0

    async def asyncTearDown(self):
        # this should have the effect of making the update listener task stop, which the main task is awaiting.
        # it should then run it's own async_close method, which runs the async shutdown methods on zeroconf
        # and any connected clients.
        await self.ob.aiozc.async_unregister_all_services()

        for a in self.anchor_servers:
            a.shutdown()
        self.gripper_server.shutdown()
        # self.mock_camera.stop_server()
        self.ob_task.cancel()
        # allow up to 2 seconds for shutdown.
        # result = await asyncio.wait_for(asyncio.gather(*self.server_tasks, self.ob_task, self.mock_cam_task), 2)
        result = await asyncio.wait_for(asyncio.gather(*self.server_tasks, self.ob_task), 2)

        for p in self.patchers:
            p.stop()

    def move_fake_gantry(self, pos):
        self.mock_camera.update_gantry_pose((*pos, 0,0,0))

    async def test_connect_gripper(self):

        self.ob.test_gantry_goal_callback = self.move_fake_gantry
        await self.start_observer()
        await asyncio.sleep(0.1)

        print('setting addresses for local component servers')
        name = "123.cranebot-gripper-arpeggio-service.test"
        self.ob.config.gripper.service_name = name
        self.ob.config.gripper.address = '127.0.0.1'
        self.ob.config.gripper.port = 8764

        await self.ob.gripper_client_connected.wait()

        self.assertFalse(self.server_tasks[0].done(), "Gripper server should still be running after getting a connection")
        self.assertEqual(len(self.ob.bot_clients), 1)
        
        self.ws_out = None
        self.gripper_connstatus_received = asyncio.Event()
        self.last_commanded_grip_received = asyncio.Event()
        self.last_commanded_grip = None

        def handle(message):
            batch = telemetry.TelemetryBatchUpdate().parse(message)
            for item in batch.updates:
                print(item)
                if item.component_conn_status:
                    # set event if connection statuses and anchor poses are received
                    cs = item.component_conn_status
                    if cs.is_gripper and cs.websocket_status == telemetry.ConnStatus.CONNECTED:
                        self.gripper_connstatus_received.set()
                elif item.last_commanded_grip:
                    self.last_commanded_grip_received.set()
                    self.last_commanded_grip = item.last_commanded_grip


        async def sendControl(**kwargs):
            batch = control.ControlBatchUpdate(
                robot_id="12345",
                updates=[control.ControlItem(**kwargs)],
            )
            r = await self.ws_out.send(bytes(batch))

        # Connect to telemetry as a UI
        async def listenTelemetry():
            async with websockets.connect("ws://127.0.0.1:4249") as ws:
                self.ws_out = ws
                print('listening to telemetry stream')
                await asyncio.sleep(0.01)
                try:
                    async for message in ws:
                        handle(message)
                except (websockets.ConnectionClosedOK, asyncio.exceptions.CancelledError) as e:
                    return
            print('out loop')

        listen_task = asyncio.create_task(listenTelemetry())
        await asyncio.wait_for(self.gripper_connstatus_received.wait(), 5)

        # Test controls specific to gripper
        await sendControl(move=control.CombinedMove(wrist_speed=20,finger_speed=21))
        await asyncio.sleep(0.01)
        # confirm gripper server received command by interrogating internal state
        self.assertEqual(self.gripper_server.desired_wrist_speed, 20)
        self.assertEqual(self.gripper_server.desired_finger_speed, 21)
        # confirm observer sends back telemetry about last commanded grip
        await asyncio.wait_for(self.last_commanded_grip_received.wait(), 0.4)
        self.assertEqual(self.last_commanded_grip.wrist_speed, 20)
        self.assertEqual(self.last_commanded_grip.finger_speed, 21)

        # stop listening to telemetry stream
        listen_task.cancel()
        r = await asyncio.wait_for(listen_task, 4)


        # result = await asyncio.wait_for(self.ob.full_auto_calibration(), 60*100)
        # # confirm nothing caused it to disconnect from the clients
        # self.assertEqual(len(self.ob.bot_clients), 5)
        # check that none of the calibration parameters obtained from SLSQP are hard up agains their bounds.

# when a self.seek_gantry_goal method finishes in a test, it should set the new position of the mock enviroment gantry box