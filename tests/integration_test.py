import sys
import os
# This will let us import files and modules located in the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import unittest
import numpy as np
from unittest.mock import patch, Mock, MagicMock, ANY
from multiprocessing import Queue
from anchor_server import RaspiAnchorServer
from gripper_server import RaspiGripperServer
from debug_motor import DebugMotor
from observer import AsyncObserver
from inventorhatmini import InventorHATMini, SERVO_1, SERVO_2
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_vl53l1x import VL53L1X
from zeroconf import IPVersion
from zeroconf.asyncio import AsyncZeroconf
from mock_rpicam_vid import RPiCamVidMock
from scipy.spatial.transform import Rotation
from math import pi
from cv_common import compose_poses
import model_constants

# this test starts four anchor servers and a gripper server and then starts up a full instances of the observer
# which is expoected to discover and connect to all of them.
# and we then test the auto calibration on the system'
# the UI is left out of this integration test (for now)

def convert_pose(pose):
    # convert to X Y Z H P R
    # heading pitch and roll are euler angles in degrees
    hpr = Rotation.from_rotvec(pose[0]).as_euler('xyz', degrees=True)
    return (pose[1][0], pose[1][1], pose[1][2], hpr[0], hpr[1], hpr[2])

class TestSystemIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patchers = []

        self.setup_gripper_mocks()

        for p in self.patchers:
            p.start()

        self.to_ui_q = Queue()
        self.to_ob_q = Queue()
        self.to_ui_q.cancel_join_thread()
        self.to_ob_q.cancel_join_thread()

        self.anchor_servers = []
        self.server_tasks = []
        self.camera_mocks = []

        # before starting any service, set it's zeroconf instance to a special version that searches on localhost only
        self.zc = AsyncZeroconf(ip_version=IPVersion.All, interfaces=["127.0.0.1"])

        # anchor poses to use when rendering simulated camera views
        anchor_poses = np.array([
            ((0, 0, -pi/4), (3, 3, 3)),
            ((0, 0, -3*pi/4), (3, -3, 3)),
            ((0, 0, pi/4), (-3, 3, 3)),
            ((0, 0, 3*pi/4), (-3, -3, 3)),
        ])

        # Make four of these on different ports
        for i in range(4):
            mock_camera = RPiCamVidMock(
                width=4608, height=2592, framerate=15, port=8888+i,
                initial_image_filepath='../boards/origin.png',
                # this is the panda3d coordinate space, so +Y is up.
                camera_pose=convert_pose(compose_poses([anchor_poses[i], model_constants.anchor_camera])),
                billboard_initial_pose=(0, 0, 0, 0, 0, 0) # Billboard at origin, no rotation (facing up)
            )
            self.camera_mocks.append(mock_camera)
            server = RaspiAnchorServer(power_anchor=False, mock_motor=DebugMotor(), mock_camera=mock_camera)
            server.zc = self.zc
            self.anchor_servers.append(server)
            self.server_tasks.append(asyncio.create_task(server.main(port=i+8765, name=f'cranebot-anchor-service.test_{i}')))
            await asyncio.sleep(0.1)  # Give the server a moment to start

        self.gripper_server = RaspiGripperServer(DebugMotor())
        self.gripper_server.zc = self.zc
        self.server_tasks.append(asyncio.create_task(self.gripper_server.main(port=8764, name=f'cranebot-gripper-service.test')))
        await asyncio.sleep(0.1)  # Give the server a moment to start

        # Start the observer
        # Zeroconf will attempt to discover services on localhost only.
        self.ob = AsyncObserver(self.to_ui_q, self.to_ob_q)
        self.ob.aiozc = self.zc
        self.ob_task = asyncio.create_task(self.ob.main())
        await asyncio.sleep(0.1)

    def setup_gripper_mocks(self):
        # mock inventor hat mini
        self.mock_hat_class = Mock(spec=InventorHATMini)
        self.patchers.append(patch('gripper_server.InventorHATMini', self.mock_hat_class))
        self.mock_hat = self.mock_hat_class.return_value

        # gpio pins
        self.mock_hat.gpio_pin_value.return_value = 1.0

        self.servos = {SERVO_1: MagicMock(), SERVO_2: MagicMock()}
        self.mock_hat.servos = MagicMock()
        self.mock_hat.servos.__getitem__.side_effect = lambda key: self.servos[key]

        self.encoders = {0: MagicMock(), 1: MagicMock()}
        self.mock_hat.encoders = MagicMock()
        self.mock_hat.encoders.__getitem__.side_effect = lambda key: self.encoders[key]

        # prevent this call from failing         i2c = busio.I2C(board.SCL, board.SDA)
        self.patchers.append(patch('gripper_server.board.SCL', None, create=True))
        self.patchers.append(patch('gripper_server.board.SDA', None, create=True))
        self.patchers.append(patch('gripper_server.busio.I2C', lambda a, b: None,))

        # mock IMU
        self.mock_imu_class = Mock(spec=BNO08X_I2C)
        self.patchers.append(patch('gripper_server.BNO08X_I2C', self.mock_imu_class))
        self.mock_imu = self.mock_imu_class.return_value
        self.mock_imu.quaternion = (1,2,3,4)

        # mock Rangefinder
        self.mock_range_class = Mock(spec=VL53L1X)
        self.patchers.append(patch('gripper_server.VL53L1X', self.mock_range_class))
        self.mock_range = self.mock_range_class.return_value
        self.mock_range.model_info = (1,2,3)
        self.mock_range.data_ready = True
        self.mock_range.distance = 30.0

    async def asyncTearDown(self):
        # this should have the effect of making the update listener task stop, which the main task is awaiting.
        # it should then run it's own async_close method, which runs the async shutdown methods on zeroconf
        # and any connected clients.
        await self.ob.aiozc.async_unregister_all_services()
        self.to_ob_q.put({'STOP':None})
        await asyncio.wait_for(self.ob_task, 1)

        for a in self.anchor_servers:
            a.shutdown()
        self.gripper_server.shutdown()
        result = await asyncio.wait_for(asyncio.gather(*self.server_tasks), 1)

        for p in self.patchers:
            p.stop()

    async def test_auto_calibration(self):
        await asyncio.sleep(16)
        self.assertEqual(len(self.ob.bot_clients), 5)
        await self.ob.full_auto_calibration()
        self.assertEqual(len(self.ob.bot_clients), 5)
        self.assertEqual(1, 0)

