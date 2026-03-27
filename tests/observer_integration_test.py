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

from nf_robot.robot.anchor_server import RaspiAnchorServer
from nf_robot.robot.gripper_arp_server import GripperArpServer
from nf_robot.robot.debug_motor import DebugMotor
from nf_robot.robot.simple_st3215 import SimpleSTS3215
from nf_robot.host.observer import AsyncObserver
from nf_robot.common.pose_functions import compose_poses
import nf_robot.common.definitions as model_constants
from nf_robot.generated.nf import telemetry, control, common

class TestSystemIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patchers = []
        self.setup_gripper_mocks()

        # Mock computer vision specific functions that would require real cameras/calibration
        self.mock_project_pixels = MagicMock(return_value=[[1.5, 2.5, 0.0]])
        self.patchers.append(patch('nf_robot.host.observer.project_pixels_to_floor', self.mock_project_pixels))

        for p in self.patchers:
            p.start()

        self.anchor_servers = []
        self.server_tasks = []

        # before starting any service, set it's zeroconf instance to a special version that searches on localhost only
        self.zc = AsyncZeroconf(ip_version=IPVersion.All, interfaces=["127.0.0.1"])

        local_tight_var = {'tight': True}
        def local_t():
            return local_tight_var['tight']

        # Make four of these on different ports
        for i in range(4):
            server = RaspiAnchorServer(power_anchor=(i==0), mock_motor=DebugMotor())
            server.zc = self.zc
            server.tight_check = local_t
            self.anchor_servers.append(server)
            self.server_tasks.append(asyncio.create_task(server.main(port=i+8765, name=f'cranebot-anchor-service.test_{i}')))
            await asyncio.sleep(0.05)  

        self.gripper_server = GripperArpServer()
        self.gripper_server.zc = self.zc
        self.server_tasks.append(asyncio.create_task(self.gripper_server.main(port=8764, name=f'cranebot-gripper-arpeggio-service.test')))
        await asyncio.sleep(0.05)  

        # Start the observer
        self.ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=4249)
        self.ob.aiozc = self.zc
        
        # Test utilities
        self.ws_out = None
        self.listen_task = None
        self.received_telemetry = []

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
            "position": 2000, "speed": 0, "load": 0, "voltage": 7.4, "temp": 20, "moving": 0,
        }
        self.mock_servo_controller.get_feedback = self.motor_feedback

        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.board.SCL', None, create=True))
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.board.SDA', None, create=True))
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.busio.I2C', lambda a, b: None,))

        self.mock_imu_class = Mock(spec=MPU6050)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.MPU6050', self.mock_imu_class))
        self.mock_imu = self.mock_imu_class.return_value
        self.mock_imu.gyro = (0,0,0)

        self.mock_ads_class = Mock(spec=ADS1015)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.ADS1015', self.mock_ads_class))
        self.mock_ads = self.mock_ads_class.return_value

        self.mock_analog_class = Mock(spec=AnalogIn)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.AnalogIn', self.mock_analog_class))
        self.mock_analog = self.mock_analog_class.return_value
        self.mock_analog.voltage = (2.5)

        self.mock_range_class = Mock(spec=VL53L1X)
        self.patchers.append(patch('nf_robot.robot.gripper_arp_server.VL53L1X', self.mock_range_class))
        self.mock_range = self.mock_range_class.return_value
        self.mock_range.model_info = (1,2,3)
        self.mock_range.data_ready = True
        self.mock_range.distance = 30.0

    async def asyncTearDown(self):
        if self.listen_task:
            self.listen_task.cancel()
        if self.ws_out:
            await self.ws_out.close()

        await self.ob.aiozc.async_unregister_all_services()
        for a in self.anchor_servers:
            a.shutdown()
        self.gripper_server.shutdown()
        if hasattr(self, 'ob_task'):
            self.ob_task.cancel()
        
        await asyncio.wait_for(asyncio.gather(*self.server_tasks, self.ob_task, return_exceptions=True), 2)

        for p in self.patchers:
            p.stop()

    async def _setup_fully_connected_system(self):
        """Helper to bypass Zeroconf discovery time and directly connect to all components."""
        await self.start_observer()
        await asyncio.sleep(0.1)

        self.ob.config.anchor_type = common.AnchorType.PILOT

        # Force addresses for gripper
        self.ob.config.gripper.service_name = "123.cranebot-gripper-arpeggio-service.test"
        self.ob.config.gripper.address = '127.0.0.1'
        self.ob.config.gripper.port = 8764

        # Force addresses for anchors
        for i in range(4):
            if len(self.ob.config.anchors) <= i:
                break
            self.ob.config.anchors[i].service_name = f"123.cranebot-anchor-service.test_{i}"
            self.ob.config.anchors[i].address = '127.0.0.1'
            self.ob.config.anchors[i].port = i + 8765

        # Wait for connections to establish
        await self.ob.gripper_client_connected.wait()
        await self.ob.any_anchor_connected.wait()
        
        # Give a moment for the keep_robot_connected loop to pick up all 5 clients
        timeout = 5.0
        while len([b for b in self.ob.bot_clients.values() if b.connected]) < 5 and timeout > 0:
            await asyncio.sleep(0.1)
            timeout -= 0.1

        # Connect UI websocket
        async def listenTelemetry():
            async with websockets.connect("ws://127.0.0.1:4249") as ws:
                self.ws_out = ws
                try:
                    async for message in ws:
                        batch = telemetry.TelemetryBatchUpdate().parse(message)
                        for item in batch.updates:
                            self.received_telemetry.append(item)
                except (websockets.ConnectionClosedOK, asyncio.exceptions.CancelledError):
                    return

        self.listen_task = asyncio.create_task(listenTelemetry())
        # wait for connection to register
        while self.ws_out is None:
            await asyncio.sleep(0.01)

    async def _send_control(self, **kwargs):
        """Helper to send a control batch from the simulated UI."""
        batch = control.ControlBatchUpdate(
            robot_id="12345",
            updates=[control.ControlItem(**kwargs)],
        )
        await self.ws_out.send(bytes(batch))
        await asyncio.sleep(0.05) # Yield to observer loop

    def _get_latest_telemetry(self, attr_name):
        """Helper to grab the most recent telemetry item of a specific type."""
        for item in reversed(self.received_telemetry):
            if getattr(item, attr_name, None):
                return getattr(item, attr_name)
        return None

    async def test_connect_gripper_only(self):
        """Connect only gripper, ensures basic connectivity and CombinedMove processing."""
        await self.start_observer()
        await asyncio.sleep(0.1)

        # normally the observer would discover the components with zeroconf. but that is being tested in observer_connection_test
        # in this test, direct inform the observer of the address and port of the gripper server but don't tell it about the anchors
        # it should connect only to the gripper.
        print('setting addresses for local gripper server')
        self.ob.config.gripper.service_name = "123.cranebot-gripper-arpeggio-service.test"
        self.ob.config.gripper.address = '127.0.0.1'
        self.ob.config.gripper.port = 8764

        await self.ob.gripper_client_connected.wait()

        self.assertFalse(self.server_tasks[0].done())
        self.assertEqual(len(self.ob.bot_clients), 1)
        
        self.gripper_connstatus_received = asyncio.Event()
        self.last_commanded_grip_received = asyncio.Event()

        async def listenTelemetry():
            async with websockets.connect("ws://127.0.0.1:4249") as ws:
                self.ws_out = ws
                try:
                    async for message in ws:
                        batch = telemetry.TelemetryBatchUpdate().parse(message)
                        for item in batch.updates:
                            if item.component_conn_status:
                                cs = item.component_conn_status
                                if cs.is_gripper and cs.websocket_status == telemetry.ConnStatus.CONNECTED:
                                    self.gripper_connstatus_received.set()
                            elif item.last_commanded_grip:
                                self.last_commanded_grip_received.set()
                                self.last_commanded_grip = item.last_commanded_grip
                except (websockets.ConnectionClosedOK, asyncio.exceptions.CancelledError):
                    return

        self.listen_task = asyncio.create_task(listenTelemetry())
        await asyncio.wait_for(self.gripper_connstatus_received.wait(), 5)

        await self._send_control(move=control.CombinedMove(wrist_speed=20, finger_speed=21))
        
        # TODO [SERVER VERIFICATION]: Verify that the gripper server correctly parsed the CombinedMove command and updated its internal desired_wrist_speed and desired_finger_speed.

        await asyncio.wait_for(self.last_commanded_grip_received.wait(), 0.4)
        self.assertEqual(self.last_commanded_grip.wrist_speed, 20)
        self.assertEqual(self.last_commanded_grip.finger_speed, 21)

    async def test_gantry_movement_and_jogging(self):
        """Tests direction commands, jog controls, and gantry goal seeking."""
        await self._setup_fully_connected_system()

        # Test Jogging (Anchor)
        await self._send_control(jog_spool=control.JogSpool(is_gripper=False, anchor_num=1, speed=0.15))
        # TODO [SERVER VERIFICATION]: Verify that anchor 1 received the 'aim_speed' command with 0.15.

        # Test Jogging (Gripper)
        await self._send_control(jog_spool=control.JogSpool(is_gripper=True, offset=5.0))
        # TODO [SERVER VERIFICATION]: Verify that the gripper server received the 'jog' command with 5.0 offset.

        # Test Vector Move
        # Send a movement vector (X=1, Y=0, Z=0)
        from nf_robot.generated.nf.common import Vec3
        await self._send_control(move=control.CombinedMove(direction=Vec3(x=1, y=0, z=0), speed=0.2))
        
        self.assertIn('default', self.ob.input_velocities)
        self.assertTrue(np.any(self.ob.input_velocities['default'] != 0))
        # TODO [SERVER VERIFICATION]: Verify that all anchor servers received synchronized 'aim_speed' updates based on inverse kinematics.

        # Test Gantry Goal Positioning
        goal = Vec3(x=0.5, y=0.5, z=1.0)
        await self._send_control(gantry_goal_pos=control.GantryGoalPos(pos=goal))
        
        # The observer should kick off the seek_gantry_goal motion task
        self.assertIsNotNone(self.ob.motion_task)
        self.assertEqual(self.ob.motion_task.get_name(), "seek_gantry_goal")
        
        # Verify Telemetry published the goal marker
        marker_telem = self._get_latest_telemetry('named_position')
        self.assertIsNotNone(marker_telem)
        if marker_telem:
            self.assertEqual(marker_telem.name, 'gantry_goal_marker')

    async def test_system_commands(self):
        """Tests full system actions like Stop, Half Calibration, and Episode Control."""
        await self._setup_fully_connected_system()

        # Start an actual motion task to verify STOP_ALL cancels it correctly
        self.ob.gantry_goal_pos = np.array([5.0, 0.0, 1.0])
        await self.ob.invoke_motion_task(self.ob.seek_gantry_goal())
        self.assertIsNotNone(self.ob.motion_task)
        self.assertFalse(self.ob.motion_task.done())
        copy_of_task = self.ob.motion_task

        # Test STOP_ALL
        print('sending stop all')
        await self._send_control(command=control.CommonCommand(name=control.Command.STOP_ALL))
        print('sent command to stop')
        # Wait for the task to finish its cancellation cleanup
        done, pending = await asyncio.wait([copy_of_task], timeout=16.0)
        self.assertFalse(pending, "Task failed to cancel within 2 seconds")
        self.assertTrue(copy_of_task.done() or copy_of_task.cancelled())
        # TODO [SERVER VERIFICATION]: Verify that all spools on the anchors and gripper received a 'slow_stop_spool' command.

        # Test HALF_CAL
        await self._send_control(command=control.CommonCommand(name=control.Command.HALF_CAL))
        self.assertIsNotNone(self.ob.motion_task)
        self.assertEqual(self.ob.motion_task.get_name(), "half_auto_calibration")

        # Test Episode Control (Lerobot)
        await self._send_control(episode_control=common.EpisodeControl(command=common.EpCommand.START_OR_COMPLETE))
        ep_telem = self._get_latest_telemetry('episode_control')
        self.assertIsNotNone(ep_telem)
        if ep_telem:
            self.assertEqual(ep_telem.command, common.EpCommand.START_OR_COMPLETE)

    async def test_targets(self):
        """Tests adding camera targets, and deleting targets."""
        await self._setup_fully_connected_system()

        # Test Adding a target from a camera click
        # Ensure pe.point_inside_work_area_2d allows the mocked pixel projection
        with patch.object(self.ob.pe, 'point_inside_work_area_2d', return_value=True):
            await self._send_control(
                add_cam_target=control.AddTargetFromAnchorCam(anchor_num=0, img_norm_x=0.5, img_norm_y=0.5)
            )

        # The target queue should have populated with our mocked floor location [1.5, 2.5]
        snapshot = self.ob.target_queue.get_queue_snapshot()
        self.assertGreater(len(snapshot.targets), 0)
        target_id = snapshot.targets[0].id

        # Test Deleting the target
        await self._send_control(delete_target=control.DeleteTarget(target_id=target_id))
        
        snapshot_after = self.ob.target_queue.get_queue_snapshot()
        self.assertEqual(len(snapshot_after.targets), 0)

    async def test_single_component(self):
        """Tests targeted single component actions, swing cancellation"""
        await self._setup_fully_connected_system()

        # Test Single Component Action
        await self._send_control(
            single_component_action=control.SingleComponentAction(
                is_gripper=False, anchor_num=2, action=control.ComponentAction.IDENTIFY
            )
        )
        # TODO [SERVER VERIFICATION]: Verify that anchor 2 specifically received an {'identify': None} command payload.

        # Test Swing Cancellation Toggle
        await self._send_control(set_swing_cancellation=control.SetSwingCancellation(enabled=True, present='.'))
        self.assertIsNotNone(self.ob.swing_cancellation_task)
        self.assertFalse(self.ob.swing_cancellation_task.done())
        
        swing_telem = self._get_latest_telemetry('swing_cancellation_state')
        self.assertIsNotNone(swing_telem)
        if swing_telem:
            self.assertTrue(swing_telem.enabled)
            
        await self._send_control(set_swing_cancellation=control.SetSwingCancellation(enabled=False, present='.'))
        self.assertTrue(self.ob.swing_cancellation_task.done() or self.ob.swing_cancellation_task.cancelled())