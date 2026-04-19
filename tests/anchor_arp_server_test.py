"""
Tests for AnchorArpServer.
Mocks all hardware interfaces (DaMiaoController, DamiaoSpoolController) to test
command dispatch, tighten/stow state machines, sensor polling, and shutdown behaviour.

Behaviour already covered by spool_dm_test.py (DamiaoSpoolController internals,
tracking loop math, jog logic, etc.) is not repeated here.
"""
import pytest
pytestmark = pytest.mark.pi
pytest.importorskip("gpiodevice")

import sys
import unittest
from unittest.mock import patch, Mock, MagicMock
import asyncio
import websockets
import json
import time

# damiao_motor is hardware-only; stub the whole module before local imports resolve it.
sys.modules.setdefault('damiao_motor', MagicMock())

from nf_robot.robot.anchor_arp_server import AnchorArpServer


class TestAnchorArpServer(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.run_loop = True

        def mock_tracking_loop():
            while self.run_loop:
                time.sleep(0.05)

        self.patchers = [
            patch('nf_robot.robot.anchor_arp_server.DaMiaoController'),
            patch('nf_robot.robot.anchor_arp_server.get_mac_address', return_value='aa:bb:cc:dd:ee:ff'),
            patch('nf_robot.robot.anchor_arp_server.DamiaoSpoolController'),
            patch('nf_robot.robot.anchor_server.stream_command', ['sleep', 'infinity']),
        ]
        (self.mock_dm_class,
         _,
         self.mock_spool_class,
         _) = [p.start() for p in self.patchers]

        self.mock_controller = self.mock_dm_class.return_value

        # AnchorArpServer creates two DamiaoSpoolController instances; hand back a
        # distinct Mock for each so per-spool assertions are unambiguous.
        self.mock_spools = [Mock(), Mock()]
        self.mock_spool_class.side_effect = self.mock_spools

        for spool in self.mock_spools:
            spool.trackingLoop = mock_tracking_loop
            spool.popMeasurements.return_value = []
            spool.last_tension = 0.0

        self.server = AnchorArpServer(power=False)
        self.server_task = asyncio.create_task(self.server.main())
        await asyncio.sleep(0.1)  # let server start up

    async def asyncTearDown(self):
        self.run_loop = False
        self.server.shutdown()
        await asyncio.wait_for(self.server_task, timeout=2.0)
        for p in self.patchers:
            p.stop()

    # ------------------------------------------------------------------ helpers

    async def send_command(self, command, sleep=0.1):
        """Open a websocket, send one command dict, wait, close."""
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(json.dumps(command))
            await asyncio.sleep(sleep)
            self.assertFalse(self.server_task.done(), "Server crashed after command")

    # ------------------------------------------------------------------ startup

    async def test_startup_creates_two_spools_and_controller(self):
        """DaMiaoController and two DamiaoSpoolController instances are created at boot."""
        self.mock_dm_class.assert_called_once()
        self.assertEqual(self.mock_spool_class.call_count, 2,
                         "Expected exactly two DamiaoSpoolController instances")
        self.assertFalse(self.server_task.done(), "Server should be running")

    async def test_power_flag_changes_first_spool_diameter(self):
        """The power=True flag passes the power-line full diameter to spool 0."""
        import nf_robot.common.definitions as defs

        # Run a second server with power=True on a different port to avoid
        # colliding with the already-bound self.server on 8765.
        loop2_state = {'running': True}
        def tracking_loop2():
            while loop2_state['running']:
                time.sleep(0.05)
        spools2 = [Mock(), Mock()]
        for s in spools2:
            s.trackingLoop = tracking_loop2
            s.popMeasurements.return_value = []

        self.mock_spool_class.reset_mock()
        self.mock_spool_class.side_effect = spools2

        server2 = AnchorArpServer(power=True)
        server2_task = asyncio.create_task(server2.main(port=8766))
        await asyncio.sleep(0.1)

        # Extract the full_diameter keyword from the first call (spool 0 = power spool)
        _, kwargs0 = self.mock_spool_class.call_args_list[0]
        self.assertEqual(kwargs0['full_diameter'], defs.damiao_full_spool_diameter_power_line)

        # Tear down the second server
        loop2_state['running'] = False
        server2.shutdown()
        await asyncio.wait_for(server2_task, timeout=2.0)

        # Restore side_effect for subsequent tests
        self.mock_spool_class.side_effect = self.mock_spools

    # ------------------------------------------------------------------ process_imu / connection

    async def test_process_imu_enables_motors_and_resumes_spools(self):
        """Every motor is enabled and every spool resumes when a client connects."""
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await asyncio.sleep(0.1)
            for motor in self.server.motors:
                motor.enable.assert_called()
            for spool in self.server.spools:
                spool.resumeTrackingLoop.assert_called()

    # ------------------------------------------------------------------ two_reference_lengths

    async def test_two_reference_lengths_sets_both_spools(self):
        """two_reference_lengths dispatches setReferenceLength to each spool."""
        await self.send_command({'two_reference_lengths': [3.5, 4.2]})
        self.mock_spools[0].setReferenceLength.assert_called_once_with(3.5)
        self.mock_spools[1].setReferenceLength.assert_called_once_with(4.2)

    # ------------------------------------------------------------------ aim_speed

    async def test_aim_speed_zero_stops_both_spools(self):
        """aim_speed=0 sends setAimSpeed(0) to both spools."""
        await self.send_command({'aim_speed': 0})
        self.mock_spools[0].setAimSpeed.assert_called_with(0)
        self.mock_spools[1].setAimSpeed.assert_called_with(0)

    async def test_aim_speed_targets_correct_spool(self):
        """aim_speed=(speed, spool_no) routes only to the named spool."""
        await self.send_command({'aim_speed': [0.25, 1]})
        self.mock_spools[1].setAimSpeed.assert_called_with(0.25)
        self.mock_spools[0].setAimSpeed.assert_not_called()

    async def test_aim_speed_invalid_format_no_crash(self):
        """A malformed aim_speed payload does not crash the server."""
        await self.send_command({'aim_speed': 'not_a_tuple'})
        self.assertFalse(self.server_task.done())

    async def test_aim_speed_invalid_spool_number_no_crash(self):
        """aim_speed with spool_no outside [0,1] does not crash the server."""
        await self.send_command({'aim_speed': [0.1, 5]})
        self.assertFalse(self.server_task.done())

    # ------------------------------------------------------------------ jog

    async def test_jog_targets_correct_spool(self):
        """jog=(delta, spool_no) calls jog() on the named spool only."""
        await self.send_command({'jog': [0.3, 0]})
        self.mock_spools[0].jog.assert_called_once_with(0.3)
        self.mock_spools[1].jog.assert_not_called()

    async def test_jog_invalid_format_no_crash(self):
        """A malformed jog payload does not crash the server."""
        await self.send_command({'jog': 'bad'})
        self.assertFalse(self.server_task.done())

    # ------------------------------------------------------------------ readOtherSensors

    async def test_read_other_sensors_populates_update_dict(self):
        """readOtherSensors drains both spools and writes keyed entries to self.update."""
        self.mock_spools[0].popMeasurements.return_value = [(1.0, 2.0, 0.05, 1.5)]
        self.mock_spools[1].popMeasurements.return_value = [(2.0, 3.0, 0.10, 1.4)]

        self.server.readOtherSensors()

        self.assertEqual(self.server.update['spool0'], [(1.0, 2.0, 0.05, 1.5)])
        self.assertEqual(self.server.update['spool1'], [(2.0, 3.0, 0.10, 1.4)])
        self.mock_spools[0].popMeasurements.assert_called_once()
        self.mock_spools[1].popMeasurements.assert_called_once()

    async def test_read_other_sensors_caps_measurements_at_50(self):
        """readOtherSensors silently truncates bursts larger than 50 records."""
        self.mock_spools[0].popMeasurements.return_value = list(range(80))
        self.mock_spools[1].popMeasurements.return_value = []

        self.server.readOtherSensors()

        self.assertEqual(len(self.server.update['spool0']), 50)

    # ------------------------------------------------------------------ tighten

    async def test_tighten_invalid_spool_is_noop(self):
        """tighten() with spool_no outside (0, 1) returns without touching spools."""
        await self.server.tighten(2)
        self.mock_spools[0].setAimSpeed.assert_not_called()
        self.mock_spools[1].setAimSpeed.assert_not_called()

    async def test_tighten_reels_in_until_tight_then_stops(self):
        """
        While the line is slack, tighten reels in at TIGHTENING_SPEED.
        Once tension exceeds the threshold, it stops and the server keeps running.
        """
        spool = self.mock_spools[0]
        spool.last_tension = 0.0  # slack

        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(json.dumps({'tighten': 0}))
            await asyncio.sleep(0.1)

            spool.setAimSpeed.assert_called_with(self.server.conf['TIGHTENING_SPEED'])

            spool.last_tension = 2.0  # tight
            await asyncio.sleep(0.1)

            spool.setAimSpeed.assert_called_with(0)
            self.assertFalse(self.server_task.done())
            # closing websocket cancels the still-running monitoring sub-task
            await asyncio.wait_for(ws.close(), timeout=2.0)

    async def test_tighten_retries_with_reduced_speed_on_slip(self):
        """
        When the line re-loosens during the monitoring window, tighten retries
        using TIGHTENING_SPEED * 0.7.
        """
        spool = self.mock_spools[1]
        initial_speed = self.server.conf['TIGHTENING_SPEED']
        spool.last_tension = 2.0  # tight from the start → slack loop skipped

        tighten_task = asyncio.create_task(self.server.tighten(1))

        # Let monitoring begin, then inject a slip.
        await asyncio.sleep(0.1)
        spool.last_tension = 0.0  # slip detected by monitoring loop

        # Give the monitoring loop one check-interval to detect the slip and
        # let the retry's slack loop issue at least one reduced-speed command.
        await asyncio.sleep(0.2)

        all_speeds = [c.args[0] for c in spool.setAimSpeed.call_args_list]
        expected_retry_speed = initial_speed * 0.7
        self.assertTrue(
            any(abs(s - expected_retry_speed) < 1e-9 for s in all_speeds),
            f"Retry speed {expected_retry_speed:.4f} not found in {all_speeds}",
        )

        tighten_task.cancel()
        try:
            await tighten_task
        except asyncio.CancelledError:
            pass

    async def test_tighten_exhausts_max_retries_and_stops_motor(self):
        """
        After five consecutive slips tighten gives up and leaves the spool stopped.
        time.monotonic is patched to hold the monitoring window open indefinitely so
        only explicit slip injections can terminate each monitoring phase.
        """
        spool = self.mock_spools[0]
        spool.last_tension = 2.0  # tight; slack loop is always skipped

        with patch('nf_robot.robot.anchor_arp_server.time') as mock_time:
            mock_time.monotonic.return_value = 0.0  # end_time = 3 s, never naturally expires
            mock_time.sleep = time.sleep

            tighten_task = asyncio.create_task(self.server.tighten(0))

            for _ in range(5):
                await asyncio.sleep(0.1)   # monitoring loop is running
                spool.last_tension = 0.0   # inject slip → monitoring breaks
                await asyncio.sleep(0.1)   # let the break propagate and retry begin
                spool.last_tension = 2.0   # tight for next attempt's slack loop

            await asyncio.wait_for(tighten_task, timeout=3.0)

        # Last call must be setAimSpeed(0) — the motor is stopped on failure.
        self.assertEqual(spool.setAimSpeed.call_args[0][0], 0)

    # ------------------------------------------------------------------ stow

    async def test_stow_invalid_spool_is_noop(self):
        """stow() with spool_no outside (0, 1) returns without touching spools."""
        await self.server.stow(3)
        self.mock_spools[0].setAimSpeed.assert_not_called()
        self.mock_spools[1].setAimSpeed.assert_not_called()

    async def test_stow_immediately_tight_disables_motor(self):
        """When the line is already tight stow stops the spool and disables the motor."""
        spool = self.mock_spools[0]
        motor = self.server.motors[0]
        spool.last_tension = 2.0  # already tight

        await self.server.stow(0)

        spool.setAimSpeed.assert_called_with(0)
        spool.pauseTrackingLoop.assert_called_once()
        motor.disable.assert_called_once()

    async def test_stow_reels_in_when_slack_then_disables(self):
        """stow reels in while the line is slack, then stops and disables the motor."""
        spool = self.mock_spools[1]
        motor = self.server.motors[1]
        spool.last_tension = 0.0  # slack

        stow_task = asyncio.create_task(self.server.stow(1))
        await asyncio.sleep(0.1)

        spool.setAimSpeed.assert_called_with(self.server.conf['TIGHTENING_SPEED'])

        spool.last_tension = 2.0  # tight
        await asyncio.wait_for(stow_task, timeout=1.0)

        spool.pauseTrackingLoop.assert_called_once()
        motor.disable.assert_called_once()

    async def test_stow_dispatched_from_command(self):
        """The 'stow' key in an incoming message is dispatched to stow()."""
        spool = self.mock_spools[0]
        spool.last_tension = 2.0  # avoid blocking in the stow loop

        await self.send_command({'stow': 0})

        spool.pauseTrackingLoop.assert_called()
        self.server.motors[0].disable.assert_called()

    # ------------------------------------------------------------------ identify

    async def test_identify_pauses_loop_jogs_motor_then_resumes(self):
        """identify() pauses the tracking loop, drives the motor, then resumes it."""
        async with websockets.connect("ws://127.0.0.1:8765") as ws:
            await ws.send(json.dumps({'identify': None}))
            await asyncio.sleep(0.3)  # identify runs ~0.1 s of real time.sleep calls

            # Default spool_no for identify is 1
            spool = self.mock_spools[1]
            motor = self.server.motors[1]

            spool.pauseTrackingLoop.assert_called()
            spool.resumeTrackingLoop.assert_called()
            motor.send_cmd_vel.assert_called()

    # ------------------------------------------------------------------ shutdown

    async def test_shutdown_fast_stops_all_spools_and_closes_controller(self):
        """shutdown() calls fastStop on every spool and shuts down the CAN controller."""
        self.server.shutdown()

        for spool in self.server.spools:
            spool.fastStop.assert_called()
        self.mock_controller.shutdown.assert_called()


if __name__ == '__main__':
    unittest.main()
