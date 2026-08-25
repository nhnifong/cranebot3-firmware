"""
Unit tests for AsyncObserver.prefer_swing_cancellation, the shared way for an operation to
ask for swing cancellation without deciding for itself whether this robot's is trustworthy.

The context manager is bound to a stub carrying only what it reads, so the tests cover the
decision and the restore without a gripper, a config file, or a running event loop's worth of
the real cancellation task.
"""

import unittest
from unittest.mock import Mock

from nf_robot.generated.nf import config as nf_config
from nf_robot.host.observer import AsyncObserver, with_swing_cancellation_preferred


class StubObserver:
    """Records what prefer_swing_cancellation asked for instead of running the real task."""

    def __init__(self, verified=True, gripper=True, already_running=False):
        self.config = nf_config.StringmanPilotConfig(swing_cancellation_verified=verified)
        self.gripper_client = Mock() if gripper else None
        self.running = already_running
        self.calls = []

    def set_swing_cancellation(self, enabled):
        was_running = self.running
        self.running = enabled
        self.calls.append(enabled)
        return was_running

    prefer_swing_cancellation = AsyncObserver.prefer_swing_cancellation

    @with_swing_cancellation_preferred
    async def a_long_task(self, value):
        self.running_inside = self.running
        return value * 2


class TestPreferSwingCancellation(unittest.IsolatedAsyncioTestCase):

    async def test_verified_robot_gets_it_on_inside_the_block(self):
        stub = StubObserver()
        async with stub.prefer_swing_cancellation():
            self.assertTrue(stub.running)
        self.assertFalse(stub.running)
        self.assertEqual(stub.calls, [True, False])

    async def test_unverified_robot_is_left_alone(self):
        stub = StubObserver(verified=False)
        async with stub.prefer_swing_cancellation():
            self.assertFalse(stub.running)
        self.assertEqual(stub.calls, [])

    async def test_a_config_predating_the_field_counts_as_unverified(self):
        stub = StubObserver()
        stub.config = nf_config.StringmanPilotConfig().from_json('{"swingLatency": 0.27}')
        async with stub.prefer_swing_cancellation():
            pass
        self.assertEqual(stub.calls, [])

    async def test_no_gripper_means_nothing_to_cancel_from(self):
        stub = StubObserver(gripper=False)
        async with stub.prefer_swing_cancellation():
            self.assertFalse(stub.running)
        self.assertEqual(stub.calls, [])

    async def test_cancellation_already_on_is_left_on(self):
        stub = StubObserver(already_running=True)
        async with stub.prefer_swing_cancellation():
            self.assertTrue(stub.running)
        self.assertTrue(stub.running)

    async def test_the_previous_state_is_restored_after_a_failure(self):
        stub = StubObserver()
        with self.assertRaises(RuntimeError):
            async with stub.prefer_swing_cancellation():
                raise RuntimeError('grasp blew up')
        self.assertFalse(stub.running)

    async def test_an_unverified_robot_keeps_a_manually_enabled_cancellation(self):
        """The preference is not a veto: an operator who switched it on keeps it on."""
        stub = StubObserver(verified=False, already_running=True)
        async with stub.prefer_swing_cancellation():
            self.assertTrue(stub.running)
        self.assertTrue(stub.running)


class TestWithSwingCancellationPreferred(unittest.IsolatedAsyncioTestCase):
    """The whole-method form, used by the long motion tasks."""

    async def test_the_method_runs_under_the_preference(self):
        stub = StubObserver()
        self.assertEqual(await stub.a_long_task(21), 42)
        self.assertTrue(stub.running_inside)
        self.assertFalse(stub.running)

    async def test_an_unverified_robot_is_left_alone(self):
        stub = StubObserver(verified=False)
        await stub.a_long_task(1)
        self.assertFalse(stub.running_inside)
        self.assertEqual(stub.calls, [])

    def test_the_name_survives_for_invoke_motion_task(self):
        """invoke_motion_task names the motion task from the coroutine, and the abort paths
        match on that name."""
        coro = StubObserver().a_long_task(1)
        try:
            self.assertEqual(coro.__name__, 'a_long_task')
        finally:
            coro.close()


if __name__ == '__main__':
    unittest.main()
