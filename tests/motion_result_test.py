import asyncio
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from nf_robot.host.component_client import ComponentClient
from nf_robot.host.maneuver import Maneuver, OverTension, SafetyPolicy
from nf_robot.host.observer import AsyncObserver


class Listener(Maneuver):
    """Writes down every result and safety event it is told about."""
    name = 'listener'

    def __init__(self, ob):
        super().__init__(ob)
        self.results = []
        self.safety_events = []

    def on_motion_task_done(self, result):
        self.results.append(result)

    def on_safety_event(self, kind, message):
        self.safety_events.append((kind, message))


async def returns_five():
    return 5


async def raises():
    raise ValueError('the arm is on backwards')


async def sleeps():
    await asyncio.sleep(10)


async def other_sleeps():
    await asyncio.sleep(10)


class MotionResultTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        self.ob.send_ui = lambda **kwargs: None
        self.listener = self.ob.add_maneuver(Listener)

    async def asyncTearDown(self):
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    async def ended(self, task):
        try:
            await task
        except BaseException:
            pass
        await asyncio.sleep(0)
        return self.ob.last_motion_result()


class TestOutcomes(MotionResultTestCase):
    async def test_a_finished_task_reports_its_value_and_owner(self):
        task = await self.ob.invoke_motion_task(returns_five(), owner=self.listener)
        result = await self.ended(task)
        self.assertEqual((result.name, result.owner, result.outcome), ('returns_five', 'listener', 'finished'))
        self.assertEqual(result.value, 5)
        self.assertIsNone(result.reason)
        self.assertIn('Returns five finished', result.message)
        self.assertEqual(self.listener.results, [result])

    async def test_a_failure_carries_the_exception_in_words_and_as_a_traceback(self):
        task = await self.ob.invoke_motion_task(raises())
        result = await self.ended(task)
        self.assertEqual(result.outcome, 'failed')
        self.assertIn('ValueError: the arm is on backwards', result.message)
        self.assertIn('Traceback', result.error)

    async def test_a_stop_says_it_was_stopped(self):
        task = await self.ob.invoke_motion_task(sleeps())
        await asyncio.sleep(0)
        await self.ob.stop_all()
        result = await self.ended(task)
        self.assertEqual((result.outcome, result.reason), ('cancelled', 'stopped'))
        self.assertFalse(result.safety_trip)

    async def test_a_replaced_task_names_what_replaced_it(self):
        first = await self.ob.invoke_motion_task(sleeps())
        await asyncio.sleep(0)
        await self.ob.invoke_motion_task(other_sleeps())
        first_result = self.listener.results[-1]
        self.assertEqual(first_result.name, 'sleeps')
        self.assertEqual(first_result.reason, 'replaced')
        self.assertIn('Other sleeps started in its place', first_result.message)

    async def test_a_refusal_while_parked_is_reported_without_running(self):
        with patch.object(self.ob, '_gantry_parked', return_value=True):
            task = await self.ob.invoke_motion_task(sleeps(), owner=self.listener)
        self.assertIsNone(task)
        result = self.ob.last_motion_result()
        self.assertEqual((result.outcome, result.reason, result.owner), ('refused', 'parked', 'listener'))
        self.assertIn('Unpark first', result.message)

    async def test_the_running_task_can_be_seen(self):
        self.assertIsNone(self.ob.current_motion_task())
        await self.ob.invoke_motion_task(sleeps(), owner=self.listener)
        running = self.ob.current_motion_task()
        self.assertEqual((running.name, running.owner), ('sleeps', 'listener'))


class TestSafetyTrips(MotionResultTestCase):
    async def test_an_over_tension_abort_is_a_safety_trip_with_the_line_named(self):
        task = await self.ob.invoke_motion_task(sleeps(), owner=self.listener)
        await asyncio.sleep(0)
        tensions = np.array([1.0, 1.0, 99.0, 1.0])
        self.ob._respond_to_over_tension(tensions)
        result = await self.ended(task)
        self.assertEqual((result.outcome, result.reason), ('cancelled', 'tension'))
        self.assertTrue(result.safety_trip)
        self.assertIn('line 2 reached 99.0 N', result.message)
        self.assertEqual(self.listener.abort_reason, 'tension')
        self.assertEqual([kind for kind, _ in self.listener.safety_events], ['over_tension'])

    async def test_a_later_stop_does_not_hide_the_trip(self):
        task = await self.ob.invoke_motion_task(sleeps())
        await asyncio.sleep(0)
        self.ob._respond_to_over_tension(np.full(4, 99.0))
        await self.ob.stop_all()
        result = await self.ended(task)
        self.assertEqual(result.reason, 'tension')

    async def test_a_trip_under_notify_is_an_event_but_the_task_runs_on(self):
        await self.ob.invoke_motion_task(sleeps(), owner=self.listener,
                                         safety=SafetyPolicy(on_over_tension=OverTension.NOTIFY))
        self.ob._respond_to_over_tension(np.full(4, 99.0))
        await asyncio.sleep(0)
        self.assertIsNotNone(self.ob.current_motion_task())
        self.assertEqual(self.listener.results, [])
        self.assertEqual(len(self.listener.safety_events), 1)

    async def test_an_over_tension_with_nothing_running_is_still_an_event(self):
        self.ob._respond_to_over_tension(np.full(4, 99.0))
        self.assertEqual(len(self.listener.safety_events), 1)

    async def test_a_marker_fault_aborts_a_task_that_needs_the_marker(self):
        task = await self.ob.invoke_motion_task(
            sleeps(), safety=SafetyPolicy(needs_gantry_marker=True))
        await asyncio.sleep(0)
        self.ob._report_gantry_marker_fault(
            'unseen', 'the gripper marker has not been seen for 10 seconds', 'Popup text', 'detail')
        result = await self.ended(task)
        self.assertEqual(result.reason, 'marker')
        self.assertIn('has not been seen for 10 seconds', result.message)
        self.assertEqual(self.listener.safety_events, [('marker', 'Popup text')])


class FakeCamera:
    capture_raw_frame = ComponentClient.capture_raw_frame

    def __init__(self, frame=None, timestamp=None):
        self.frame_lock = threading.Lock()
        self.frame = frame
        self.last_frame_cap_time = timestamp


class TestAnchorFrames(MotionResultTestCase):
    async def test_the_newest_frame_is_a_copy(self):
        frame = np.zeros((4, 6, 3), np.uint8)
        self.ob.anchors[1] = FakeCamera(frame, time.time())
        self.ob.anchors[3] = FakeCamera()
        self.assertEqual(self.ob.anchor_cameras(), [1])
        timestamp, got = await self.ob.anchor_capture(1, after=0)
        self.assertEqual(got.shape, (4, 6, 3))
        self.assertIsNot(got, frame)

    async def test_a_stale_or_missing_camera_gives_nothing(self):
        self.ob.anchors[1] = FakeCamera(np.zeros((4, 6, 3), np.uint8), time.time() - 5)
        self.assertEqual(await self.ob.anchor_capture(1, timeout=0.05), (None, None))
        self.assertEqual(await self.ob.anchor_capture(2, after=0), (None, None))
