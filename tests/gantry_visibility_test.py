"""
Unit tests for AsyncObserver.monitor_gantry_visibility, the background watch on how the anchor
cameras see the gantry marker.

The loop is driven directly, bound to a stub carrying only the attributes it touches, so the
tests can feed a datastore poll by poll without standing up an observer or any hardware. Its
sleep is replaced by one that advances a fake clock, so an outage costs nothing however long
the loop's timeout is set.

Nothing here hard codes that timeout, or the loop's cadence or warmup: outages are asked for in
seconds either side of UNSEEN_LIMIT_S, so changing the constant changes what the tests wait for.
"""

import asyncio
import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

from nf_robot.host import observer as observer_module
from nf_robot.host.data_store import DataStore
from nf_robot.host.observer import (AsyncObserver, UNSEEN_LIMIT_S, VISIBILITY_POLL_S,
                                    _widest_gap)

T0 = 10_000.0  # any fixed start; the loop only ever measures differences
# Stamp for a sighting that has to look newer than every other one a test files. The loop only
# asks whether one camera's newest sighting moved on, never how it compares to the host clock.
LATE_SIGHTING_T = T0 + 1e6


class FakeClock:
    """Stands in for the time module inside observer, which only calls time.time()."""

    def __init__(self):
        self.now = T0

    def time(self):
        return self.now


class StubObserver:
    """Only what monitor_gantry_visibility and _report_gantry_marker_fault read."""

    def __init__(self, motion_task=None):
        self.datastore = DataStore()
        self.anchors = {0: Mock(anchor_num=0), 1: Mock(anchor_num=1)}
        self.run_command_loop = True
        self.any_anchor_connected = asyncio.Event()
        self.any_anchor_connected.set()
        self.motion_task = motion_task
        self.gantry_marker_fault = None
        self._gantry_marker_warned = set()
        self._gantry_marker_popped = set()
        self.popups = []

    def send_ui(self, **kwargs):
        if 'pop_message' in kwargs:
            self.popups.append(kwargs['pop_message'].message)

    def sight(self, t, anchor_num, position):
        """File one anchor camera's detection of the gantry marker, as its client does."""
        self.datastore.gantry_pos.insert(np.concatenate([[t], [anchor_num], position]))

    monitor_gantry_visibility = AsyncObserver.monitor_gantry_visibility
    _report_gantry_marker_fault = AsyncObserver._report_gantry_marker_fault


async def run_monitor(stub, polls):
    """Run the monitor for len(polls) turns, calling polls[i](i) before the i'th one.

    Each poll callback takes the turn number and may file sightings. Every sleep the loop takes
    advances the clock by what it asked for, so the fake clock stays in step with the loop's own
    cadence whatever that is set to. The loop's one-off warmup sleep is not a turn: it costs the
    clock its delay and nothing else, so the length of the warmup cannot shift what a test feeds.
    """
    clock = FakeClock()
    turns = iter(range(len(polls)))
    warmed_up = False

    async def fake_sleep(delay):
        nonlocal warmed_up
        clock.now += delay
        if not warmed_up:
            warmed_up = True
            return
        try:
            i = next(turns)
        except StopIteration:
            stub.run_command_loop = False
            return
        polls[i](i)

    with patch.object(observer_module, 'time', clock), \
            patch.object(asyncio, 'sleep', fake_sleep):
        await stub.monitor_gantry_visibility()


def quiet(seconds):
    """Polls covering `seconds` of the loop's own time, in which no camera reports anything."""
    return [lambda i: None] * max(0, math.ceil(seconds / VISIBILITY_POLL_S))


# The loop starts counting the outage from the poll that takes in the sighting already filed, and
# runs one last turn after the polls are exhausted, so these clear the limit either way by a turn
# to spare - whatever UNSEEN_LIMIT_S is.
PAST_THE_LIMIT_S = UNSEEN_LIMIT_S + 3 * VISIBILITY_POLL_S
SHORT_OF_THE_LIMIT_S = UNSEEN_LIMIT_S - 2 * VISIBILITY_POLL_S


class TestWidestGap(unittest.TestCase):

    def test_gap_is_the_largest_pairwise_distance(self):
        self.assertAlmostEqual(_widest_gap([[0, 0, 0], [0.1, 0, 0], [3.0, 0, 0]]), 3.0)


class TestGantryVisibility(unittest.IsolatedAsyncioTestCase):

    async def test_marker_seen_normally_raises_nothing(self):
        stub = StubObserver()

        def poll(i):
            for anchor_num in (0, 1):
                stub.sight(T0 + i, anchor_num, [0.1 * i, 0, 1.0])

        await run_monitor(stub, [poll] * 6)
        self.assertEqual(stub.popups, [])

    async def test_a_fast_traverse_is_not_a_duplicate(self):
        """Sightings from different frames are never compared to each other, so covering
        several meters inside one history window is not a second tag."""
        stub = StubObserver()

        def poll(i):
            stub.sight(T0 + i, 0, [0.5 * i, 0, 1.0])  # 0.5 m/s across the whole window

        await run_monitor(stub, [poll] * 9)
        self.assertEqual(stub.popups, [])

    async def test_marker_in_two_places_in_one_frame_is_reported(self):
        stub = StubObserver()

        def poll(i):
            stub.sight(T0 + i, 0, [0, 0, 1.0])
            if i in (1, 3):  # a mirror, caught by the once-a-second full frame scan
                stub.sight(T0 + i, 0, [0, 3.0, 1.0])

        await run_monitor(stub, [poll] * 6)
        self.assertEqual(len(stub.popups), 1)
        self.assertIn('appears to anchor 0 in multiple places', stub.popups[0])

    async def test_one_split_frame_is_not_enough(self):
        stub = StubObserver()

        def poll(i):
            stub.sight(T0 + i, 0, [0, 0, 1.0])
            if i == 2:
                stub.sight(T0 + i, 0, [0, 3.0, 1.0])

        await run_monitor(stub, [poll] * 6)
        self.assertEqual(stub.popups, [])

    async def test_two_detections_of_the_same_tag_are_not_a_duplicate(self):
        """Overlapping crops can detect one tag twice in a frame; those land together."""
        stub = StubObserver()

        def poll(i):
            stub.sight(T0 + i, 0, [0, 0, 1.0])
            stub.sight(T0 + i, 0, [0.02, 0.01, 1.0])

        await run_monitor(stub, [poll] * 6)
        self.assertEqual(stub.popups, [])

    async def test_unseen_marker_is_reported_after_the_timeout(self):
        stub = StubObserver()
        stub.sight(T0, 0, [0, 0, 1.0])
        await run_monitor(stub, quiet(PAST_THE_LIMIT_S))
        self.assertEqual(len(stub.popups), 1)
        self.assertIn(f"hasn't been detected in {UNSEEN_LIMIT_S:.0f} seconds", stub.popups[0])

    async def test_unseen_marker_is_not_reported_before_the_timeout(self):
        stub = StubObserver()
        stub.sight(T0, 0, [0, 0, 1.0])
        await run_monitor(stub, quiet(SHORT_OF_THE_LIMIT_S))
        self.assertEqual(stub.popups, [])

    async def test_unseen_marker_is_reported_only_once_per_outage(self):
        stub = StubObserver()
        stub.sight(T0, 0, [0, 0, 1.0])
        await run_monitor(stub, quiet(3 * UNSEEN_LIMIT_S))
        self.assertEqual(len(stub.popups), 1)

    async def test_a_second_outage_is_not_popped_again(self):
        stub = StubObserver()
        stub.sight(T0, 0, [0, 0, 1.0])
        seen_again = [lambda i: stub.sight(LATE_SIGHTING_T, 0, [0, 0, 1.0])]
        polls = quiet(PAST_THE_LIMIT_S) + seen_again + quiet(PAST_THE_LIMIT_S)
        await run_monitor(stub, polls)
        self.assertEqual(len(stub.popups), 1)

    async def test_recovery_rearms_the_calibration_abort(self):
        task = Mock()
        task.done.return_value = False
        task.get_name.return_value = 'full_auto_calibration'
        stub = StubObserver(motion_task=task)
        stub.sight(T0, 0, [0, 0, 1.0])
        seen_again = [lambda i: stub.sight(LATE_SIGHTING_T, 0, [0, 0, 1.0])]
        polls = quiet(PAST_THE_LIMIT_S) + seen_again + quiet(PAST_THE_LIMIT_S)
        await run_monitor(stub, polls)
        # the popup is once a session, the abort is not: the second outage would otherwise let
        # a re-run of the calibration fit a room out of stale sightings
        self.assertEqual(task.cancel.call_count, 2)

    async def test_no_anchors_connected_is_not_blamed_on_the_marker(self):
        stub = StubObserver()
        stub.anchors = {}
        await run_monitor(stub, quiet(2 * UNSEEN_LIMIT_S))
        self.assertEqual(stub.popups, [])

    async def test_fault_during_calibration_aborts_it(self):
        task = Mock()
        task.done.return_value = False
        task.get_name.return_value = 'full_auto_calibration'
        stub = StubObserver(motion_task=task)
        stub.sight(T0, 0, [0, 0, 1.0])
        await run_monitor(stub, quiet(PAST_THE_LIMIT_S))
        task.cancel.assert_called_once()
        self.assertIn('has not been seen', stub.gantry_marker_fault)

    async def test_fault_outside_calibration_does_not_cancel_the_motion_task(self):
        task = Mock()
        task.done.return_value = False
        task.get_name.return_value = 'pick_and_place_loop'
        stub = StubObserver(motion_task=task)
        stub.sight(T0, 0, [0, 0, 1.0])
        await run_monitor(stub, quiet(PAST_THE_LIMIT_S))
        task.cancel.assert_not_called()
        self.assertIsNone(stub.gantry_marker_fault)
        self.assertEqual(len(stub.popups), 1)  # still told about it


if __name__ == '__main__':
    unittest.main()
