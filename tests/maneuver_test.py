import asyncio
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from nf_robot.generated.nf import common, control
from nf_robot.host.maneuver import (Maneuver, OverTension, SafetyPolicy, TiltWatch, command,
                                    control_item, startup_step, verb)
from nf_robot.host.observer import AsyncObserver, DEFAULT_STARTUP_SEQUENCE


class Recorder(Maneuver):
    """A maneuver that answers to one of everything and writes down what reached it."""
    name = 'recorder'

    def __init__(self, ob):
        super().__init__(ob)
        self.calls = []
        self.owner_seen = None
        self.over_tension = []

    @command(control.Command.ZERO_WINCH, motion=True)
    async def go(self):
        self.calls.append('go')
        self.owner_seen = self.ob._motion_owner

    @verb('hello')
    async def hello(self, *words):
        self.calls.append(('hello', words))

    @control_item('scale_room')
    def scaled(self, item):
        self.calls.append(('scale_room', item.scale))

    @startup_step('first')
    async def first(self):
        self.calls.append('first')
        self.owner_seen = self.ob._motion_owner

    @startup_step('second', safety=SafetyPolicy(on_over_tension=OverTension.IGNORE))
    async def second(self):
        self.calls.append(('second', self.ob._motion_safety.on_over_tension))

    def on_stop_all(self):
        self.calls.append('stop_all')

    def on_over_tension(self, tensions):
        self.over_tension.append(tensions)


class ManeuverTestCase(unittest.IsolatedAsyncioTestCase):
    def make_observer(self):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        self.ui = []
        ob.send_ui = lambda **kwargs: self.ui.append(kwargs)
        return ob


class TestRegistration(ManeuverTestCase):
    async def test_parking_is_built_in(self):
        ob = self.make_observer()
        parking = ob.maneuver('parking')
        for cmd in (control.Command.PARK, control.Command.UNPARK, control.Command.RECORD_PARK):
            self.assertIs(ob._maneuver_commands[cmd][3], parking)
        self.assertIn('park', ob._startup_steps)
        self.assertIn('unpark', ob._startup_steps)

    async def test_a_second_maneuver_of_the_same_name_is_refused(self):
        ob = self.make_observer()
        ob.add_maneuver(Recorder)
        with self.assertRaises(ValueError):
            ob.add_maneuver(Recorder)

    async def test_a_command_already_answered_is_refused(self):
        class ClaimsPark(Maneuver):
            name = 'claims_park'

            @command(control.Command.PARK)
            async def park(self):
                pass

        class ClaimsStop(Maneuver):
            name = 'claims_stop'

            @command(control.Command.STOP_ALL)
            async def stop_everything(self):
                pass

        ob = self.make_observer()
        for cls in (ClaimsPark, ClaimsStop):
            with self.assertRaises(ValueError):
                ob.add_maneuver(cls)
        self.assertNotIn('claims_park', ob.maneuvers)

    async def test_a_builtin_verb_is_refused(self):
        class ClaimsSpincal(Maneuver):
            name = 'claims_spincal'

            @verb('spincal')
            async def spin(self):
                pass

        with self.assertRaises(ValueError):
            self.make_observer().add_maneuver(ClaimsSpincal)


class TestDispatch(ManeuverTestCase):
    async def test_a_motion_command_runs_as_the_motion_task_owned_by_its_maneuver(self):
        ob = self.make_observer()
        rec = ob.add_maneuver(Recorder)
        await ob._handle_common_command(control.Command.ZERO_WINCH)
        self.assertEqual(ob.motion_task.get_name(), 'go')
        await ob.motion_task
        self.assertEqual(rec.calls, ['go'])
        self.assertIs(rec.owner_seen, rec)

    async def test_a_verb_gets_the_rest_of_the_words(self):
        ob = self.make_observer()
        rec = ob.add_maneuver(Recorder)
        await ob._handle_debug_command(control.Debug(action='hello big world'))
        self.assertEqual(rec.calls, [('hello', ('big', 'world'))])

    async def test_an_unclaimed_control_item_reaches_its_maneuver(self):
        # scale_room is no longer handled by the observer itself
        ob = self.make_observer()
        rec = ob.add_maneuver(Recorder)
        await ob._dispatch_update(control.ControlItem(scale_room=control.ScaleRoom(scale=2.0)))
        self.assertEqual(rec.calls, [('scale_room', 2.0)])

    async def test_stop_all_tells_every_maneuver(self):
        ob = self.make_observer()
        rec = ob.add_maneuver(Recorder)
        await ob.stop_all()
        self.assertEqual(rec.calls, ['stop_all'])

    async def test_an_inbound_move_may_not_use_a_reserved_key(self):
        ob = self.make_observer()
        ob.move_direction_speed = AsyncMock()
        for key in ('maneuver:parking', 'ob:default'):
            await ob._handle_movement(control.CombinedMove(
                direction=common.Vec3(x=1.0, y=0.0, z=0.0), speed=0.1, source_key=key))
        ob.move_direction_speed.assert_not_awaited()


class TestStartupSequence(ManeuverTestCase):
    async def test_the_default_is_what_auto_start_always_did(self):
        ob = self.make_observer()
        self.assertEqual(ob.startup_sequence_names, list(DEFAULT_STARTUP_SEQUENCE))
        ob._check_startup_sequence()

    async def test_steps_run_in_the_order_given_each_under_its_own_policy(self):
        ob = self.make_observer()
        rec = ob.add_maneuver(Recorder)
        ob.set_startup_sequence(['second', 'first'])
        await ob.startup_sequence()
        self.assertEqual(rec.calls, [('second', OverTension.IGNORE), 'first'])
        self.assertIs(rec.owner_seen, rec)
        # nothing is left attributed to the maneuver once the sequence is over
        self.assertIsNone(ob._motion_owner)
        self.assertEqual(ob._motion_safety, SafetyPolicy())

    async def test_an_unknown_step_is_caught_before_main_runs(self):
        ob = self.make_observer()
        ob.set_startup_sequence(['unpark', 'no_such_step'])
        with self.assertRaises(ValueError):
            ob._check_startup_sequence()

    async def test_parking_steps_decide_for_themselves(self):
        ob = self.make_observer()
        parking = ob.maneuver('parking')
        parking.park = AsyncMock()
        parking.unpark = AsyncMock()

        parking.data.parked = False
        parking.data.pos = None
        await parking.unpark_if_parked()
        await parking.park_if_recorded()
        parking.unpark.assert_not_awaited()
        parking.park.assert_not_awaited()

        parking.data.parked = True
        parking.data.pos = common.Vec3(x=1.0, y=2.0, z=2.5)
        await parking.unpark_if_parked()
        await parking.park_if_recorded()
        parking.unpark.assert_awaited_once()
        parking.park.assert_awaited_once()


class TestOverTension(ManeuverTestCase):
    async def running(self, ob, policy):
        rec = ob.add_maneuver(Recorder)
        ob.motion_task = asyncio.create_task(asyncio.sleep(10))
        ob._motion_owner = rec
        ob._motion_safety = SafetyPolicy(on_over_tension=policy)
        return rec

    async def asyncTearDown(self):
        # the sleeping tasks started by running()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    async def test_abort_cancels_and_says_why(self):
        ob = self.make_observer()
        rec = await self.running(ob, OverTension.ABORT)
        ob._respond_to_over_tension(np.full(4, 20.0))
        await asyncio.sleep(0)
        self.assertTrue(ob.motion_task.cancelled())
        self.assertEqual(rec.abort_reason, 'tension')

    async def test_notify_leaves_it_running_and_tells_it(self):
        ob = self.make_observer()
        rec = await self.running(ob, OverTension.NOTIFY)
        ob._respond_to_over_tension(np.full(4, 20.0))
        await asyncio.sleep(0)
        self.assertFalse(ob.motion_task.done())
        self.assertEqual(len(rec.over_tension), 1)
        self.assertTrue(ob.tension_over_limit)

    async def test_ignore_leaves_it_running(self):
        ob = self.make_observer()
        rec = await self.running(ob, OverTension.IGNORE)
        ob._respond_to_over_tension(np.full(4, 20.0))
        await asyncio.sleep(0)
        self.assertFalse(ob.motion_task.done())
        self.assertEqual(rec.over_tension, [])
        self.assertIsNone(rec.abort_reason)

    async def test_torque_is_shed_whatever_the_policy(self):
        ob = self.make_observer()
        await self.running(ob, OverTension.IGNORE)
        ob.pe.tension = np.full(4, 1000.0)
        torque = []

        async def set_torque(enabled):
            torque.append(enabled)
            ob.run_command_loop = False

        ob.set_torque = set_torque
        with patch('asyncio.sleep', new=AsyncMock()):
            await ob.passive_safety()
        self.assertEqual(torque[0], False)
        self.assertFalse(ob.motion_task.done())

    async def test_override_is_undone_when_the_block_exits(self):
        ob = self.make_observer()
        with ob.override_safety(on_over_tension=OverTension.NOTIFY):
            self.assertEqual(ob._motion_safety.on_over_tension, OverTension.NOTIFY)
        self.assertEqual(ob._motion_safety, SafetyPolicy())


class TestManeuverHelpers(ManeuverTestCase):
    async def test_velocity_keys_are_prefixed(self):
        rec = self.make_observer().add_maneuver(Recorder)
        self.assertEqual(rec.velocity_key(), 'maneuver:recorder')
        self.assertEqual(rec.velocity_key('lift'), 'maneuver:recorder:lift')

    async def test_spawned_tasks_end_with_the_maneuver(self):
        ob = self.make_observer()
        rec = ob.add_maneuver(Recorder)
        task = rec.spawn(asyncio.sleep(10))
        await ob._stop_maneuvers()
        self.assertTrue(task.cancelled())

    async def test_parking_location_is_replayed_to_a_new_ui(self):
        ob = self.make_observer()
        parking = ob.maneuver('parking')
        parking.data.pos = common.Vec3(x=1.0, y=2.0, z=2.5)
        parking.send_setup_telemetry()
        names = [kw['named_position'].name for kw in self.ui if 'named_position' in kw]
        self.assertEqual(names, ['parking_location'])

    async def test_set_parked_writes_the_owned_field(self):
        ob = self.make_observer()
        parking = ob.maneuver('parking')
        parking.set_parked(True)
        self.assertTrue(ob.config.park_data.parked)


class FakeTilt:
    def __init__(self):
        self.tilt = None

    def pole_tilt(self, max_age=1.0):
        return self.tilt


class TestTiltWatch(unittest.TestCase):
    def test_a_lean_trips_only_once_confirmed(self):
        ob = FakeTilt()
        watch = TiltWatch(ob, tilt_deg=8.0, confirm_s=0.0)
        self.assertIsNone(watch.check())
        ob.tilt = 12.0
        self.assertIsNone(watch.check())     # starts the confirmation window
        self.assertIn('12', watch.check())
        self.assertEqual(watch.worst_tilt, 12.0)

    def test_a_stale_reading_never_trips(self):
        watch = TiltWatch(FakeTilt(), confirm_s=0.0)
        for _ in range(3):
            self.assertIsNone(watch.check())


if __name__ == '__main__':
    unittest.main()
