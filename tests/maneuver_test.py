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


class TestBuiltins(ManeuverTestCase):
    async def test_each_behavior_answers_where_it_did_before(self):
        ob = self.make_observer()
        owner = lambda cmd: ob._maneuver_commands[cmd][3].name
        self.assertEqual(owner(control.Command.PICK_AND_DROP), 'pick_and_place')
        self.assertEqual(owner(control.Command.SUBMIT_TARGETS_TO_DATASET), 'pick_and_place')
        self.assertEqual(owner(control.Command.HORIZONTAL_CHECK), 'diagnostics')
        verbs = {word: entry[3].name for word, entry in ob._maneuver_verbs.items()}
        self.assertEqual(verbs, {
            'droppoint': 'drop_point', 'fingerplates': 'plates', 'floorplates': 'plates',
            'objectplates': 'plates', 'linear': 'diagnostics', 'goalseek': 'diagnostics',
            'ferry': 'ferry',
            'cluster': 'cluster_sort',
            'basketdata': 'basket_episodes', 'findbasket': 'find_basket',
            'fixdrops': 'pick_and_place',
        })
        controls = {field: handler.__self__.name for field, handler in ob._maneuver_controls.items()}
        self.assertEqual(controls, {
            'episode_control': 'lerobot', 'manage_lerobot_session': 'lerobot',
            'add_cam_target': 'pick_and_place', 'add_room_target': 'pick_and_place',
            'delete_target': 'pick_and_place', 'move_gripper_to': 'pick_and_place',
            'set_target_model': 'pick_and_place',
        })

    async def test_lerobot_grasp_is_an_option_of_the_lerobot_maneuver(self):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0, lerobot_grasp=True)
        lerobot = ob.maneuver('lerobot')
        self.assertTrue(lerobot.use_for_grasp)
        lerobot.grasp = AsyncMock(return_value=True)
        self.assertTrue(await ob.grasp())

    async def test_stop_all_abandons_lerobot_episodes(self):
        ob = self.make_observer()
        await ob.stop_all()
        commands = [kw['episode_control'].command for kw in self.ui if 'episode_control' in kw]
        self.assertEqual(commands, [common.EpCommand.ABANDON])

    async def test_an_inbound_lerobot_status_is_remembered_for_later_peers(self):
        ob = self.make_observer()
        lerobot = ob.maneuver('lerobot')
        lerobot.process_task = asyncio.create_task(asyncio.sleep(10))
        status = common.LerobotSessionStatus(status=common.LerobotStatus.RECORDING)
        with patch.object(ob, 'flush_tele_buffer', new=AsyncMock()):
            await ob._dispatch_update(control.ControlItem(
                episode_control=common.EpisodeControl(status=status)))
        self.assertTrue(lerobot.session_status_event.is_set())
        self.ui.clear()
        lerobot.send_setup_telemetry()
        self.assertEqual(self.ui[0]['episode_control'].status, status)
        lerobot.process_task.cancel()

    async def test_the_target_list_is_replayed_to_a_new_ui(self):
        ob = self.make_observer()
        pnp = ob.maneuver('pick_and_place')
        pnp.target_queue.add_user_target((1.0, 1.0), dropoff='hamper')
        pnp.send_tq_to_ui()
        self.ui.clear()
        pnp.send_setup_telemetry()
        self.assertTrue(any('target_list' in kw for kw in self.ui))
        self.assertTrue(any('auto_targeting_state' in kw for kw in self.ui))

    async def test_targets_on_the_route_destination_are_left_alone(self):
        ob = self.make_observer()
        ob.set_named_position('hamper', np.array([1.0, 1.0, 0.0]))
        ob.set_route(destination=common.RoutePoint.HAMPER)
        pnp = ob.maneuver('pick_and_place')
        kept = pnp._reject_targets_at_dropoff([
            {'position': np.array([1.02, 1.0, 0.0])},
            {'position': np.array([2.0, 1.0, 0.0])},
        ])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]['position'][0], 2.0)


class TestRoute(ManeuverTestCase):
    async def test_set_route_saves_and_shows_it(self):
        ob = self.make_observer()
        with patch('nf_robot.host.observer.save_config') as save:
            ob.set_route(source=common.RoutePoint.ALL_TARGETS, destination=common.RoutePoint.TRASH)
        save.assert_called_once()
        self.assertEqual(ob.route(), (common.RoutePoint.ALL_TARGETS, common.RoutePoint.TRASH))
        self.assertEqual(ob.config.last_route_destination, common.RoutePoint.TRASH)
        status = [kw['task_status'] for kw in self.ui if 'task_status' in kw][-1]
        self.assertEqual(status.route_destination, common.RoutePoint.TRASH)

    async def test_the_ui_setting_the_route_saves_it(self):
        ob = self.make_observer()
        ob.flush_tele_buffer = AsyncMock()
        with patch('nf_robot.host.observer.save_config') as save:
            await ob._handle_set_point(control.SetPoint(route_destination=common.RoutePoint.TOYBOX))
        save.assert_called_once()
        self.assertEqual(ob.config.last_route_destination, common.RoutePoint.TOYBOX)

    async def test_the_origin_has_a_position_and_an_unseen_tag_does_not(self):
        ob = self.make_observer()
        np.testing.assert_array_equal(ob.route_point_position(common.RoutePoint.ORIGIN), np.zeros(3))
        self.assertIsNone(ob.route_point_position(common.RoutePoint.GAMEPAD))


class TestSeekGoal(ManeuverTestCase):
    def flying(self):
        ob = self.make_observer()
        ob.move_direction_speed = AsyncMock()
        ob.pe.gant_pos = np.array([0.0, 0.0, 1.0])
        return ob

    async def test_a_second_goal_steers_the_same_flight(self):
        ob = self.flying()
        self.assertFalse(await ob.seek_goal(np.array([2.0, 0.0, 1.0]), timeout=0.15))
        flight = ob._seek_task
        self.assertFalse(flight.done())
        self.assertFalse(await ob.seek_goal(np.array([0.0, 2.0, 1.0]), timeout=0.15))
        self.assertIs(ob._seek_task, flight)
        np.testing.assert_array_equal(ob._goal_pos, [0.0, 2.0, 1.0])
        ob.pe.gant_pos = np.array([0.0, 2.0, 1.0])
        self.assertTrue(await ob.seek_goal(np.array([0.0, 2.0, 1.0])))
        self.assertTrue(flight.done())

    async def test_clearing_the_goal_ends_the_flight_without_arriving(self):
        ob = self.flying()
        self.assertFalse(await ob.seek_goal(np.array([2.0, 0.0, 1.0]), timeout=0.05))
        await ob.clear_goal()
        self.assertFalse(await ob._seek_task)

    async def test_cancelling_the_caller_lands_the_flight(self):
        ob = self.flying()
        caller = asyncio.create_task(ob.seek_goal(np.array([2.0, 0.0, 1.0])))
        await asyncio.sleep(0.15)
        flight = ob._seek_task
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertTrue(flight.done())

    async def test_a_new_motion_task_ends_a_flight_nobody_is_waiting_on(self):
        ob = self.flying()
        await ob.seek_goal(np.array([2.0, 0.0, 1.0]), timeout=0.05)
        flight = ob._seek_task
        await ob.invoke_motion_task(asyncio.sleep(0))
        self.assertTrue(flight.done())
        await ob.motion_task


class TestSettingsAndHooks(ManeuverTestCase):
    async def test_settings_survive_a_restart(self):
        import tempfile
        from pathlib import Path
        from nf_robot.common.config_loader import load_config
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'configuration.json'
            ob = AsyncObserver(terminate_with_ui=False, config_path=path, port=0)
            rec = ob.add_maneuver(Recorder)
            rec.save_settings_json({'pots': 3})
            reloaded = load_config(path)
        self.assertEqual(reloaded.maneuver_settings['recorder'], '{"pots": 3}')
        # and a maneuver that never stored anything reads None
        self.assertIsNone(self.make_observer().add_maneuver(Recorder).settings_json)

    async def test_component_connections_reach_every_maneuver(self):
        ob = self.make_observer()
        seen = []

        class Listener(Maneuver):
            name = 'listener'

            def on_component_connected(self, kind, anchor_num=None):
                seen.append(('up', kind, anchor_num))

            def on_component_disconnected(self, kind, anchor_num=None):
                seen.append(('down', kind, anchor_num))

        ob.add_maneuver(Listener)
        ob._announce_component('anchor', 1, True)
        ob._announce_component('gripper', None, False)
        self.assertEqual(seen, [('up', 'anchor', 1), ('down', 'gripper', None)])

    async def test_a_clear_item_image_is_kept_only_in_range(self):
        ob = self.make_observer()
        gripper = type('Gripper', (), {})()
        gripper.last_output_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        gripper.last_output_frame[..., 2] = 255      # blue; decoded frames are RGB
        gripper.last_frame_cap_time = 123.0
        ob.gripper_client = gripper
        ob.pe.gant_pos = np.array([1.0, 2.0, 0.5])
        ranges = iter([1.0, 0.2])
        ob.laser_range = lambda: next(ranges, 0.2)
        sleeps = 0

        async def fake_sleep(_):
            nonlocal sleeps
            sleeps += 1
            if sleeps > 2:
                ob.run_command_loop = False

        with patch('asyncio.sleep', new=fake_sleep):
            await ob._watch_for_clear_item()
        seen = ob.last_clear_item_image()
        self.assertEqual(seen.laser_range, 0.2)
        self.assertEqual(seen.timestamp, 123.0)
        self.assertEqual(seen.image_rgb[0, 0, 2], 255)   # kept as RGB


class FakeTilt:
    def __init__(self):
        self.tilt = None

    def pole_tilt(self, max_age=1.0):
        return self.tilt


class TestTiltWatch(unittest.TestCase):
    def test_a_lean_trips_only_once_confirmed(self):
        ob = FakeTilt()
        watch = TiltWatch(ob, tilt_deg=8.0, confirm_s=0.3)
        self.assertIsNone(watch.check())
        ob.tilt = 12.0
        self.assertIsNone(watch.check())     # starts the confirmation window
        self.assertIsNone(watch.check())     # still inside it
        watch.leaning_since -= 1.0           # as though the lean had held for a second
        self.assertIn('12', watch.check())
        self.assertEqual(watch.worst_tilt, 12.0)

    def test_a_stale_reading_never_trips(self):
        watch = TiltWatch(FakeTilt(), confirm_s=0.0)
        for _ in range(3):
            self.assertIsNone(watch.check())


if __name__ == '__main__':
    unittest.main()
