import asyncio
import json
import logging
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip('mcp')

from mcp import Client

from nf_robot.generated.nf import telemetry
from nf_robot.host.component_client import ComponentClient
from nf_robot.host.maneuver import Maneuver, verb
from nf_robot.host.mcp_bridge.bridge import McpBridge, contact_sheet, crop_frame
from nf_robot.host.observer import AsyncObserver

HELLO_V1 = '''
import logging
from nf_robot.host.maneuver import Maneuver, verb

logger = logging.getLogger(__name__)

class Hello(Maneuver):
    name = 'hello'

    @verb('hello')
    async def hello(self, *words):
        """Say hello, version one."""
        logger.info('hello v1 %s', words)
'''

HELLO_V2 = HELLO_V1.replace('version one', 'version two').replace('hello v1', 'hello v2')

BROKEN_AT_RUN = '''
from nf_robot.host.maneuver import Maneuver, verb

class Broken(Maneuver):
    name = 'broken'

    @verb('broken', motion=True)
    async def go(self):
        x = 1
        raise RuntimeError('the agent wrote a bug')
'''


class Starts(Maneuver):
    name = 'starts'

    def __init__(self, ob):
        super().__init__(ob)
        self.started = False

    async def start(self):
        self.started = True

    @verb('starts')
    async def act(self):
        pass


class FailsToStart(Maneuver):
    name = 'fails_to_start'

    async def start(self):
        raise RuntimeError('no')

    @verb('fails')
    async def act(self):
        pass


class FakeCamera:
    capture_raw_frame = ComponentClient.capture_raw_frame

    def __init__(self):
        self.frame_lock = threading.Lock()
        self.frame = np.full((240, 320, 3), 128, np.uint8)
        self.last_frame_cap_time = time.time()


class BridgeTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        self.bridge = self.ob.add_maneuver(McpBridge, allow_code=True)
        logging.getLogger().addHandler(self.bridge._log_handler)
        logging.getLogger('agent_maneuvers').setLevel(logging.INFO)
        logging.getLogger('agent_scripts').setLevel(logging.INFO)

    async def asyncTearDown(self):
        logging.getLogger().removeHandler(self.bridge._log_handler)
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()

    async def call_raw(self, tool, **args):
        """A tool's CallToolResult, through a client as an agent would see it."""
        async with Client(self.bridge.mcp) as client:
            return await client.call_tool(tool, args)

    async def call(self, tool, **args):
        """A tool's result: the structured result if it has one, else its content blocks."""
        result = await self.call_raw(tool, **args)
        if result.is_error:
            raise AssertionError(result.content[0].text)
        if result.structured_content is not None:
            return result.structured_content.get('result', result.structured_content)
        blocks = []
        for block in result.content:
            if block.type == 'text':
                try:
                    blocks.append(json.loads(block.text))
                except ValueError:
                    blocks.append(block.text)
            else:
                blocks.append(block)
        return blocks[0] if len(blocks) == 1 else blocks


class TestLiveRegistration(BridgeTestCase):
    async def test_a_running_maneuver_is_started(self):
        m = await self.ob.add_running_maneuver(Starts)
        self.assertTrue(m.started)

    async def test_a_maneuver_that_fails_to_start_is_not_left_registered(self):
        with self.assertRaises(RuntimeError):
            await self.ob.add_running_maneuver(FailsToStart)
        self.assertNotIn('fails_to_start', self.ob.maneuvers)
        self.assertNotIn('fails', self.ob._maneuver_verbs)

    async def test_removal_frees_the_name_and_everything_it_answered_to(self):
        await self.ob.add_running_maneuver(Starts)
        await self.ob.remove_maneuver('starts')
        self.assertNotIn('starts', self.ob._maneuver_verbs)
        await self.ob.add_running_maneuver(Starts)

    async def test_a_builtin_cannot_be_removed(self):
        with self.assertRaises(ValueError):
            await self.ob.remove_maneuver('parking')

    async def test_removal_cancels_a_motion_task_it_owns(self):
        m = await self.ob.add_running_maneuver(Starts)
        await self.ob.invoke_motion_task(asyncio.sleep(10), owner=m)
        await asyncio.sleep(0)
        await self.ob.remove_maneuver('starts')
        await asyncio.sleep(0)
        self.assertEqual(self.ob.last_motion_result().reason, 'removed')

    async def test_entries_name_their_owner_and_doc(self):
        entries = self.ob.maneuver_entries()
        ferry = next(e for e in entries if e['kind'] == 'verb' and e['key'] == 'ferry')
        self.assertEqual(ferry['owner'], 'ferry')
        self.assertIn('Carry', ferry['doc'])
        park = next(e for e in entries if e['kind'] == 'command' and e['key'] == 'PARK')
        self.assertEqual(park['owner'], 'parking')


class TestTelemetryListener(BridgeTestCase):
    async def test_popups_become_events_and_pending_popups(self):
        self.ob.pending_popup_acks[7] = asyncio.get_running_loop().create_future()
        self.ob.telemetry.send(pop_message=telemetry.Popup(message='Proceed?', id=7, buttons=['Yes', 'No']))
        state = await self.call('get_state')
        self.assertEqual(state['pending_popups'], [{'id': 7, 'message': 'Proceed?', 'buttons': ['Yes', 'No']}])
        events = (await self.call('get_events'))['events']
        self.assertEqual(events[-1]['kind'], 'popup')

    async def test_named_positions_are_tracked(self):
        from nf_robot.generated.nf import common
        self.ob.telemetry.send(named_position=telemetry.NamedObjectPosition(
            name='hamper', position=common.Vec3(x=1.0, y=2.0, z=0.0)))
        self.assertEqual((await self.call('get_state'))['named_positions'], {'hamper': [1.0, 2.0, 0.0]})


class TestAgentCode(BridgeTestCase):
    async def test_a_loaded_verb_runs_and_its_info_logs_are_events(self):
        loaded = await self.call('load_maneuver', module_name='hello', source=HELLO_V1)
        self.assertTrue(loaded['loaded'], loaded)
        self.assertEqual(loaded['maneuvers'], ['hello'])
        await self.call('run_verb', text='hello big world')
        logs = [e for e in (await self.call('get_events'))['events'] if e['kind'] == 'log']
        self.assertIn("hello v1 ('big', 'world')", logs[-1]['message'])

    async def test_reloading_replaces_the_old_version(self):
        await self.call('load_maneuver', module_name='hello', source=HELLO_V1)
        await self.call('load_maneuver', module_name='hello', source=HELLO_V2)
        doc = next(e['doc'] for e in self.ob.maneuver_entries() if e['key'] == 'hello')
        self.assertEqual(doc, 'Say hello, version two.')

    async def test_a_maneuver_reloads_from_its_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'hello.py'
            path.write_text(HELLO_V1)
            loaded = await self.call('load_maneuver', path=str(path))
            self.assertTrue(loaded['loaded'], loaded)
            path.write_text(HELLO_V2)
            await self.call('load_maneuver', path=str(path))
        self.assertIn('hello', self.bridge.agent_code.modules)
        doc = next(e['doc'] for e in self.ob.maneuver_entries() if e['key'] == 'hello')
        self.assertEqual(doc, 'Say hello, version two.')

    async def test_load_wants_exactly_one_of_source_and_path(self):
        result = await self.call_raw('load_maneuver', module_name='hello', source=HELLO_V1,
                                     path='/nowhere.py')
        self.assertTrue(result.is_error)
        self.assertIn('exactly one', result.content[0].text)

    async def test_source_that_does_not_import_leaves_the_old_version_running(self):
        await self.call('load_maneuver', module_name='hello', source=HELLO_V1)
        result = await self.call('load_maneuver', module_name='hello', source='def (:')
        self.assertFalse(result['loaded'])
        self.assertIn('SyntaxError', result['error'])
        self.assertIn('hello', self.ob.maneuvers)

    async def test_a_failing_motion_verb_reports_the_agents_own_line(self):
        await self.call('load_maneuver', module_name='broken', source=BROKEN_AT_RUN)
        await self.call('run_verb', text='broken')
        report = await self.call('wait', until='motion_done', timeout=5)
        result = report['motion_results'][-1]
        self.assertEqual(result['outcome'], 'failed')
        self.assertIn('the agent wrote a bug', result['message'])
        self.assertIn("raise RuntimeError('the agent wrote a bug')", result['error'])

    async def test_a_script_returns_its_value(self):
        result = await self.call('run_script', source='async def run(ob, m):\n    return 6 * 7\n',
                                 motion=False)
        self.assertEqual(result, {'finished': True, 'value': 42})

    async def test_a_motion_script_ends_in_a_motion_result(self):
        started = await self.call('run_script', source='async def run(ob, m):\n    return "done"\n')
        self.assertEqual(started['motion_task'], 'script')
        report = await self.call('wait', until='motion_done', timeout=5)
        self.assertEqual(report['motion_results'][-1]['value'], 'done')

    async def test_code_tools_are_refused_unless_allowed(self):
        self.bridge.allow_code = False
        result = await self.call_raw('run_script', source='async def run(ob, m): pass')
        self.assertTrue(result.is_error)
        self.assertIn('--mcp_allow_code', result.content[0].text)

    async def test_a_spawned_task_that_fails_is_reported(self):
        await self.call('load_maneuver', module_name='spawner', source='''
from nf_robot.host.maneuver import Maneuver, verb

class Spawner(Maneuver):
    name = 'spawner'

    @verb('spawn')
    async def go(self):
        async def fail():
            raise ValueError('background bug')
        self.spawn(fail())
''')
        await self.call('run_verb', text='spawn')
        await asyncio.sleep(0.01)
        logs = [e for e in (await self.call('get_events'))['events'] if e['kind'] == 'log']
        self.assertIn('background bug', logs[-1]['message'])


class TestMotionTools(BridgeTestCase):
    async def test_a_refusal_comes_back_at_once(self):
        with patch.object(self.ob, '_gantry_parked', return_value=True):
            result = await self.call('nudge', dx=0.0, dy=0.0, dz=0.05)
        self.assertFalse(result['started'])
        self.assertEqual(result['result']['reason'], 'parked')

    async def test_stop_reports_the_task_it_stopped(self):
        await self.ob.invoke_motion_task(asyncio.sleep(10))
        await asyncio.sleep(0)
        result = await self.call('stop')
        self.assertEqual(result['last_motion_result']['reason'], 'stopped')

    async def test_wait_for_motion_returns_at_once_when_nothing_runs(self):
        report = await self.call('wait', until='motion_done', timeout=30)
        self.assertTrue(report['satisfied'])
        self.assertLess(report['waited_s'], 1)


class TestImages(BridgeTestCase):
    async def test_an_anchor_frame_comes_back_as_a_jpeg(self):
        self.ob.anchors[0] = FakeCamera()
        caption, img = await self.call('get_image', camera='anchor0', max_px=128)
        self.assertIn('anchor0 320x240', caption)
        self.assertEqual(img.mime_type, 'image/jpeg')

    async def test_a_missing_camera_is_an_error_that_says_where_to_look(self):
        result = await self.call_raw('get_image', camera='anchor2')
        self.assertTrue(result.is_error)
        self.assertIn('anchor_cameras', result.content[0].text)

    async def test_several_cameras_come_back_in_one_call_each_cropped(self):
        self.ob.anchors[0] = FakeCamera()
        self.ob.anchors[1] = FakeCamera()
        blocks = await self.call('get_images', views=[
            dict(camera='anchor0'),
            dict(camera='anchor1', crop=[0.25, 0.25, 0.5, 0.5], max_px=320)])
        captions = [b for b in blocks if isinstance(b, str)]
        images = [b for b in blocks if not isinstance(b, str)]
        self.assertEqual(len(images), 2)
        self.assertIn('anchor0 320x240', captions[0])
        self.assertIn('(80x60 px)', captions[1])

    async def test_a_missing_camera_is_reported_without_failing_the_others(self):
        self.ob.anchors[0] = FakeCamera()
        blocks = await self.call('get_images', views=[dict(camera='anchor2'), dict(camera='anchor0')])
        self.assertIn('anchor2: RuntimeError', blocks[0])
        self.assertEqual(sum(not isinstance(b, str) for b in blocks), 1)

    def test_a_crop_takes_fractions_of_the_frame(self):
        frame = np.arange(100 * 200).reshape(100, 200)
        part = crop_frame(frame, [0.5, 0.0, 1.0, 0.5])
        self.assertEqual(part.shape, (50, 100))
        self.assertEqual(part[0, 0], frame[0, 100])
        with self.assertRaises(ValueError):
            crop_frame(frame, [0.5, 0.5, 0.5, 1.0])

    def test_contact_sheet_lays_frames_in_a_grid(self):
        frames = [np.zeros((100, 200, 3), np.uint8)] * 5
        sheet = contact_sheet(frames, [str(i) for i in range(5)], 100)
        self.assertEqual(sheet.shape, (100, 300, 3))


class TestHttp(BridgeTestCase):
    async def test_a_client_can_list_and_call_tools_over_http(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            self.bridge.port = s.getsockname()[1]
        await self.bridge._start_server()
        try:
            async with Client(f'http://127.0.0.1:{self.bridge.port}/mcp') as client:
                tools = {t.name for t in (await client.list_tools()).tools}
                self.assertTrue({'guide', 'get_state', 'wait', 'load_maneuver'} <= tools)
                result = await client.call_tool('guide', {})
                self.assertIn('Stringman', result.content[0].text)
        finally:
            await self.bridge.stop()


class TestGuide(BridgeTestCase):
    async def test_the_guides_example_maneuver_loads(self):
        from nf_robot.host.mcp_bridge.bridge import GUIDE_PATH
        guide = GUIDE_PATH.read_text()
        example = next(block.split('\n', 1)[1] for block in guide.split('```python')[1:]
                       if 'class Hover' in block).split('```')[0]
        loaded = await self.call('load_maneuver', module_name='hover', source=example)
        self.assertTrue(loaded['loaded'], loaded)
        self.assertEqual([e['key'] for e in loaded['entries']], ['hover'])
