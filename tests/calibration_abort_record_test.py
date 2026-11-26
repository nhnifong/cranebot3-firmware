"""
Unit tests for the abort record calibration leaves in calibration_diagnostics_<timestamp>.pkl.

The recorded passes say what the optimizer was given; they cannot say that the run never
reached the next one, or why. These cover the step being captured from the progress messages
the operator sees, and the record that captures it landing in the pickle.

Every test runs in a temp directory: _flush_calibration_diagnostics writes to a relative path,
and the repo root may hold real calibration_diagnostics_*.pkl files.
"""

import asyncio
import glob
import os
import pickle
import tempfile
import unittest
from unittest.mock import Mock

from nf_robot.generated.nf import telemetry
from nf_robot.host.observer import AsyncObserver


class StubObserver:
    """Only what send_ui and the diagnostics recorders read."""

    def __init__(self, rec_diagnostics=True):
        self.rec_diagnostics = rec_diagnostics
        self._calibration_diagnostics = []
        self._calibration_diagnostics_path = None
        self._calibration_step = (0.0, None)
        self.last_ep_ctrl_status = None
        self.telemetry = Mock()

    send_ui = AsyncObserver.send_ui
    _flush_calibration_diagnostics = AsyncObserver._flush_calibration_diagnostics
    _record_calibration_abort = AsyncObserver._record_calibration_abort

    def progress(self, percent, action, name='Calibration'):
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=percent, name=name, current_action=action))


class TestCalibrationAbortRecord(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)
        self.stub = StubObserver()

    def written(self):
        paths = glob.glob('calibration_diagnostics_*.pkl')
        self.assertEqual(len(paths), 1, paths)
        with open(paths[0], 'rb') as fh:
            return pickle.load(fh)

    def test_step_comes_from_the_progress_message_on_screen(self):
        self.stub.progress(6.0, 'Observe diamond right')
        self.stub._record_calibration_abort('Aborted: line tension exceeded the safe limit')
        record = self.written()[-1]
        self.assertEqual(record['abort'], 'Aborted: line tension exceeded the safe limit')
        self.assertEqual(record['step'], 'Observe diamond right')
        self.assertEqual(record['percent_complete'], 6.0)
        self.assertIsNone(record['error'])

    def test_the_latest_step_wins(self):
        self.stub.progress(6.0, 'Observe diamond right')
        self.stub.progress(12.0, 'Observe diamond top')
        self.stub._record_calibration_abort('Cancelled by user')
        self.assertEqual(self.written()[-1]['step'], 'Observe diamond top')

    def test_other_operations_do_not_move_the_step(self):
        self.stub.progress(6.0, 'Observe diamond right')
        self.stub.progress(50.0, 'Flashing', name='Update Component Firmware')
        self.stub._record_calibration_abort('Cancelled by user')
        self.assertEqual(self.written()[-1]['step'], 'Observe diamond right')

    def test_an_empty_action_does_not_erase_the_step(self):
        self.stub.progress(6.0, 'Observe diamond right')
        self.stub.progress(6.0, '')
        self.stub._record_calibration_abort('Cancelled by user')
        self.assertEqual(self.written()[-1]['step'], 'Observe diamond right')

    def test_the_record_is_appended_after_the_passes(self):
        self.stub._calibration_diagnostics.append({'pass': 'anchors_pass1', 'args': {}})
        self.stub.progress(22.0, 'Running 2nd optimization pass')
        self.stub._record_calibration_abort('Cancelled by user')
        records = self.written()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]['pass'], 'anchors_pass1')
        self.assertNotIn('pass', records[1])  # readers tell them apart by which key is there
        self.assertNotIn('abort', records[0])

    def test_a_failure_carries_its_traceback(self):
        self.stub.progress(0.0, 'Observing markers')
        self.stub._record_calibration_abort('Failed: RuntimeError()', error='Traceback...')
        self.assertEqual(self.written()[-1]['error'], 'Traceback...')

    def test_nothing_is_written_without_rec_diagnostics(self):
        stub = StubObserver(rec_diagnostics=False)
        stub.progress(6.0, 'Observe diamond right')
        stub._record_calibration_abort('Cancelled by user')
        self.assertEqual(stub._calibration_diagnostics, [])
        self.assertFalse(os.path.exists('calibration_diagnostics.pkl'))


class TestCancelAndWait(unittest.TestCase):
    """The abort handler cancels the diamond's finger calibration task and waits for it. Awaiting
    a cancelled task raises CancelledError, which used to end the handler before it recorded
    the abort, sent the abort message, or cleared the fault flags."""

    def test_returns_normally_for_a_task_it_cancels(self):
        async def run():
            task = asyncio.create_task(asyncio.sleep(10))
            await asyncio.sleep(0)
            await AsyncObserver._cancel_and_wait(task)
            return task.cancelled()
        self.assertTrue(asyncio.run(run()))

    def test_an_abort_handler_reaches_the_record(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(tmp.name)
        self.addCleanup(os.chdir, cwd)
        stub = StubObserver()
        stub.progress(3.0, 'Observe diamond bottom')

        async def calibration():
            finger_task = asyncio.create_task(asyncio.sleep(10))
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await AsyncObserver._cancel_and_wait(finger_task)
                stub._record_calibration_abort('Aborted: line tension exceeded the safe limit')
                raise

        async def run():
            task = asyncio.create_task(calibration())
            await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertEqual(stub._calibration_diagnostics[-1]['step'], 'Observe diamond bottom')


if __name__ == '__main__':
    unittest.main()
