"""Progress reporting for operations whose real progress cannot be observed.

Some operations are one opaque blocking call - loading a model checkpoint, for
instance - so there is no honest percentage to report, only a decent guess at how
long it usually takes. FakeProgress ramps a bar over that guess so the UI has
something to show, and never claims completion until the operation actually returns.
"""

import asyncio
import contextlib
import logging
import time

from nf_robot.generated.nf import telemetry

logger = logging.getLogger(__name__)


class FakeProgress:
    """An operation_progress bar driven by a clock instead of by real progress.

    Sends 0% on entry, then ramps linearly toward `expected_s`, holding at
    `max_percent` if the operation outlives the estimate, and sends 100% on exit
    however the block ends. Used as an async context manager:

        async with FakeProgress(self.send_ui, "Target Model", "Loading checkpoint..."):
            model = await asyncio.to_thread(load_sync)

    The UI treats 100% as "operation over" and by default pops up name + action, so
    `done_action` is what the user reads when it finishes. Pass
    `suppress_completion_popup=True` for work the user did not explicitly ask for, and
    the bar just disappears instead.
    """

    def __init__(self, send_ui, name, current_action='', done_action='Done',
                 failed_action='Failed', expected_s=5.0, interval_s=0.2, max_percent=99.0,
                 suppress_completion_popup=False):
        self.send_ui = send_ui
        self.name = name
        self.current_action = current_action
        self.done_action = done_action
        self.failed_action = failed_action
        self.expected_s = expected_s
        self.interval_s = interval_s
        self.max_percent = max_percent
        self.suppress_completion_popup = suppress_completion_popup
        self._failed = False
        self._task = None

    def _send(self, percent, action):
        self.send_ui(operation_progress=telemetry.OperationProgress(
            percent_complete=percent, name=self.name, current_action=action,
            suppress_completion_popup=self.suppress_completion_popup))

    def set_action(self, action):
        """Change the sub-caption the ticker reports from here on."""
        self.current_action = action

    def fail(self, action=None):
        """Report failure on exit even though nothing was raised.

        For operations that handle their own errors and return normally, which would
        otherwise leave the bar claiming the thing it just failed at is done.
        """
        self._failed = True
        if action is not None:
            self.failed_action = action

    async def _tick(self):
        started = time.time()
        while True:
            await asyncio.sleep(self.interval_s)
            elapsed = time.time() - started
            percent = min(self.max_percent, 100.0 * elapsed / self.expected_s)
            self._send(percent, self.current_action)

    async def __aenter__(self):
        self._send(0.0, self.current_action)
        self._task = asyncio.create_task(self._tick())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        # 100% either way: a bar left part-filled forever is worse than a wrong caption.
        ok = exc_type is None and not self._failed
        self._send(100.0, self.done_action if ok else self.failed_action)
        return False
