import asyncio
import time
import logging

logger = logging.getLogger(__name__)


def _describe_handle(handle):
    """Best-effort human description of what an asyncio Handle is running.

    When the loop resumes a Task, the handle's callback is the Task's __step
    bound method, so callback.__self__ is the Task itself. repr(task) includes
    the coroutine and the exact source line it is currently parked at, which is
    exactly what we want to blame for hogging the loop. For plain callbacks
    (call_soon/call_later) we fall back to the handle's own repr.
    """
    cb = getattr(handle, '_callback', None)
    owner = getattr(cb, '__self__', None)
    if isinstance(owner, asyncio.Task):
        return repr(owner)
    return repr(handle)


class LoopMonitor:
    def __init__(self, interval=0.1, threshold=0.05):
        """
        interval: How often to check the loop (seconds).
        threshold: How much lag is considered "blocking" (seconds).
        """
        self.interval = interval
        self.threshold = threshold
        self.running = False
        self._task = None

        # Slowest single Handle._run() observed since we last reset (per interval).
        self._slowest_duration = 0.0
        self._slowest_desc = None
        self._original_handle_run = None

    def start(self):
        if not self.running:
            self.running = True
            self._patch_handle_run()
            self._task = asyncio.create_task(self._monitor())

    async def stop(self):
        self.running = False
        self._unpatch_handle_run()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def _patch_handle_run(self):
        """Wrap asyncio.Handle._run to time every callback the loop executes."""
        if self._original_handle_run is not None:
            return
        original_run = asyncio.Handle._run
        self._original_handle_run = original_run
        monitor = self

        def timed_run(handle):
            start = time.perf_counter()
            try:
                return original_run(handle)
            finally:
                duration = time.perf_counter() - start
                if duration > monitor._slowest_duration:
                    monitor._slowest_duration = duration
                    monitor._slowest_desc = _describe_handle(handle)

        asyncio.Handle._run = timed_run

    def _unpatch_handle_run(self):
        if self._original_handle_run is not None:
            asyncio.Handle._run = self._original_handle_run
            self._original_handle_run = None

    async def _monitor(self):
        logger.info("Starting Event Loop Monitor...")
        while self.running:
            # Reset the per-interval "slowest task" tracker right before sleeping
            # so what we report is attributable to this interval only.
            self._slowest_duration = 0.0
            self._slowest_desc = None

            # We want to sleep for 'interval', but we measure how long it *actually* takes.
            start_time = time.perf_counter()
            await asyncio.sleep(self.interval)
            end_time = time.perf_counter()

            actual_duration = end_time - start_time
            lag = actual_duration - self.interval

            # If the lag is significant, we are being starved.
            if lag > self.threshold:
                saturation_pct = (lag / actual_duration) * 100
                # Capture the culprit now, before the next reset can clobber it.
                slowest_desc = self._slowest_desc
                slowest_duration = self._slowest_duration
                culprit = (
                    f" Slowest callback this interval: {slowest_duration:.4f}s in {slowest_desc}"
                    if slowest_desc is not None else ""
                )
                logger.warning(
                    f"⚠️ EVENT LOOP BLOCKED! "
                    f"Expected sleep: {self.interval}s, "
                    f"Actual: {actual_duration:.4f}s, "
                    f"Lag: {lag:.4f}s "
                    f"({saturation_pct:.1f}% blocked)."
                    f"{culprit}"
                )
