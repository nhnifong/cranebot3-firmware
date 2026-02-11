import asyncio
import time
import logging

logger = logging.getLogger(__name__)

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

    def start(self):
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._monitor())

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _monitor(self):
        logger.info("Starting Event Loop Monitor...")
        while self.running:
            # We want to sleep for 'interval', but we measure how long it *actually* takes.
            start_time = time.perf_counter()
            await asyncio.sleep(self.interval)
            end_time = time.perf_counter()
            
            actual_duration = end_time - start_time
            lag = actual_duration - self.interval

            # If the lag is significant, we are being starved.
            if lag > self.threshold:
                saturation_pct = (lag / actual_duration) * 100
                logger.warning(
                    f"⚠️ EVENT LOOP BLOCKED! "
                    f"Expected sleep: {self.interval}s, "
                    f"Actual: {actual_duration:.4f}s, "
                    f"Lag: {lag:.4f}s "
                    f"({saturation_pct:.1f}% blocked)"
                )