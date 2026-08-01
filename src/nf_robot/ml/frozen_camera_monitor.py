"""Detect a frozen camera feed live, during a recording session.

A camera can stall while the rest of the session looks healthy: the stream keeps
delivering frames at full rate, but every frame is the same image. That silently
ruins every episode recorded afterwards (naavox/move_clutter episodes 289-325 and
448-481, and justink04/laundry-in-hamper2-8-1-26 episodes 41-60, are all whole
runs of episodes with a dead gripper camera that nobody noticed until training
data was being assembled). The operator can only fix it by restarting the
session, so they need to hear about it while it is still cheap.

The signal is that consecutive frames are byte-identical. Live video never is:
sensor noise moves pixels even with a motionless camera and a motionless scene.
The one caveat is that these cameras run below the recording fps, so the same
image legitimately arrives a few times in a row - hence the several-second
threshold rather than "any repeat".

Cost is a strided subsample plus an array compare per frame, so this can sit
directly in the video decode loop. See lerobot_find_frozen_video.py for the
offline version that audits an already-recorded dataset.
"""

import time

import numpy as np

# Seconds of unchanging video before a feed is called frozen. Healthy feeds in
# recorded datasets never repeat an image for more than ~1.8 s (usually < 0.7 s),
# while real stalls last for whole episodes, so anything in between works.
FROZEN_SECONDS = 3.0

# Re-warn this often while a feed stays frozen, so the alert doesn't scroll away.
REPEAT_ALERT_SECONDS = 10.0

# Compare every Nth pixel. Frozen frames are identical everywhere, so a subsample
# is just as decisive and ~64x cheaper. Keep it a strided view rather than a
# resize: averaging would smooth away exactly the sensor noise this relies on.
_STRIDE = 8


def fingerprint(frame: np.ndarray) -> np.ndarray:
    """Cheap content fingerprint of a frame: a strided subsample of its pixels."""
    return np.ascontiguousarray(frame[::_STRIDE, ::_STRIDE])


class FrozenCameraMonitor:
    """Tracks, per camera feed, how long its image has been unchanging.

    note_frame() is called from each feed's decode thread; frozen_feeds() is
    polled from the recording loop. Feeds register themselves on their first
    frame, so a feed that has not connected yet is not reported as frozen.
    """

    def __init__(self, frozen_seconds: float = FROZEN_SECONDS, names: dict | None = None):
        self.frozen_seconds = frozen_seconds
        self.names = names or {}
        self._last_fingerprint: dict = {}
        self._last_change_t: dict = {}
        self._alerted_at: dict = {}

    def name(self, feed) -> str:
        return self.names.get(feed, f"feed {feed}")

    def note_frame(self, feed, frame: np.ndarray) -> None:
        """Record that `frame` arrived on `feed`. Called once per decoded frame."""
        fp = fingerprint(frame)
        prev = self._last_fingerprint.get(feed)
        if prev is None or prev.shape != fp.shape or not np.array_equal(fp, prev):
            self._last_change_t[feed] = time.monotonic()
        self._last_fingerprint[feed] = fp

    def forget(self, feed) -> None:
        """Drop a feed's state, e.g. when its stream is torn down and reconnected."""
        self._last_fingerprint.pop(feed, None)
        self._last_change_t.pop(feed, None)
        self._alerted_at.pop(feed, None)

    def frozen_feeds(self) -> dict:
        """Feeds whose image has not changed for longer than the threshold.

        Maps feed -> seconds frozen. A feed whose stream has died outright is
        also caught: no frames arrive, so its last change only gets older.
        """
        now = time.monotonic()
        return {feed: now - t for feed, t in self._last_change_t.items()
                if now - t >= self.frozen_seconds}

    def new_alerts(self) -> list:
        """Frozen feeds worth telling the operator about right now.

        Returns [(feed, seconds_frozen), ...] for feeds that just froze or that
        have stayed frozen since the last alert, so polling this every tick
        produces one message per feed per REPEAT_ALERT_SECONDS rather than one
        per tick. Recovery clears a feed's alert state.
        """
        now = time.monotonic()
        frozen = self.frozen_feeds()
        for feed in list(self._alerted_at):
            if feed not in frozen:
                del self._alerted_at[feed]

        alerts = []
        for feed, seconds in sorted(frozen.items(), key=lambda kv: str(kv[0])):
            last = self._alerted_at.get(feed)
            if last is None or now - last >= REPEAT_ALERT_SECONDS:
                self._alerted_at[feed] = now
                alerts.append((feed, seconds))
        return alerts

    def describe(self, alerts: list) -> str:
        """One-line operator-facing message for the output of new_alerts().

        Deliberately identical every time the same feeds are frozen - no elapsed
        seconds - because the UI pops a dialog on each *changed* error string,
        and a repeating alert must not keep reopening it. How long it has been
        frozen goes in the log line instead.
        """
        return ("FROZEN CAMERA: " + ", ".join(self.name(feed) for feed, _ in alerts) +
                " - the image has stopped changing. Episodes recorded now are unusable; "
                "restart the session.")
