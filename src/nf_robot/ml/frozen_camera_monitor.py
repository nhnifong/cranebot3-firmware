"""Detect a camera feed that keeps delivering the same byte-identical frame during recording.

Frozen cameras can silently corrupt teleop data. This detector produces a user visible warning
that the current ep should be abandoned and the dataset closed out.

the robot may even need a hard reboot.
"""

import time

import numpy as np

# Seconds of unchanging video before a feed is called frozen; healthy feeds never repeat for
# more than ~1.8s.
FROZEN_SECONDS = 3.0

# Re-warn this often while a feed stays frozen, so the alert doesn't scroll away.
REPEAT_ALERT_SECONDS = 10.0

# Compare every Nth pixel, a strided view rather than a resize so sensor noise isn't
# averaged away.
_STRIDE = 8


def fingerprint(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frame[::_STRIDE, ::_STRIDE])


class FrozenCameraMonitor:
    """Tracks how long each camera feed's image has been unchanging."""

    def __init__(self, frozen_seconds: float = FROZEN_SECONDS, names: dict | None = None):
        self.frozen_seconds = frozen_seconds
        self.names = names or {}
        self._last_fingerprint: dict = {}
        self._last_change_t: dict = {}
        self._alerted_at: dict = {}

    def name(self, feed) -> str:
        return self.names.get(feed, f"feed {feed}")

    def note_frame(self, feed, frame: np.ndarray) -> None:
        """Record that `frame` arrived on `feed`."""
        fp = fingerprint(frame)
        prev = self._last_fingerprint.get(feed)
        if prev is None or prev.shape != fp.shape or not np.array_equal(fp, prev):
            self._last_change_t[feed] = time.monotonic()
        self._last_fingerprint[feed] = fp

    def forget(self, feed) -> None:
        self._last_fingerprint.pop(feed, None)
        self._last_change_t.pop(feed, None)
        self._alerted_at.pop(feed, None)

    def frozen_feeds(self) -> dict:
        now = time.monotonic()
        return {feed: now - t for feed, t in self._last_change_t.items()
                if now - t >= self.frozen_seconds}

    def new_alerts(self) -> list:
        """Frozen feeds to alert about now, at most once per feed per REPEAT_ALERT_SECONDS."""
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
        """One-line operator message, identical while the same feeds stay frozen so the UI
        doesn't reopen its dialog."""
        return ("FROZEN CAMERA: " + ", ".join(self.name(feed) for feed, _ in alerts) +
                " - the image has stopped changing. Episodes recorded now are unusable; "
                "restart the session.")
