"""
Unit tests for the live frozen-camera detector in frozen_camera_monitor.py.

Tests cover:
- A feed delivering fresh (noisy) frames is never reported frozen
- A feed repeating one image is reported once the threshold elapses
- A dead stream (no frames at all) is reported the same way
- Feeds are only tracked once they have delivered a frame
- Alerts repeat on a timer rather than every poll, and clear on recovery
- forget() resets a reconnecting feed
"""

import unittest
from unittest.mock import patch

import numpy as np

from nf_robot.ml.frozen_camera_monitor import (
    REPEAT_ALERT_SECONDS,
    FrozenCameraMonitor,
    fingerprint,
)


def _noisy_frame(rng, shape=(384, 684, 3)):
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


class TestFingerprint(unittest.TestCase):

    def test_identical_frames_share_a_fingerprint(self):
        frame = np.random.default_rng(0).integers(0, 256, (100, 120, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(fingerprint(frame), fingerprint(frame.copy())))

    def test_subsample_is_much_smaller_than_the_frame(self):
        frame = np.zeros((384, 684, 3), dtype=np.uint8)
        self.assertLess(fingerprint(frame).size, frame.size // 32)


class TestFrozenCameraMonitor(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1234)
        self.monitor = FrozenCameraMonitor(frozen_seconds=3.0, names={0: "gripper_camera"})

    def test_live_feed_never_reported_frozen(self):
        now = 100.0
        with patch("time.monotonic", side_effect=lambda: now):
            for _ in range(300):  # 10 s at 30 fps
                self.monitor.note_frame(0, _noisy_frame(self.rng))
                now += 1 / 30
                self.assertEqual(self.monitor.frozen_feeds(), {})

    def test_repeated_image_reported_after_threshold(self):
        now = 100.0
        stuck = _noisy_frame(self.rng)
        with patch("time.monotonic", side_effect=lambda: now):
            self.monitor.note_frame(0, stuck)
            for _ in range(60):  # 2 s of the same image: not yet a stall
                now += 1 / 30
                self.monitor.note_frame(0, stuck.copy())
            self.assertEqual(self.monitor.frozen_feeds(), {})

            for _ in range(45):  # 1.5 s more, past the 3 s threshold
                now += 1 / 30
                self.monitor.note_frame(0, stuck.copy())
            frozen = self.monitor.frozen_feeds()
            self.assertIn(0, frozen)
            self.assertAlmostEqual(frozen[0], 3.5, places=1)

    def test_dead_stream_delivering_no_frames_is_reported(self):
        now = 100.0
        with patch("time.monotonic", side_effect=lambda: now):
            self.monitor.note_frame(0, _noisy_frame(self.rng))
            now += 5.0  # decode loop died; nothing arrives at all
            self.assertIn(0, self.monitor.frozen_feeds())

    def test_feed_without_any_frame_is_not_tracked(self):
        now = 100.0
        with patch("time.monotonic", side_effect=lambda: now):
            now += 60.0
            self.assertEqual(self.monitor.frozen_feeds(), {})
            self.assertEqual(self.monitor.new_alerts(), [])

    def test_alerts_are_rate_limited_and_clear_on_recovery(self):
        now = 100.0
        stuck = _noisy_frame(self.rng)
        with patch("time.monotonic", side_effect=lambda: now):
            self.monitor.note_frame(0, stuck)
            now += 4.0
            self.assertEqual(len(self.monitor.new_alerts()), 1)
            # Still frozen, but too soon to repeat the warning.
            now += 1.0
            self.assertEqual(self.monitor.new_alerts(), [])
            now += REPEAT_ALERT_SECONDS
            self.assertEqual(len(self.monitor.new_alerts()), 1)

            # Camera recovers, then freezes again: that is a fresh alert.
            self.monitor.note_frame(0, _noisy_frame(self.rng))
            self.assertEqual(self.monitor.new_alerts(), [])
            now += 4.0
            self.assertEqual(len(self.monitor.new_alerts()), 1)

    def test_alert_message_names_the_camera(self):
        now = 100.0
        stuck = _noisy_frame(self.rng)
        with patch("time.monotonic", side_effect=lambda: now):
            self.monitor.note_frame(0, stuck)
            now += 4.0
            message = self.monitor.describe(self.monitor.new_alerts())
        self.assertIn("gripper_camera", message)
        self.assertIn("FROZEN CAMERA", message)

    def test_forget_resets_a_reconnecting_feed(self):
        now = 100.0
        stuck = _noisy_frame(self.rng)
        with patch("time.monotonic", side_effect=lambda: now):
            self.monitor.note_frame(0, stuck)
            now += 4.0
            self.assertIn(0, self.monitor.frozen_feeds())
            self.monitor.forget(0)
            self.assertEqual(self.monitor.frozen_feeds(), {})

    def test_feeds_are_tracked_independently(self):
        now = 100.0
        stuck = _noisy_frame(self.rng)
        with patch("time.monotonic", side_effect=lambda: now):
            for _ in range(150):  # 5 s: feed 0 stuck, feed 1 live
                self.monitor.note_frame(0, stuck.copy())
                self.monitor.note_frame(1, _noisy_frame(self.rng))
                now += 1 / 30
            frozen = self.monitor.frozen_feeds()
        self.assertIn(0, frozen)
        self.assertNotIn(1, frozen)

    def test_frame_size_change_counts_as_a_change(self):
        now = 100.0
        with patch("time.monotonic", side_effect=lambda: now):
            self.monitor.note_frame(0, np.zeros((384, 684, 3), dtype=np.uint8))
            now += 4.0
            self.monitor.note_frame(0, np.zeros((512, 512, 3), dtype=np.uint8))
            self.assertEqual(self.monitor.frozen_feeds(), {})


if __name__ == "__main__":
    unittest.main()
