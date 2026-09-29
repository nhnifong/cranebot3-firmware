"""
Unit tests for DerailDetector
"""
import unittest

from nf_robot.host.derail_detector import DerailDetector


def records(n, speed, tension, start=0.0, dt=0.02):
    """n line records 50 Hz apart. tension may be a function of the record index."""
    return [(start + k * dt, 5.0, speed, tension(k) if callable(tension) else tension)
            for k in range(n)]


class TestDerailDetector(unittest.TestCase):
    def test_no_verdict_before_enough_payout(self):
        d = DerailDetector(1)
        d.add(0, records(30, 0.0, -2.0), 0.3)
        self.assertEqual(d.verdict(0), (False, None, None))

    def test_healthy_payout(self):
        d = DerailDetector(1)
        d.add(0, records(150, 0.28, 3.0), 0.3)
        derailed, moved, resisted = d.verdict(0)
        self.assertFalse(derailed)
        self.assertGreater(moved, 0.9)

    def test_slack_line_muted_is_not_derailed(self):
        # stalled, but nothing resists it: tension sits near zero
        d = DerailDetector(1)
        d.add(0, records(150, 0.0, lambda k: 0.02 * (-1) ** k), 0.3)
        derailed, moved, resisted = d.verdict(0)
        self.assertFalse(derailed)
        self.assertLess(moved, 0.1)
        self.assertEqual(resisted, 0)

    def test_derailed_spool(self):
        # stalled, and every fifth record the mute lets it try and it is resisted hard
        d = DerailDetector(1)
        d.add(0, records(150, 0.0, lambda k: -2.5 if k % 5 == 0 else 0.2), 0.3)
        derailed, moved, resisted = d.verdict(0)
        self.assertTrue(derailed)
        self.assertAlmostEqual(resisted, 0.2, delta=0.01)

    def test_reel_in_is_not_judged(self):
        d = DerailDetector(1)
        d.add(0, records(150, -0.3, -2.5), -0.3)
        self.assertEqual(d.verdict(0), (False, None, None))

    def test_window_expires(self):
        d = DerailDetector(1)
        d.add(0, records(150, 0.0, lambda k: -2.5 if k % 5 == 0 else 0.2), 0.3)
        # 4 s of standing still later, the stall has aged out of the window
        d.add(0, records(200, 0.0, 0.2, start=3.0), 0.0)
        self.assertEqual(d.verdict(0), (False, None, None))

    def test_reset(self):
        d = DerailDetector(2)
        for line in (0, 1):
            d.add(line, records(150, 0.0, lambda k: -2.5 if k % 5 == 0 else 0.2), 0.3)
        d.reset(0)
        self.assertEqual(d.verdict(0), (False, None, None))
        self.assertTrue(d.verdict(1)[0])


if __name__ == '__main__':
    unittest.main()
