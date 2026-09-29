"""Recognizes a spool whose line has come off it, from the line records every anchor sends.

A derailed spool cannot pay out: whatever the line has jammed against resists the spool turning
outward, the extra torque reads as a strongly negative tension, and the soft mute stops the
payout. So the spool sits still while it is told to pay out, and its tension jumps well below
zero each time the mute lets it try. A healthy line that has gone slack can also be muted, but
nothing resists it, so its tension sits near zero instead.

Reeling in is no help to tell them apart: a derailed spool turns freely without the tension
rising, but so does a healthy line taking up slack.
"""

from collections import deque

# (s) how far back a line's payout records are judged over. Long enough that the accel limit
# and the mute's brief stops on a healthy line even out.
WINDOW_S = 3.0
# (s) of payout that must have been commanded within the window before it is judged
MIN_PAYOUT_S = 1.0
# fraction of that commanded payout a derailed spool delivers: 0.01-0.02 across a whole
# derailment. A healthy line muted while slack can stall just as completely for a window, so
# this alone does not tell them apart; the resisted tension below does.
MAX_MOVED_FRACTION = 0.1
# (N) a payout tension this far below zero is the spool being resisted, not a slack line.
RESISTED_TENSION_N = -1.0
# fraction of the window's payout records that must read that low. Over a stalled window
# healthy lines stayed under 2%, except the other spool on a derailed spool's anchor, which
# after the derailment reached up to 8%.
MIN_RESISTED_FRACTION = 0.10
# (m/s) commanded speed below which a line is not counted as being told to pay out. Low,
# because once a spool has derailed a seek mostly asks it for a few cm/s either way.
MIN_PAYOUT_CMD_MPS = 0.02
# (s) gap between records beyond which the time between them is not integrated
MAX_RECORD_GAP_S = 0.1


class DerailDetector:
    """Feed it every line record as it arrives with the speed the line was told to run at;
    it says which lines look derailed."""

    def __init__(self, n_lines):
        # per line, (time, commanded m, moved m, tension N, dt) for each payout record in the window
        self.windows = [deque() for _ in range(n_lines)]
        self.cmd_total = [0.0] * n_lines
        self.moved_total = [0.0] * n_lines
        self.payout_s = [0.0] * n_lines
        self.last_t = [None] * n_lines

    def reset(self, line_no=None):
        lines = range(len(self.windows)) if line_no is None else [line_no]
        for i in lines:
            self.windows[i].clear()
            self.cmd_total[i] = 0.0
            self.moved_total[i] = 0.0
            self.payout_s[i] = 0.0
            self.last_t[i] = None

    def add(self, line_no, records, cmd_speed):
        """records: [(time, length, speed, tension), ...] oldest first. cmd_speed: the aim
        speed (m/s, positive pays out) the line was last told, or None if unknown."""
        for t, _, speed, tension in records:
            prev = self.last_t[line_no]
            self.last_t[line_no] = t
            if prev is None:
                continue
            dt = t - prev
            if not 0 < dt <= MAX_RECORD_GAP_S:
                continue
            if cmd_speed is None or cmd_speed < MIN_PAYOUT_CMD_MPS:
                continue
            entry = (t, cmd_speed * dt, speed * dt, tension, dt)
            self.windows[line_no].append(entry)
            self.cmd_total[line_no] += entry[1]
            self.moved_total[line_no] += entry[2]
            self.payout_s[line_no] += dt
        # expire by the newest record seen, so time spent not paying out ages the window too
        window = self.windows[line_no]
        now = self.last_t[line_no]
        while window and now is not None and now - window[0][0] > WINDOW_S:
            old = window.popleft()
            self.cmd_total[line_no] -= old[1]
            self.moved_total[line_no] -= old[2]
            self.payout_s[line_no] -= old[4]

    def verdict(self, line_no):
        """(derailed, moved fraction, resisted fraction) over the line's payout window. Both
        fractions are None until the window holds MIN_PAYOUT_S of commanded payout."""
        window = self.windows[line_no]
        if self.payout_s[line_no] < MIN_PAYOUT_S or self.cmd_total[line_no] <= 0:
            return False, None, None
        moved = self.moved_total[line_no] / self.cmd_total[line_no]
        resisted = sum(1 for e in window if e[3] < RESISTED_TENSION_N) / len(window)
        derailed = moved < MAX_MOVED_FRACTION and resisted >= MIN_RESISTED_FRACTION
        return derailed, moved, resisted
