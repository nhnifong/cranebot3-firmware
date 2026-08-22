"""Pendulum math behind swing cancellation and its latency calibration.

The gripper hangs from the gantry on a pole and behaves as a pendulum of fixed length, so
its motion is one frequency and a two-column state matrix is enough to describe it: row 0
is the X swing and row 1 the Y, column 0 the gyro velocity (sine) and column 1 the
quarter-turn-ahead phase (cosine). The gripper fits that matrix to its gyro at 100Hz and
publishes it; everything here reads it.

The frequency comes from the pole the gripper hangs on, which differs by robot, so it is
carried on a Pendulum built from the config rather than fixed at import.

Only the arithmetic lives here. Driving spools, timing trials and talking to the UI stay
with their callers, which is what makes these testable without a robot.
"""

import logging

import numpy as np

import nf_robot.common.definitions as model_constants

logger = logging.getLogger(__name__)

GRAVITY = 9.81

SWING_CANCEL_GAIN = -0.12
CENTERING_GAIN = 0.4
# a paused control loop leaves a huge dt that would wreck the centering integrator
MAX_INTEGRATION_DT_S = 0.5

# what a latency calibration trial reads off a settle
MEASURE_PERIODS = 3          # average the swing over this many final periods
SAFETY_AMP_RAD = 0.4         # stop a trial early if the swing grows past this
MIN_SAMPLES = 10             # fewer amplitude readings than this is not a settle worth reading


class Pendulum:
    """The gripper swinging on one particular pole.

    Everything the swing frequency touches hangs off this, so a robot with a different
    pole gets a correction at the right phase without any of the callers knowing which
    pole it is.
    """

    def __init__(self, length):
        self.length = float(length)
        self.omega = np.sqrt(GRAVITY / self.length)
        self.period = 2 * np.pi / self.omega
        self.half_period = np.pi / self.omega

    def project(self, sm, dt):
        """The swing state matrix rotated forward by dt seconds.

        The pendulum's phase is all that evolves between updates, so advancing the model
        is a rotation of the state rather than a re-fit, and it stays exact for as long as
        the swing lasts.
        """
        angle = self.omega * dt
        c, s = np.cos(angle), np.sin(angle)
        return sm @ np.array([[c, -s], [s, c]])

    def cancel_velocity(self, sm, dt):
        """Gantry velocity, in the gripper's IMU frame, that opposes the swing dt from now.

        Projecting before opposing is what compensates for control latency: dt is the
        round trip from the IMU measurement to the spools actually moving, so the
        correction lands in phase rather than a fraction of a period late.
        """
        # angular acceleration is the derivative of the gyro velocity, which for this
        # model is omega * [-sin(theta), cos(theta)] - column 1 of the projected state.
        future_accel = self.omega * self.project(sm, dt)[:, 1]
        return future_accel * SWING_CANCEL_GAIN

    def tilt(self, sm, dt=0.0):
        """Gripper tilt [theta_x, theta_y, 0] in radians, dt seconds from the model's time."""
        projected = self.project(sm, dt)
        # displacement is the integral of velocity, so with col 0 velocity (A*sin) it is
        # -A/omega*cos: the phase tracker in col 1, over omega
        return np.array([projected[0, 1] / self.omega, projected[1, 1] / self.omega, 0])

    def amplitude(self, sm):
        """Angular amplitude of the swing in radians, 0.0 if there is none (or no IMU).

        Phase independent, so it can be read at any instant rather than by watching for a
        peak over a full period.
        """
        if sm is None:
            return 0.0
        return float(np.linalg.norm(sm) / self.omega)

    def trial_residual(self, ts, amps, aborted):
        """How much swing one latency trial settled to, or None if the trial says nothing.

        A good latency drives the swing to nothing and a bad one leaves a steady residual,
        so the average amplitude over the final periods ranks candidates. A trial that
        pumped past the safety cap is definitively bad and scores the cap; one cut short
        by drift or tension never settled and is excluded rather than counted as good.
        """
        if aborted == 'amp_cap':
            return SAFETY_AMP_RAD
        if aborted in ('tension', 'drift'):
            return None
        ts, amps = np.asarray(ts), np.asarray(amps)
        if len(amps) < MIN_SAMPLES:
            return None
        late = amps[ts > ts[-1] - MEASURE_PERIODS * self.period]
        return float(np.mean(late)) if len(late) else float(np.mean(amps))


def pendulum_for(config):
    """The Pendulum this robot's configured pole gives it."""
    return Pendulum(model_constants.pole_geometry(config).swing_length)


def integrate_centering(raw_vel, offset, dt):
    """Correct raw_vel to pull the gantry back toward where it started, and return
    (velocity, new offset).

    Cancelling swing is a net push in whichever direction the swing was going, so the
    platform drifts. Integrating our own commanded velocity gives the drift without
    needing a position estimate, and feeding it back cancels it.
    """
    if dt > MAX_INTEGRATION_DT_S or dt < 0:
        dt = 0.0
    vel = raw_vel - offset * CENTERING_GAIN
    return vel, offset + vel * dt


# ===== latency calibration =====

# seconds; spread wide enough to bracket the ideal even under heavy loop contention
COARSE_CANDS = (0.3, 0.0, 0.6)
# a coarse trial damped this well is worth refining around immediately
COARSE_GOOD_ENOUGH_RAD = 0.15
# fine pass spans +/- this around the coarse best (covers the gap between coarse samples)
FINE_HALF_WIDTH = 0.15
FINE_COUNT = 7
FINE_CLIP = (0.0, 0.75)      # keep refined candidates within a sane latency range

# "as good as the best" = within this (or 50%) of the smallest residual
FLOOR_MARGIN = 0.010

ALTITUDE_HOLD_GAIN = 4.0       # 1/s, proportional gain pulling z back to the start altitude
ALTITUDE_HOLD_MAX_MPS = 0.15   # cap on the vertical hold speed


def fine_candidates(best_coarse):
    """Latencies for the fine pass: an even spread around the coarse winner.

    Rounded to milliseconds and deduplicated so a spread that clips against the ends of
    the sane range does not spend trials measuring the same latency twice.
    """
    fine = np.clip(np.linspace(best_coarse - FINE_HALF_WIDTH, best_coarse + FINE_HALF_WIDTH,
                               FINE_COUNT), *FINE_CLIP)
    return sorted({float(x) for x in np.round(fine, 3)})


def altitude_hold_velocity(z_error):
    """Vertical speed that holds the gantry at its starting altitude.

    Cancellation drives the gantry sideways and, because it hangs from four lines, that
    pulls it upward; without this a long trial climbs out of the range the next one needs.
    """
    return float(np.clip(ALTITUDE_HOLD_GAIN * z_error, -ALTITUDE_HOLD_MAX_MPS, ALTITUDE_HOLD_MAX_MPS))


def select_min_residual(results):
    """Pick the center of the range of latencies that all damp the swing fully.

    The swing measurement can't read below a small floor (~20 mrad), so every latency that
    fully damps ties near that floor -- the best isn't a single point but a range. Any
    latency in that range works; we return its midpoint, which sits farthest from the
    edges where damping starts to fail and is more repeatable than picking an edge.

    results is a list of (latency, residual). Duplicate latencies keep their best reading
    so one bad settle doesn't reject an otherwise-good latency.
    """
    groups = {}
    for lat, r in results:
        groups.setdefault(round(lat, 3), []).append(r)
    lats = np.array(sorted(groups))
    resid = np.array([min(groups[l]) for l in lats])

    rmin = float(resid.min())
    at_floor = resid <= rmin + max(0.5 * rmin, FLOOR_MARGIN)

    i0 = int(np.argmin(resid))
    lo = hi = i0
    while lo - 1 >= 0 and at_floor[lo - 1]:
        lo -= 1
    while hi + 1 < len(lats) and at_floor[hi + 1]:
        hi += 1
    best = float((lats[lo] + lats[hi]) / 2)
    logger.info(f'Fully-damped latency range {lats[lo]:.3f}-{lats[hi]:.3f}s; picking center {best:.3f}s')
    return best
