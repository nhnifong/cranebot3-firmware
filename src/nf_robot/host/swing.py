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

# The residual below which cancellation counts as damping this robot rather than fighting it.
# A settled swing reads well under this and one that pumps hits SAFETY_AMP_RAD instead, so
# there is room either side. Both places that judge a robot's cancellation - the latency sweep
# and the check at the end of calibration - answer the question with this one number, and what
# they conclude is stored as config.swing_cancellation_verified.
DAMPED_RESIDUAL_RAD = 0.15


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
COARSE_GOOD_ENOUGH_RAD = DAMPED_RESIDUAL_RAD
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


# ===== measuring the pole =====

# The swing has to be somewhere in here to be a swing at all: a 0.25m pendulum rings at
# 1Hz and a 1.5m one at 0.4Hz, and anything outside that is a bumped gantry or gyro noise.
FREQ_SEARCH_HZ = (0.35, 1.10)
# A shorter recording holds too few swings to time; ~6 of them.
MIN_RECORD_S = 8.0
# resampling rate, comfortably above the swing but below the IMU's 100Hz
RESAMPLE_HZ = 50.0
# zero-pad the spectrum by this factor, so the coarse peak lands near enough to the real
# one for a narrow band around it to contain the swing
FFT_PAD = 8
# half-width of that band, wide enough to keep the peak even when the coarse pick is a
# little off, narrow enough that noise outside it cannot make a zero crossing
BAND_HALF_WIDTH_HZ = 0.25
# Stop reading once the swing has decayed to this fraction of its strongest. Past that
# point the crossings are being timed off noise rather than off the pendulum.
MIN_SWING_FRACTION = 0.25
# fewer crossings than this is not a line worth fitting
MIN_CROSSINGS = 6
# How far the refined frequency may sit from the spectrum's peak. Beyond this the crossings
# found something other than the swing, and the coarse peak is the safer answer.
MAX_REFINEMENT_SHIFT = 0.10


def length_for_frequency(freq_hz):
    """Pendulum length in metres that swings at freq_hz."""
    return GRAVITY / (2 * np.pi * freq_hz) ** 2


def _resample(samples):
    """Gyro samples on an even time grid, each axis with its bias removed.

    The IMU loop is an asyncio sleep, so its samples come near 100Hz but never exactly,
    and everything below wants uniform spacing. Removing each axis's mean takes the gyro's
    bias out with it, which would otherwise be the loudest thing in the spectrum.
    """
    ts = samples[:, 0]
    n = int((ts[-1] - ts[0]) * RESAMPLE_HZ)
    grid = np.linspace(ts[0], ts[-1], n)
    signal = np.stack([np.interp(grid, ts, samples[:, i]) - np.mean(samples[:, i])
                       for i in (1, 2)])
    return grid, signal


def _coarse_frequency(signal):
    """Loudest in-band frequency, from the spectrum of both gyro axes together.

    A swing induced along one axis still leaks into the other as the gripper spins, so
    summing the two spectra keeps that energy rather than making the caller pick an axis.
    """
    n = signal.shape[1]
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(n), n=n * FFT_PAD)).sum(axis=0)
    freqs = np.fft.rfftfreq(n * FFT_PAD, 1 / RESAMPLE_HZ)
    in_band = (freqs >= FREQ_SEARCH_HZ[0]) & (freqs <= FREQ_SEARCH_HZ[1])
    if not in_band.any():
        return None
    return float(freqs[int(np.argmax(np.where(in_band, spectrum, 0.0)))])


def _bandpass(signal, freq):
    """Both axes with everything but a narrow band around freq removed."""
    n = signal.shape[1]
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1 / RESAMPLE_HZ)
    spectrum[:, (freqs < freq - BAND_HALF_WIDTH_HZ) | (freqs > freq + BAND_HALF_WIDTH_HZ)] = 0
    return np.fft.irfft(spectrum, n=n)


def _swinging_span(x, freq):
    """The slice of a band-passed axis that still holds a real swing.

    Amplitude is read as an RMS over one period, so a decay that runs into the noise floor
    is cut off there rather than contributing crossings that time the noise.
    """
    width = max(3, int(RESAMPLE_HZ / freq))
    rms = np.sqrt(np.convolve(x ** 2, np.ones(width) / width, mode='same'))
    loud = rms >= MIN_SWING_FRACTION * rms.max()
    start = int(np.argmax(loud))
    quiet_after = np.argmax(~loud[start:])
    return start, len(x) if quiet_after == 0 else start + int(quiet_after)


def _refine_frequency(grid, filtered, freq):
    """Frequency from the timing of the zero crossings, or None if there are too few.

    The spectrum can only place the peak to within a bin, and a bin over a recording this
    short is worth several millimetres of length. Crossing times carry the period far more
    precisely, and unlike the spectrum they do not care that the swing is decaying - so
    the peak picks the band and the crossings measure inside it.
    """
    x = filtered[int(np.argmax(filtered.std(axis=1)))]
    start, end = _swinging_span(x, freq)
    x, grid = x[start:end], grid[start:end]
    if len(x) < RESAMPLE_HZ * 4 / freq:
        return None

    crossings = []
    # half a period, so noise riding on a crossing cannot be counted as several
    min_gap = 0.4 / freq
    for i in np.where(np.sign(x[:-1]) != np.sign(x[1:]))[0]:
        fraction = x[i] / (x[i] - x[i + 1])
        t = grid[i] + fraction * (grid[i + 1] - grid[i])
        if crossings and t - crossings[-1] < min_gap:
            continue
        crossings.append(t)

    # the band-pass rings at the ends of the record, which moves the outermost crossings
    if len(crossings) > MIN_CROSSINGS + 2:
        crossings = crossings[1:-1]
    if len(crossings) < MIN_CROSSINGS:
        return None
    # crossings come every half period, so a line through them has that for its slope
    half_period = np.polyfit(np.arange(len(crossings)), crossings, 1)[0]
    return float(1 / (2 * half_period))


def measure_swing_frequency(samples):
    """Swing frequency in Hz from raw gyro samples, or None if there is no swing in them.

    samples is (n, 3) of (time, gyro x, gyro y) at whatever rate and jitter the gripper
    delivered.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        return None
    if samples[-1, 0] - samples[0, 0] < MIN_RECORD_S:
        return None

    grid, signal = _resample(samples)
    coarse = _coarse_frequency(signal)
    if coarse is None:
        return None
    refined = _refine_frequency(grid, _bandpass(signal, coarse), coarse)
    if refined is None or abs(refined - coarse) / coarse > MAX_REFINEMENT_SHIFT:
        return coarse
    return refined


def measure_pendulum(samples):
    """(frequency in Hz, effective length in metres) from raw gyro samples.

    The effective length is what the swing behaves as, not the pole's own length: the
    gripper is a body with its own moment of inertia hanging on the end of it, so this is
    measured rather than reached for with a tape. It is the number POLE_GEOMETRY wants.

    Returns (None, None) when the recording holds no usable swing.
    """
    freq = measure_swing_frequency(samples)
    if freq is None or freq <= 0:
        return None, None
    return freq, length_for_frequency(freq)
