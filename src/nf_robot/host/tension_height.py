"""
Estimate the height of the gantry from line tension alone.

The thesis under test is that the four tensions know how high the gantry is. They do, and
the falloff is geometry rather than an empirical curve that has to be measured. Each line
pulls the gantry straight at its anchor, so only the vertical component of each tension
holds up the suspended weight:

    sum_i  T_i * (z_i - z) / L_i  =  W

Near the ceiling the lines lie nearly flat, (z_i - z) / L_i approaches zero, and the tension
needed to hold W blows up like 1 / sin(elevation) - which is why two opposing corners hit the
28 N limp about 10 cm below the top of the work area. Well below the ceiling the lines are
steep, the ratio approaches one, and the sum of tensions approaches the weight itself.

Two useful consequences:

  * Extra tension the spools hold beyond what equilibrium requires - the tension floor, or one
    line fighting the line opposite it - lives in the null space of that force balance, so it
    adds nothing to the vertical sum and `height_from_tension` is blind to it. In the measured
    room geometries that null direction is close to (-1, +1, -1, +1): one diagonal pair pulls
    harder while the other slacks off by nearly the same amount. It very nearly cancels out of
    the plain sum of tensions too, which is why `height_from_summed_tension` also survives
    preload. What that cruder form does not survive is being off-center - see its docstring.
  * The horizontal components of the same sum should cancel. They don't cancel when a spool's
    reported torque is wrong, so `force_balance(...)[0:2]` is a free trustworthiness check on
    the tension readings that needs no ground truth.

How sharp the estimate is depends entirely on how close to the ceiling the gantry is, because
that is where the tension curve is steep. In the bedroom room geometry, carrying no payload:

    depth below ceiling   0.05 m   0.10 m   0.25 m   0.50 m   1.0 m    2.0 m
    peak line tension     50 N     32 N     15 N     7.7 N    4.0 N    2.2 N
    height error per      27 mm    39 mm    77 mm    143 mm   287 mm   665 mm
      newton of error

So this is a decent altimeter in the top half meter and close to useless down low. Note also
where that table crosses the 28 N limp threshold: about 11 cm below the ceiling, which is the
independently observed limp height, and a fair check on the whole model. `TensionHeightProbe`
logs the local error-per-newton with every sample so a reading can be weighed on the spot.

What this cannot see: the weight is only constant while the whole assembly hangs free. Setting
the gripper down on the floor, or picking up a payload, changes W and therefore shifts every
height this module reports. Line weight is also not modelled - at a few newtons of total load
the cables sag noticeably, which biases the effective anchor direction. Both show up as drift
in the implied weight that `TensionHeightProbe` logs alongside the estimate.
"""
import csv
import logging
import time
from math import sqrt

import numpy as np

logger = logging.getLogger(__name__)

# a line pulling less than this is slack: it is holding nothing up and its direction is
# whatever the loose line happens to be doing, so it is left out of the force balance.
SLACK_TENSION_N = 0.0275

GRAVITY = 9.80665

# suspended mass with no payload, in kilograms: gantry 245 g, pole 21 g, gripper 132 g.
HANGING_MASS_KG = (245 + 21 + 132) / 1000.0
HANGING_WEIGHT_N = HANGING_MASS_KG * GRAVITY


def force_balance(anchor_points, tensions, position):
    """Net force in newtons that the measured tensions apply to a body at `position`.

    Gravity is not included, so at equilibrium this equals (0, 0, W): the z component is the
    upward pull holding the gantry up and the xy components are a residual that is only near
    zero when the tension readings and the anchor geometry agree with each other.
    """
    d = anchor_points - position
    lengths = np.linalg.norm(d, axis=1)
    return (tensions / lengths) @ d


def height_from_tension(anchor_points, tensions, xy, weight_n, z_floor=0.0, tol=1e-4):
    """Height at which the vertical components of `tensions` would hold up `weight_n`.

    `xy` is the horizontal position of the gantry, which the lines' elevation angles depend on
    but which tension cannot resolve on its own. Returns None when no height in the work area
    balances the weight, which means the lines are collectively pulling too weakly to be
    holding the gantry up at all - the usual cause is a payload or the gripper on the floor.
    """
    ceiling = float(np.min(anchor_points[:, 2]))
    # horizontal distance squared to each anchor, fixed for the whole search
    r2 = (anchor_points[:, 0] - xy[0]) ** 2 + (anchor_points[:, 1] - xy[1]) ** 2
    az = anchor_points[:, 2]

    def vertical_pull(z):
        dz = az - z
        return float(np.sum(tensions * dz / np.sqrt(r2 + dz * dz)))

    # every line gets steeper as the gantry descends, so vertical pull decreases monotonically
    # in z and a bisection is both safe and faster than a general root finder.
    if vertical_pull(z_floor) < weight_n:
        return None
    if vertical_pull(ceiling) > weight_n:
        return ceiling

    lo, hi = z_floor, ceiling
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if vertical_pull(mid) > weight_n:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def height_sensitivity(anchor_points, tensions, xy, z):
    """Meters of height error per newton of error in the weight (or in the summed tension).

    The derivative of the force balance at the solved height, which is how the estimate should
    be weighed against anything else: it runs from a few cm/N just under the ceiling to most of
    a meter per newton down low, where every line is nearly vertical and tension stops carrying
    height information at all. Returned negative, since reading more tension means less height.
    """
    dz = anchor_points[:, 2] - z
    r2 = (anchor_points[:, 0] - xy[0]) ** 2 + (anchor_points[:, 1] - xy[1]) ** 2
    lengths = np.sqrt(r2 + dz * dz)
    slope = float(np.sum(tensions * r2 / lengths ** 3))
    if slope <= 0:
        return float('nan')
    return -1.0 / slope


def height_from_summed_tension(anchor_points, tensions, xy, weight_n):
    """The simpler thesis: height from the sum of the tensions, with no per-line geometry.

    Assumes all four lines share one elevation angle, which is true only when the gantry hangs
    at the middle of the work area. Then sum(T) / W = L / depth, and the depth below the anchor
    plane follows in closed form. Preload barely touches it, but being off-center does: in the
    bedroom geometry it agrees with the full solution to 2 mm under the centroid, is off by
    13 cm a meter out, and by 70 cm near a corner, always reading low. It is logged next to the
    full solution to show how much the per-line geometry is actually buying.
    """
    k = float(np.sum(tensions)) / weight_n
    if k <= 1.0:
        return None  # not even enough tension to hold the weight straight up
    mean_radius = float(np.mean(np.sqrt(
        (anchor_points[:, 0] - xy[0]) ** 2 + (anchor_points[:, 1] - xy[1]) ** 2)))
    depth = mean_radius / sqrt(k * k - 1)
    return float(np.mean(anchor_points[:, 2])) - depth


class TensionHeightProbe:
    """Logs a tension-only height estimate next to the fused estimate, to test the thesis.

    The weight is held fixed at the measured hanging mass rather than fitted. An earlier version
    averaged the implied weight while the gantry hung still and used that instead, which quietly
    destroyed the whole point: the implied weight is read off at the filter's position, so
    calibrating on it tunes the tension estimate to agree with the filter and z_tension stops
    being independent evidence. With the weight pinned, a disagreement is real.

    Every sample also reports that implied weight - what the suspended load would have to be for
    the measured tensions to be in equilibrium where the filter believes the gantry is. It is
    the sharpest diagnostic here: it should sit at HANGING_WEIGHT_N, and however far off it sits
    is how badly the tension readings and the anchor geometry disagree with each other.
    """

    # sample rates. the hang loop runs at 30 Hz, which is more rows than a CSV needs
    CSV_PERIOD_S = 0.1
    LOG_PERIOD_S = 2.0

    # below this speed the gantry counts as hanging still rather than being dragged around
    QUIET_SPEED_MPS = 0.02

    def __init__(self, path='tension_height_log.csv', weight_n=HANGING_WEIGHT_N):
        self.path = path
        # the weight the tensions are converted against, in newtons. constant on purpose.
        self.weight_n = weight_n
        self.implied_weight = None
        self._file = None
        self._writer = None
        self._last_csv = 0.0
        self._last_log = 0.0

    def observe(self, anchor_points, tensions, position, velocity, holding=False, ts=None):
        """Record one sample. Returns (z_from_tension, z_from_summed_tension, implied_weight).

        `position` and `velocity` are the fused estimate, used as the ground truth to compare
        against and to supply the horizontal position that tension alone cannot resolve.
        """
        ts = time.time() if ts is None else ts
        taut = tensions > SLACK_TENSION_N

        pull = force_balance(anchor_points, tensions, position)
        implied_weight = float(pull[2])
        horizontal_residual = sqrt(pull[0] ** 2 + pull[1] ** 2)
        self.implied_weight = implied_weight

        speed = sqrt(float(np.dot(velocity, velocity)))
        # samples worth believing: a payload or a gripper set down on the floor means the weight
        # is not HANGING_WEIGHT_N, a slack line means the geometry is underdetermined, and motion
        # means acceleration and spool stiction are both corrupting the torque readings.
        quiescent = (
            not holding
            and bool(np.all(taut))
            and speed < self.QUIET_SPEED_MPS
            and implied_weight > 0
        )

        z_tension = height_from_tension(anchor_points, tensions, position[:2], self.weight_n)
        z_summed = height_from_summed_tension(anchor_points, tensions, position[:2], self.weight_n)
        # how much a newton of tension error would move that answer, at wherever it landed
        sensitivity = height_sensitivity(
            anchor_points, tensions, position[:2],
            position[2] if z_tension is None else z_tension)

        self._write(ts, position, tensions, z_tension, z_summed, implied_weight,
                    horizontal_residual, sensitivity, speed, holding, quiescent)
        return z_tension, z_summed, implied_weight

    def _write(self, ts, position, tensions, z_tension, z_summed, implied_weight,
               horizontal_residual, sensitivity, speed, holding, quiescent):
        if ts - self._last_csv >= self.CSV_PERIOD_S:
            self._last_csv = ts
            if self._writer is None:
                self._open()
            if self._writer is not None:
                self._writer.writerow([
                    f'{ts:.3f}',
                    f'{position[0]:.4f}', f'{position[1]:.4f}', f'{position[2]:.4f}',
                    *[f'{t:.4f}' for t in tensions],
                    '' if z_tension is None else f'{z_tension:.4f}',
                    '' if z_summed is None else f'{z_summed:.4f}',
                    f'{implied_weight:.4f}', f'{self.weight_n:.4f}',
                    f'{horizontal_residual:.4f}', f'{sensitivity:.4f}', f'{speed:.4f}',
                    int(holding), int(quiescent),
                ])
                self._file.flush()

        if ts - self._last_log >= self.LOG_PERIOD_S:
            self._last_log = ts
            err = '' if z_tension is None else f' err={z_tension - position[2]:+.3f}'
            logger.info(
                'tension height z_kf=%.3f z_tension=%s z_summed=%s%s W_implied=%.2f W_assumed=%.2f '
                'hres=%.2f mm/N=%.0f T=[%s]',
                position[2],
                'none' if z_tension is None else f'{z_tension:.3f}',
                'none' if z_summed is None else f'{z_summed:.3f}',
                err, implied_weight, self.weight_n, horizontal_residual, abs(sensitivity) * 1000,
                ' '.join(f'{t:.2f}' for t in tensions))

    def _open(self):
        try:
            self._file = open(self.path, 'a', newline='')
        except OSError as e:
            logger.warning(f'Could not open tension height log {self.path}: {e}')
            self.CSV_PERIOD_S = 1e9  # stop retrying
            return
        self._writer = csv.writer(self._file)
        if self._file.tell() == 0:
            self._writer.writerow([
                'time', 'kf_x', 'kf_y', 'kf_z',
                'tension_0', 'tension_1', 'tension_2', 'tension_3',
                'z_tension', 'z_summed', 'implied_weight_n', 'assumed_weight_n',
                'horizontal_residual_n', 'meters_per_newton', 'speed_mps', 'holding', 'quiescent',
            ])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
