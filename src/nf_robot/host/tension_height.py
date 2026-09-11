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
independently observed limp height, and a fair check on the whole model. `height_sensitivity`
gives the local error-per-newton so a reading can be weighed on the spot.

Run the other way, the same balance predicts what each line should read at a known position,
which is what `predict_tensions` does and `TensionResidualLogger` records against the
measured tensions.

What this cannot see: the weight is only constant while the whole assembly hangs free. Setting
the gripper down on the floor, or picking up a payload, changes W and therefore shifts every
height and predicted tension this module reports. Line weight is also not modelled - at a few
newtons of total load the cables sag noticeably, which biases the effective anchor direction.
"""
import csv
import logging
from math import sqrt

import numpy as np

logger = logging.getLogger(__name__)

# a line pulling less than this is slack: it is holding nothing up and its direction is
# whatever the loose line happens to be doing, so it is left out of the force balance.
SLACK_TENSION_N = 0.0275

GRAVITY = 9.80665

HANGING_MASS_KG = 0.541
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


def predict_tensions(anchor_points, tensions, position, weight_n=HANGING_WEIGHT_N):
    """The tension each line should read for the gantry to hang still at `position`.

    Four tensions against three force equations leave one line's worth of freedom, so no
    single line's tension follows from geometry alone. Each line is instead predicted from the
    other three: the pull along line i that, added to what the other lines read, best cancels
    the weight in the least-squares sense. For a unit direction u_i toward anchor i that is

        T_i_pred = u_i . (W z_hat - sum_{j != i} T_j u_j)  =  T_i - u_i . (sum_j T_j u_j - W z_hat)

    so the residual T_i - T_i_pred is the force imbalance at `position` projected onto line i.
    Unlike solving the vertical balance alone for T_i, this does not divide by the line's
    elevation and stays well conditioned right up against the ceiling.

    A predicted tension below zero means the other three lines already pull harder than the
    weight can account for. Only meaningful while the assembly hangs free with no payload.
    """
    d = anchor_points - position
    directions = d / np.linalg.norm(d, axis=1)[:, None]
    imbalance = tensions @ directions
    imbalance[2] -= weight_n
    return tensions - directions @ imbalance


class TensionResidualLogger:
    """Writes predicted and measured line tensions, and their difference, to a CSV.

    Rows are written at whatever rate `observe` is called. A one-line summary also goes to the
    python log every LOG_PERIOD_S so a disagreement is visible without opening the CSV.
    """

    LOG_PERIOD_S = 2.0

    def __init__(self, path='tension_residual_log.csv', weight_n=HANGING_WEIGHT_N):
        self.path = path
        self.weight_n = weight_n
        self._file = None
        self._writer = None
        self._disabled = False
        self._last_log = 0.0

    def observe(self, ts, anchor_points, tensions, position, velocity, holding=False):
        """Predict every line's tension at `position` and record it. Returns the residuals,
        measured minus predicted, in newtons."""
        predicted = predict_tensions(anchor_points, tensions, position, self.weight_n)
        residual = tensions - predicted
        speed = sqrt(float(np.dot(velocity, velocity)))

        if not self._disabled:
            if self._writer is None:
                self._open()
            if self._writer is not None:
                self._writer.writerow([
                    f'{ts:.3f}',
                    *[f'{p:.4f}' for p in position],
                    *[f'{t:.4f}' for t in tensions],
                    *[f'{t:.4f}' for t in predicted],
                    *[f'{r:.4f}' for r in residual],
                    f'{speed:.4f}', int(holding),
                ])
                self._file.flush()

        if ts - self._last_log >= self.LOG_PERIOD_S:
            self._last_log = ts
            logger.info(
                'tension residual z=%.3f T=[%s] T_pred=[%s] residual=[%s] speed=%.3f holding=%s',
                position[2],
                ' '.join(f'{t:.2f}' for t in tensions),
                ' '.join(f'{t:.2f}' for t in predicted),
                ' '.join(f'{r:+.2f}' for r in residual),
                speed, holding)
        return residual

    def _open(self):
        try:
            self._file = open(self.path, 'a', newline='')
        except OSError as e:
            logger.warning(f'Could not open tension residual log {self.path}: {e}')
            self._disabled = True
            return
        self._writer = csv.writer(self._file)
        if self._file.tell() == 0:
            self._writer.writerow([
                'time', 'kf_x', 'kf_y', 'kf_z',
                'tension_0', 'tension_1', 'tension_2', 'tension_3',
                'predicted_0', 'predicted_1', 'predicted_2', 'predicted_3',
                'residual_0', 'residual_1', 'residual_2', 'residual_3',
                'speed_mps', 'holding',
            ])

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

