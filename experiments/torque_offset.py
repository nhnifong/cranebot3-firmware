"""Measure the unloaded holding torque of a DaMiao spool motor for each direction of spin.

With nothing on the line, the torque a stopped motor reports is the friction it was pushing
through while it moved, held by the speed loop. So it lands on one value after turning one
way and another after turning the other way. That value also depends on where the rotor
stops, so this walks once around a revolution in N stations, and at each one makes a short
move forward and a short move back, reading the held torque after each.

On an installed anchor the line stays attached, so --walk picks the direction of the net
revolution (walk the reel-in way so it takes up slack instead of paying loose line out), and
the run aborts, disabling every motor in --disable, if the torque implies more line tension
than --tension-limit, the rotor strays past --max-travel, or feedback stops arriving.
"""
import argparse
import math
import random
import statistics
import sys
import time

from damiao_motor import DaMiaoController

LOOP_HZ = 100
RUNTIME_ACCEL = 0.01  # spool_dm MAX_ACCEL, restored on exit
RAD_S2_PER_ACCEL = 1000.0  # ACC register 0.01 measured as a 10 rad/s^2 ramp
SPOOL_DIAMETER_M = 0.072  # empty spool, for the tension equivalent only
STILL_VEL = 0.03  # rad/s. zero reads as -0.011 (half a count)
POS_WRAP = 25.0  # reported position wraps over [-12.5, 12.5] rad
STALE_TICKS = 20  # consecutive commands with no new feedback frame before aborting


class Abort(Exception):
    pass


class Motor:
    def __init__(self, m, tension_limit_n, max_travel_rad):
        self.m = m
        self.raw = None
        self.pos = 0.0  # unwrapped rad
        self.start = None
        self.state = {}
        # empty spool diameter gives the largest tension per Nm, so this trips early, not late
        self.torque_limit = tension_limit_n * SPOOL_DIAMETER_M / 2
        self.max_travel = max_travel_rad
        self._last_frame = None
        self._stale = 0

    def tick(self, vel):
        t0 = time.time()
        self.m.send_cmd_vel(target_velocity=vel)
        time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.time() - t0)))
        # the driver replaces its state dict on every decoded frame
        frame = self.m.state
        if frame is None or frame is self._last_frame:
            self._stale += 1
            if self._stale >= STALE_TICKS:
                raise Abort(f'no feedback for {self._stale} commands')
        else:
            self._stale = 0
        self._last_frame = frame
        self.state = self.m.get_states()
        torq = self.state.get('torq')
        if torq is not None and abs(torq) > self.torque_limit:
            raise Abort(f'torque {torq:+.4f} Nm is over {self.torque_limit:.4f} Nm '
                        f'(~{abs(torq) * 2 / SPOOL_DIAMETER_M:.1f} N of line tension)')
        raw = self.state.get('pos')
        if raw is not None:
            if self.raw is not None:
                d = raw - self.raw
                d -= POS_WRAP * round(d / POS_WRAP)
                self.pos += d
            self.raw = raw
            if self.start is None:
                self.start = self.pos
            if abs(self.pos - self.start) > self.max_travel:
                raise Abort(f'rotor travelled {self.pos - self.start:+.2f} rad, past {self.max_travel} rad')
        return self.state

    def move_to(self, target, speed, accel, timeout_s=5.0):
        """Run toward target, releasing to zero early enough that decel lands near it. Speed is
        capped so a short move still gets up to speed and actually slides before it stops."""
        dist = target - self.pos
        sign = 1.0 if dist > 0 else -1.0
        speed = min(speed, math.sqrt(accel * abs(dist)))
        brake = speed * speed / (2 * accel)
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            vel = sign * self.tick(sign * speed).get('vel', 0.0)
            if sign * (target - self.pos) <= brake and vel >= 0.8 * speed:
                break

    def hold_and_read(self, still_s, read_s, timeout_s=2.0):
        """Command zero until still for still_s, then average torque over read_s."""
        t_start = time.time()
        still_since = None
        while time.time() - t_start < timeout_s:
            now = time.time()
            if abs(self.tick(0.0).get('vel', 1.0)) < STILL_VEL:
                still_since = still_since or now
                if now - still_since >= still_s:
                    break
            else:
                still_since = None
        torques = []
        t_end = time.time() + read_s
        while time.time() < t_end:
            torques.append(self.tick(0.0).get('torq', float('nan')))
        return statistics.fmean(torques)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id', type=lambda x: int(x, 0), default=0x01)
    ap.add_argument('--stations', type=int, default=12, help='stops spread over one revolution')
    ap.add_argument('--stroke', type=float, default=0.3, help='rad of the back-and-forth at each stop')
    ap.add_argument('--speed', type=float, default=3.0, help='rad/s')
    ap.add_argument('--accel', type=float, default=RUNTIME_ACCEL, help='ACC/DEC register value')
    ap.add_argument('--still', type=float, default=0.15, help='seconds still before reading')
    ap.add_argument('--read', type=float, default=0.05, help='seconds of torque to average')
    ap.add_argument('--walk', type=int, choices=(1, -1), default=1,
                    help='motor direction of the net revolution; use the reel-in direction on an installed anchor')
    ap.add_argument('--tension-limit', type=float, default=10.0, help='abort above this line tension, N')
    ap.add_argument('--max-travel', type=float, default=8.0, help='abort past this many rad from the start')
    ap.add_argument('--disable', type=lambda s: [int(x, 0) for x in s.split(',')], default=[],
                    help='other motor ids to disable on exit or abort, e.g. 1,2')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    accel = args.accel * RAD_S2_PER_ACCEL

    ctl = DaMiaoController(channel='can0', bustype='socketcan')
    motor = Motor(ctl.add_motor(motor_id=args.id, feedback_id=args.id, motor_type='G6215'),
                  args.tension_limit, args.max_travel)
    others = [ctl.add_motor(motor_id=i, feedback_id=i, motor_type='G6215')
              for i in args.disable if i != args.id]
    held = {1: [], -1: []}
    travel = 0.0
    aborted = None
    t_begin = time.time()
    try:
        motor.m.enable()
        time.sleep(0.1)
        motor.m.ensure_control_mode('VEL')
        motor.m.set_acceleration(args.accel)
        motor.m.set_deceleration(-args.accel)
        for _ in range(20):
            motor.tick(0.0)
        start = motor.pos
        step = 2 * math.pi / args.stations
        for i in range(args.stations):
            # a random spot within each station: the held torque also varies faster than the
            # station spacing, and evenly spaced stops sample that pattern at the same phase
            # every time, biasing the whole run by wherever it happened to start
            base = start + args.walk * (i + random.random()) * step
            for sign, target in ((args.walk, base + args.walk * args.stroke), (-args.walk, base)):
                motor.move_to(target, args.speed, accel)
                torque = motor.hold_and_read(args.still, args.read)
                held[sign].append(torque)
                if args.verbose:
                    print(f'station {i:2d} {"+" if sign > 0 else "-"} torque {torque:+.5f} Nm  '
                          f'pos {motor.pos - start:+.3f}', flush=True)
        travel = motor.pos - start
        for _ in range(5):
            motor.tick(0.0)
    except Abort as e:
        aborted = str(e)
    finally:
        # disable first, so an abort stops holding torque before anything else can fail
        for m in [motor.m, *others]:
            try:
                m.disable()
            except Exception as e:
                print(f'disable 0x{m.motor_id:02x} failed: {e}', file=sys.stderr)
        try:
            motor.m.set_acceleration(RUNTIME_ACCEL)
            motor.m.set_deceleration(-RUNTIME_ACCEL)
        except Exception as e:
            print(f'cleanup error: {e}', file=sys.stderr)
        ctl.shutdown()

    if aborted:
        print(f'ABORTED motor 0x{args.id:02x} after {time.time() - t_begin:.1f} s, '
              f'{motor.pos - (motor.start or 0.0):+.2f} rad from start: {aborted}. '
              f'Disabled 0x{args.id:02x}{"".join(f", 0x{m.motor_id:02x}" for m in others)}.')
        sys.exit(1)
    plus, minus = held[1], held[-1]
    if not plus or not minus:
        return
    k = 2.0 / SPOOL_DIAMETER_M  # newtons of line tension per Nm on an empty spool
    rows = []
    for name, vals in (('+', plus), ('-', minus)):
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append((name, statistics.fmean(vals), sd, sd / len(vals) ** 0.5))
    mid = (rows[0][1] + rows[1][1]) / 2
    half = (rows[0][1] - rows[1][1]) / 2
    print(f'motor 0x{args.id:02x}  {time.time() - t_begin:.1f} s  net travel {travel:+.2f} rad')
    for name, mean, sd, se in rows:
        print(f'offset after {name} spin: {mean:+.5f} Nm  sd {sd:.5f}  se {se:.5f}')
    print(f'midpoint {mid:+.5f} Nm  half-band {half:.5f} Nm  '
          f'(x{k:.1f} on a 72 mm spool: {mid * k:+.3f} N, +/-{half * k:.3f} N)')


if __name__ == '__main__':
    main()
