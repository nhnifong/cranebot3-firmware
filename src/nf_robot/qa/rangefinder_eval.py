# Live readout for quickly evaluating VL53L1X rangefinders hot swapped onto a gripper's I2C bus.
# Each time a rangefinder comes online it gets a fresh set of stats: current range, rolling
# variance, and min/max seen. Point the sensor at open space or a distant surface. A sensor with
# its protective film still under the window shows noisy, short readings from reflections off the film.

import time
import math
import argparse
from collections import deque
import board
import busio
from adafruit_vl53l1x import VL53L1X

from nf_robot.robot.gripper_i2c import VL53L1X_ADDR

POLL_INTERVAL = 0.3 # seconds between bus scans while waiting for a rangefinder
STALL_TIMEOUT = 1.0 # no new data for this long means the sensor was swapped or stopped ranging
CLEAR_LINE = '\r\x1b[K'


def rangefinder_present(i2c):
    while not i2c.try_lock():
        pass
    try:
        return VL53L1X_ADDR in i2c.scan()
    except OSError:
        return False
    finally:
        i2c.unlock()


def wait_for_rangefinder(i2c):
    print(f'{CLEAR_LINE}Waiting for a rangefinder at 0x{VL53L1X_ADDR:02x}...', end='', flush=True)
    while not rangefinder_present(i2c):
        time.sleep(POLL_INTERVAL)
    time.sleep(0.1) # let the sensor finish booting after the connector seats


class Stats:
    def __init__(self, window):
        self.recent = deque(maxlen=window)
        self.min = math.inf
        self.max = -math.inf
        self.valid = 0
        self.invalid = 0

    def add(self, distance):
        if distance is None:
            self.invalid += 1
            return
        self.valid += 1
        self.recent.append(distance)
        self.min = min(self.min, distance)
        self.max = max(self.max, distance)

    def variance(self):
        n = len(self.recent)
        if n < 2:
            return 0.0
        mean = sum(self.recent) / n
        return sum((d - mean) ** 2 for d in self.recent) / (n - 1)

    def summary(self, current):
        var = self.variance()
        total = self.valid + self.invalid
        invalid_pct = 100 * self.invalid / total if total else 0
        cur = 'none' if current is None else f'{current:6.1f}'
        lo = f'{self.min:6.1f}' if self.valid else '     -'
        hi = f'{self.max:6.1f}' if self.valid else '     -'
        return (f'range {cur} cm | var {var:8.2f} (sd {math.sqrt(var):5.2f}) | '
                f'min {lo} max {hi} | invalid {invalid_pct:4.1f}% of {total}')


def evaluate(i2c, n, args):
    """Range continuously until the sensor disappears from the bus. Returns final stats."""
    rangefinder = VL53L1X(i2c)
    rangefinder.distance_mode = args.mode
    rangefinder.timing_budget = args.budget
    rangefinder.start_ranging()
    print(f'{CLEAR_LINE}Rangefinder #{n} online.')

    stats = Stats(args.window)
    distance = None
    last_data = time.time()
    try:
        while True:
            if rangefinder.data_ready:
                distance = rangefinder.distance
                rangefinder.clear_interrupt()
                stats.add(distance)
                last_data = time.time()
                print(f'{CLEAR_LINE}#{n} {stats.summary(distance)}', end='', flush=True)
            elif time.time() - last_data > STALL_TIMEOUT:
                break
            time.sleep(0.005)
    except OSError:
        pass # unplugged mid-transaction
    return stats, distance


def main():
    parser = argparse.ArgumentParser(description='Live readout for hot swapped VL53L1X rangefinders.')
    parser.add_argument('--mode', type=int, choices=[1, 2], default=2,
        help='distance mode, 1=short 2=long (default 2, the mode the gripper uses)')
    parser.add_argument('--budget', type=int, choices=[15, 20, 33, 50, 100, 200, 500], default=50,
        help='timing budget in ms (default 50)')
    parser.add_argument('--window', type=int, default=40,
        help='number of recent valid readings the variance is computed over (default 40)')
    args = parser.parse_args()

    i2c = busio.I2C(board.SCL, board.SDA)
    n = 0
    try:
        while True:
            wait_for_rangefinder(i2c)
            n += 1
            try:
                stats, distance = evaluate(i2c, n, args)
            except (OSError, RuntimeError) as e:
                print(f'{CLEAR_LINE}Rangefinder #{n} failed to initialize: {e}')
                time.sleep(POLL_INTERVAL)
                continue
            # leave a permanent line per sensor so earlier results stay on screen
            print(f'{CLEAR_LINE}#{n} offline. final: {stats.summary(distance)}')
    except KeyboardInterrupt:
        print()


if __name__ == '__main__':
    main()
