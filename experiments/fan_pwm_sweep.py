#!/usr/bin/env python3
"""Exercise a fan on a GPIO pin by sweeping its PWM duty cycle up and down.

Standalone bench tool: drives a linear triangle wave, min -> max -> min duty
(0 -> 1 -> 0 by default), over a fixed period (10 s by default) until interrupted, printing the commanded duty cycle
alongside the RPM measured from the fan's tachometer, if there is one.

The tach line (pin 3 of a 4-wire fan) is open-collector and pulses twice per
revolution on most fans. It gets the Pi's internal pull-up to 3.3 V here; never
pull it up to the fan's 12 V/5 V rail, which would put that voltage on the GPIO.
RPM is averaged over a sliding window of TACH_WINDOW_S, so it lags the duty
cycle by about half that.

GPIO 12 and 18 are both PWM0 on the Pi, and on the Stringman image the `pwm`
overlay hands PWM0 to the kernel, so gpiozero cannot claim either pin. The fan is
driven from the hardware PWM block through sysfs instead, the same way
stringman-pilot-rpi-image/fan_control.py does it, muxed onto --pin with pinctrl.
That needs root, and the fan-control service must be stopped or it will fight
this script over the duty cycle:

    sudo systemctl stop fan-control
    sudo /opt/robot/env/bin/python experiments/fan_pwm_sweep.py --pin 18 --frequency 100

The pin is left driving full duty on exit. `sudo systemctl start fan-control`
afterwards puts the normal temperature control back.

Usage (as root, see above):
    fan_pwm_sweep.py --tach-pin 16                       # PWM GPIO 12, tach GPIO 16
    fan_pwm_sweep.py --pin 18 --period 4 --frequency 200 # gripper hat, no tach
    fan_pwm_sweep.py --pin 18 --min-duty 0.2 --max-duty 0.6
    fan_pwm_sweep.py --pin 18 --frequency 50 --min-duty 0.4 --max-duty 0.4  # hold one duty

The tach needs gpiozero (with an lgpio or RPi.GPIO backend), which the robot venv
at /opt/robot/env has.
"""

import argparse
import collections
import os
import subprocess
import sys
import time

TACH_WINDOW_S = 1.0

PWM_CHANNEL_DIR = "/sys/class/pwm/pwmchip0/pwm0"
# GPIO -> pinctrl alt function that carries PWM0 on it.
PWM0_ALT = {12: "a0", 18: "a5"}


class HardwarePwm:
    """PWM0 through sysfs, muxed onto one GPIO with pinctrl."""

    def __init__(self, pin, frequency):
        if not os.path.isdir(PWM_CHANNEL_DIR):
            with open(os.path.dirname(PWM_CHANNEL_DIR) + "/export", "w") as f:
                f.write("0")
            time.sleep(0.5)  # udev creates the attribute files a moment after export
        self.period_ns = round(1e9 / frequency)
        # duty_cycle may never exceed period, so clear it before changing period.
        self._write("duty_cycle", 0)
        self._write("period", self.period_ns)
        self._write("enable", 1)
        for other in PWM0_ALT:
            if other != pin:
                subprocess.run(["pinctrl", "set", str(other), "ip"], check=True)
        subprocess.run(["pinctrl", "set", str(pin), PWM0_ALT[pin]], check=True)

    def _write(self, name, value):
        with open(os.path.join(PWM_CHANNEL_DIR, name), "w") as f:
            f.write(str(value))

    @property
    def value(self):
        raise AttributeError("write-only")

    @value.setter
    def value(self, duty):
        self._write("duty_cycle", round(min(max(duty, 0.0), 1.0) * self.period_ns))


def triangle(phase):
    """0 -> 1 over the first half of the phase, 1 -> 0 over the second."""
    return 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pin", type=int, default=12, choices=sorted(PWM0_ALT),
                        help="BCM pin number of the fan PWM line (default 12)")
    parser.add_argument(
        "--period", type=float, default=10.0,
        help="seconds for one full 0->1->0 sweep (default 10)",
    )
    parser.add_argument(
        "--frequency", type=float, default=200.0,
        help="PWM carrier frequency in Hz (default 200)",
    )
    parser.add_argument(
        "--min-duty", type=float, default=0.0,
        help="duty cycle at the bottom of the sweep, 0-1 (default 0)",
    )
    parser.add_argument(
        "--max-duty", type=float, default=1.0,
        help="duty cycle at the top of the sweep, 0-1 (default 1)",
    )
    parser.add_argument(
        "--step", type=float, default=0.02,
        help="seconds between duty cycle updates (default 0.02)",
    )
    parser.add_argument(
        "--tach-pin", type=int,
        help="BCM pin of the fan tach line, e.g. 16. Omit for fans without a tach",
    )
    parser.add_argument(
        "--pulses-per-rev", type=int, default=2,
        help="tach pulses per fan revolution (default 2, the 4-wire fan standard)",
    )
    args = parser.parse_args()

    if args.period <= 0:
        parser.error("--period must be positive")
    if not 0.0 <= args.min_duty <= args.max_duty <= 1.0:
        parser.error("need 0 <= --min-duty <= --max-duty <= 1")
    if os.geteuid() != 0:
        sys.exit("needs root to drive the hardware PWM: sudo /opt/robot/env/bin/python " + sys.argv[0])

    print(
        f"sweeping GPIO {args.pin} at {args.frequency:g} Hz, "
        f"{args.period:g} s per sweep between duty {args.min_duty:g} and {args.max_duty:g}, "
        + (f"tach on GPIO {args.tach_pin}." if args.tach_pin is not None else "no tach.")
        + " ctrl-c to stop."
    )

    # Tach is open-collector, so with the pull-up each pulse is a falling edge,
    # which gpiozero reports as "activated" on a pull-up input.
    pulses = 0

    def on_pulse():
        nonlocal pulses
        pulses += 1

    tach = None
    if args.tach_pin is not None:
        from gpiozero import DigitalInputDevice
        tach = DigitalInputDevice(args.tach_pin, pull_up=True)
        tach.when_activated = on_pulse

    fan = HardwarePwm(args.pin, args.frequency)
    fan.value = args.min_duty
    start = time.monotonic()
    history = collections.deque([(start, 0)])
    try:
        while True:
            now = time.monotonic()
            elapsed = now - start
            sweep = triangle((elapsed % args.period) / args.period)
            duty = args.min_duty + (args.max_duty - args.min_duty) * sweep
            fan.value = duty

            history.append((now, pulses))
            while now - history[0][0] > TACH_WINDOW_S:
                history.popleft()
            t0, p0 = history[0]
            rpm = (pulses - p0) / args.pulses_per_rev / (now - t0) * 60 if now > t0 else 0.0

            rpm_s = f"  rpm={rpm:6.0f}" if tach is not None else ""
            print(f"\rt={elapsed:7.2f}s  duty={duty:5.3f}{rpm_s}  ", end="", flush=True)
            time.sleep(args.step)
    except KeyboardInterrupt:
        print()
    finally:
        # Full rather than off: with fan-control stopped, nothing else will cool the Pi.
        fan.value = 1.0
        if tach is not None:
            tach.close()
        print("fan left at full duty. sudo systemctl start fan-control to restore it.")


if __name__ == "__main__":
    main()
