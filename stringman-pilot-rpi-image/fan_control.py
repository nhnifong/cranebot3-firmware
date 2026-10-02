#!/usr/bin/env python3
"""Temperature-controlled cooling fan for Stringman components.

Drives the fan from the hardware PWM block through the kernel's sysfs PWM
interface, so the carrier is steady and costs no CPU. config.txt routes PWM0 to
the right pin per board with the `pwm` overlay:

    Raspberry Pi Zero 2 W  -> GPIO 18
    Raspberry Pi 3 A+      -> GPIO 12

Both pins are PWM0, so this daemon always drives pwmchip0 channel 0; the board
check is there to log which pin is in use and to refuse to run on a board the
overlay was not set up for.

Duty cycle follows SoC temperature linearly between MIN_DUTY at or below
TEMP_LOW_C and MAX_DUTY at or above TEMP_HIGH_C. If the temperature can't be
read, or the daemon is stopped, the fan is left at MAX_DUTY: a fan stuck on full
is loud, a fan stuck off cooks the unit.

To run it by hand:

    sudo stringman-pilot-rpi-image/fan_control.py --verbose
"""

import argparse
import os
import signal
import sys
import time

MIN_DUTY = 0.5
MAX_DUTY = 1.0
TEMP_LOW_C = 35.0   # at or below: MIN_DUTY
TEMP_HIGH_C = 48.0  # at or above: MAX_DUTY

PWM_FREQUENCY_HZ = 25000  # Intel 4-wire fan spec
DEFAULT_INTERVAL = 2.0    # seconds between temperature samples

PWM_CHIP = "/sys/class/pwm/pwmchip0"
PWM_CHANNEL = 0
THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"
DT_MODEL = "/proc/device-tree/model"

# Substring of /proc/device-tree/model -> GPIO the overlay puts PWM0 on.
BOARD_PINS = {
    "Raspberry Pi Zero 2 W": 18,
    "Raspberry Pi 3 Model A Plus": 12,
}


def log(message):
    print(message, flush=True)


def read_model():
    try:
        with open(DT_MODEL) as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return None


def fan_pin_for(model):
    for name, pin in BOARD_PINS.items():
        if model and name in model:
            return pin
    return None


def read_soc_temp_c():
    """SoC temperature in C, or None if unreadable."""
    try:
        with open(THERMAL_ZONE) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def duty_for_temp(temp_c):
    if temp_c is None or temp_c >= TEMP_HIGH_C:
        return MAX_DUTY
    if temp_c <= TEMP_LOW_C:
        return MIN_DUTY
    frac = (temp_c - TEMP_LOW_C) / (TEMP_HIGH_C - TEMP_LOW_C)
    return MIN_DUTY + frac * (MAX_DUTY - MIN_DUTY)


class SysfsPwm:
    """One channel of a kernel PWM chip, driven through sysfs."""

    def __init__(self, chip, channel, frequency_hz):
        self.path = os.path.join(chip, f"pwm{channel}")
        if not os.path.isdir(chip):
            raise RuntimeError(
                f"{chip} not found -- is the pwm overlay enabled in /boot/firmware/config.txt?"
            )
        if not os.path.isdir(self.path):
            self._write(os.path.join(chip, "export"), channel)
            # udev creates the attribute files a moment after export.
            for _ in range(50):
                if os.access(os.path.join(self.path, "enable"), os.W_OK):
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"{self.path} did not appear after export")

        self.period_ns = round(1e9 / frequency_hz)
        # duty_cycle may never exceed period, so clear it before changing period.
        self._write(os.path.join(self.path, "duty_cycle"), 0)
        self._write(os.path.join(self.path, "period"), self.period_ns)
        self._write(os.path.join(self.path, "enable"), 1)

    @staticmethod
    def _write(path, value):
        with open(path, "w") as f:
            f.write(str(value))

    def set_duty(self, duty):
        duty = min(max(duty, 0.0), 1.0)
        self._write(os.path.join(self.path, "duty_cycle"), round(duty * self.period_ns))


_running = True


def _stop(signum, frame):
    global _running
    _running = False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help=f"seconds between temperature samples (default {DEFAULT_INTERVAL:g})")
    ap.add_argument("--verbose", action="store_true",
                    help="log every sample, not just changes of a few percent")
    ap.add_argument("--frequency", type=float, default=PWM_FREQUENCY_HZ,
                    help=f"PWM carrier frequency in Hz (default {PWM_FREQUENCY_HZ})")
    args = ap.parse_args()

    model = read_model()
    pin = fan_pin_for(model)
    if pin is None:
        sys.exit(f"fan_control: unsupported board {model!r}, no fan pin configured")

    pwm = SysfsPwm(PWM_CHIP, PWM_CHANNEL, args.frequency)
    log(f"fan_control: {model}, fan on GPIO {pin} at {args.frequency:g} Hz, "
        f"duty {MIN_DUTY:g}-{MAX_DUTY:g} over {TEMP_LOW_C:g}-{TEMP_HIGH_C:g}C")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_logged = None
    try:
        while _running:
            temp = read_soc_temp_c()
            duty = duty_for_temp(temp)
            pwm.set_duty(duty)
            if args.verbose or last_logged is None or abs(duty - last_logged) >= 0.05:
                temp_s = f"{temp:.1f}C" if temp is not None else "unreadable"
                log(f"soc_temp={temp_s} duty={duty:.2f}")
                last_logged = duty
            time.sleep(args.interval)
    finally:
        pwm.set_duty(MAX_DUTY)
        log("fan_control: stopping, fan left at full speed")


if __name__ == "__main__":
    main()
