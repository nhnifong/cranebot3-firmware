#!/usr/bin/env python3
"""Temperature-controlled cooling fan for Stringman components.

Drives the fan from the hardware PWM block through the kernel's sysfs PWM
interface, so the carrier is steady and costs no CPU. The fan pin depends only
on the hat:

    anchor hat -> GPIO 12
    otherwise  -> GPIO 18

The anchor hat is recognized by its MCP2515 CAN controller: can0 only exists when
that chip answers on SPI. Both pins can carry PWM0, so this daemon always drives
pwmchip0 channel 0 and muxes PWM0 onto the chosen pin itself with pinctrl,
returning the other one to an input. The `pwm` overlay in config.txt is still
needed to enable the PWM block, but the pin it picks is only a boot default.

can0 appears some seconds into boot, possibly after this service starts, so the
pin is re-checked every sample and moved if the answer changes. Until the pin is
muxed the fan's PWM line floats high, which a 4-wire fan treats as full speed.

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
import subprocess
import time

MIN_DUTY = 0.3
MAX_DUTY = 1.0
TEMP_LOW_C = 35.0   # at or below: MIN_DUTY
TEMP_HIGH_C = 50.0  # at or above: MAX_DUTY

PWM_FREQUENCY_HZ = 25000  # Intel 4-wire fan spec
DEFAULT_INTERVAL = 2.0    # seconds between temperature samples

PWM_CHIP = "/sys/class/pwm/pwmchip0"
PWM_CHANNEL = 0
THERMAL_ZONE = "/sys/class/thermal/thermal_zone0/temp"
CAN_DEVICE = "/sys/class/net/can0"

# GPIO -> pinctrl alt function that carries PWM0 on it.
PWM0_ALT = {12: "a0", 18: "a5"}


def log(message):
    print(message, flush=True)


def fan_pin():
    """GPIO the fan's PWM line is wired to: 12 on the anchor hat, else 18."""
    return 12 if os.path.exists(CAN_DEVICE) else 18


def route_pwm0_to(pin):
    """Mux PWM0 onto pin and return the other PWM0-capable pin to an input."""
    for other in PWM0_ALT:
        if other != pin:
            subprocess.run(["pinctrl", "set", str(other), "ip"], check=True)
    subprocess.run(["pinctrl", "set", str(pin), PWM0_ALT[pin]], check=True)


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

    pwm = SysfsPwm(PWM_CHIP, PWM_CHANNEL, args.frequency)
    log(f"fan_control: {args.frequency:g} Hz, "
        f"duty {MIN_DUTY:g}-{MAX_DUTY:g} over {TEMP_LOW_C:g}-{TEMP_HIGH_C:g}C")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_logged = None
    routed_pin = None
    try:
        while _running:
            pin = fan_pin()
            if pin != routed_pin:
                route_pwm0_to(pin)
                log(f"fan_control: fan PWM on GPIO {pin}")
                routed_pin = pin
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
