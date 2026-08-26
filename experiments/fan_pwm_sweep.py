#!/usr/bin/env python3
"""Exercise a fan on a GPIO pin by sweeping its PWM duty cycle up and down.

Standalone bench tool: drives a linear triangle wave, 0 -> 1 -> 0, over a fixed
period (10 s by default) until interrupted, printing the commanded duty cycle so
you can watch it against what the fan actually does.

GPIO 12 is PWM0 on the Pi, so the pin can be driven by the hardware PWM block.
gpiozero's default pin factory still bit-bangs it from the CPU, which is fine for
a fan but hums unevenly under load; for a rock-steady carrier, run pigpiod and
start this with GPIOZERO_PIN_FACTORY=pigpio.

Beware the UART pins: GPIO 14/15 are TXD0/RXD0 and
stringman-pilot-rpi-image/config.txt sets enable_uart=1 for the MKSSERVO42C link,
so PWM on those pins does nothing useful. The script warns if you point it there.

Usage:
    python3 experiments/fan_pwm_sweep.py                 # GPIO 12, 10 s period
    python3 experiments/fan_pwm_sweep.py --pin 18 --period 4 --frequency 25000

Needs gpiozero (with an lgpio or RPi.GPIO backend):
    pip install gpiozero lgpio
"""

import argparse
import os
import sys
import time

try:
    from gpiozero import PWMOutputDevice
except ImportError:
    sys.exit(
        "gpiozero is not installed. On the Pi:\n"
        "    pip install gpiozero lgpio\n"
        "or run this with the system python, which ships it on Raspberry Pi OS."
    )

CONFIG_TXT = "/boot/firmware/config.txt"


def warn_if_uart_owns_pin(pin):
    """GPIO 14/15 are the primary UART. Warn rather than refuse -- the overlay
    may have moved the console elsewhere, and only the user can say for sure."""
    if pin not in (14, 15):
        return
    try:
        with open(CONFIG_TXT) as f:
            enabled = any(
                line.strip().startswith("enable_uart=1") for line in f
            )
    except OSError:
        return
    if enabled:
        print(
            f"warning: {CONFIG_TXT} has enable_uart=1, and GPIO {pin} is a UART pin "
            "(TXD0/RXD0). If the serial port is active the fan will not respond.",
            file=sys.stderr,
        )


def triangle(phase):
    """0 -> 1 over the first half of the phase, 1 -> 0 over the second."""
    return 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pin", type=int, default=12, help="BCM pin number (default 12)")
    parser.add_argument(
        "--period", type=float, default=10.0,
        help="seconds for one full 0->1->0 sweep (default 10)",
    )
    parser.add_argument(
        "--frequency", type=float, default=100.0,
        help="PWM carrier frequency in Hz (default 100)",
    )
    parser.add_argument(
        "--step", type=float, default=0.02,
        help="seconds between duty cycle updates (default 0.02)",
    )
    args = parser.parse_args()

    if args.period <= 0:
        parser.error("--period must be positive")

    warn_if_uart_owns_pin(args.pin)

    print(
        f"sweeping GPIO {args.pin} at {args.frequency:g} Hz, "
        f"{args.period:g} s per sweep. ctrl-c to stop."
    )

    fan = PWMOutputDevice(args.pin, frequency=args.frequency, initial_value=0.0)
    start = time.monotonic()
    try:
        while True:
            elapsed = time.monotonic() - start
            duty = triangle((elapsed % args.period) / args.period)
            fan.value = duty
            print(f"\rt={elapsed:7.2f}s  duty={duty:5.3f}  ", end="", flush=True)
            time.sleep(args.step)
    except KeyboardInterrupt:
        print()
    finally:
        fan.value = 0.0
        fan.close()
        print("fan off.")


if __name__ == "__main__":
    main()
