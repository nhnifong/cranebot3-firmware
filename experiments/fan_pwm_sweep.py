#!/usr/bin/env python3
"""Exercise a fan on a GPIO pin by sweeping its PWM duty cycle up and down.

Standalone bench tool: drives a linear triangle wave, min -> max -> min duty
(0 -> 1 -> 0 by default), over a fixed period (10 s by default) until interrupted, printing the commanded duty cycle
alongside the RPM measured from the fan's tachometer.

The tach line (pin 3 of a 4-wire fan) is open-collector and pulses twice per
revolution on most fans. It gets the Pi's internal pull-up to 3.3 V here; never
pull it up to the fan's 12 V/5 V rail, which would put that voltage on the GPIO.
RPM is averaged over a sliding window of TACH_WINDOW_S, so it lags the duty
cycle by about half that.

GPIO 12 is PWM0 on the Pi, so the pin can be driven by the hardware PWM block.
gpiozero's default pin factory still bit-bangs it from the CPU, which is fine for
a fan but hums unevenly under load; for a rock-steady carrier, run pigpiod and
start this with GPIOZERO_PIN_FACTORY=pigpio.

Beware the UART pins: GPIO 14/15 are TXD0/RXD0 and
stringman-pilot-rpi-image/config.txt sets enable_uart=1 for the MKSSERVO42C link,
so PWM on those pins does nothing useful. The script warns if you point it there.

Usage:
    python3 experiments/fan_pwm_sweep.py                 # PWM GPIO 12, tach GPIO 16
    python3 experiments/fan_pwm_sweep.py --pin 18 --period 4 --frequency 25000
    python3 experiments/fan_pwm_sweep.py --min-duty 0.2 --max-duty 0.6

Needs gpiozero (with an lgpio or RPi.GPIO backend):
    pip install gpiozero lgpio
"""

import argparse
import collections
import sys
import time

try:
    from gpiozero import DigitalInputDevice, PWMOutputDevice
except ImportError:
    sys.exit(
        "gpiozero is not installed. On the Pi:\n"
        "    pip install gpiozero lgpio\n"
        "or run this with the system python, which ships it on Raspberry Pi OS."
    )

CONFIG_TXT = "/boot/firmware/config.txt"
TACH_WINDOW_S = 1.0


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
        "--tach-pin", type=int, default=16, help="BCM pin of the fan tach line (default 16)",
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

    warn_if_uart_owns_pin(args.pin)

    print(
        f"sweeping GPIO {args.pin} at {args.frequency:g} Hz, "
        f"{args.period:g} s per sweep between duty {args.min_duty:g} and {args.max_duty:g}, "
        f"tach on GPIO {args.tach_pin}. ctrl-c to stop."
    )

    # Tach is open-collector, so with the pull-up each pulse is a falling edge,
    # which gpiozero reports as "activated" on a pull-up input.
    pulses = 0

    def on_pulse():
        nonlocal pulses
        pulses += 1

    tach = DigitalInputDevice(args.tach_pin, pull_up=True)
    tach.when_activated = on_pulse

    fan = PWMOutputDevice(args.pin, frequency=args.frequency, initial_value=args.min_duty)
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

            print(
                f"\rt={elapsed:7.2f}s  duty={duty:5.3f}  rpm={rpm:6.0f}  ",
                end="", flush=True,
            )
            time.sleep(args.step)
    except KeyboardInterrupt:
        print()
    finally:
        fan.value = 0.0
        fan.close()
        tach.close()
        print("fan off.")


if __name__ == "__main__":
    main()
