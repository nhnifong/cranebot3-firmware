#!/usr/bin/env python3
"""Measure ambient noise level with the Blue Snowball USB mic.

Records a short window through arecord and prints its RMS level in dBFS, both
unweighted and A-weighted. A-weighting follows how loud the ear finds a sound,
so it is the better number for comparing fan settings by how annoying they are;
the unweighted level is dominated by low-frequency rumble the ear barely hears.

The mic is not calibrated, so these are not SPL readings: compare levels taken
with the same mic, gain and placement. Higher (closer to 0) is louder.

The first DISCARD_S of every recording is dropped, because USB mics often start
with a click or a settling transient that would swamp a quiet room.

Usage:
    python3 experiments/mic_level.py               # one 1 s measurement
    python3 experiments/mic_level.py --repeat 10   # ten in a row
    python3 experiments/mic_level.py --device plughw:CARD=Snowball --seconds 2
"""

import argparse
import subprocess
import sys

import numpy as np

DEFAULT_DEVICE = "plughw:CARD=Snowball"
SAMPLE_RATE = 48000
DISCARD_S = 0.25
FULL_SCALE = 32768.0  # S16_LE


def record(device, seconds):
    """Mono float samples in [-1, 1) for `seconds`, after dropping the startup transient."""
    discard = int(DISCARD_S * SAMPLE_RATE)
    n = int(seconds * SAMPLE_RATE) + discard
    p = subprocess.run(
        ["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
         "-c", "1", "-t", "raw", "-s", str(n)],
        capture_output=True,
    )
    if p.returncode != 0:
        sys.exit(f"arecord failed: {p.stderr.decode().strip()}")
    samples = np.frombuffer(p.stdout, dtype="<i2").astype(np.float64) / FULL_SCALE
    return samples[discard:]


def a_weighting(freqs):
    """IEC 61672 A-weighting gain (linear) at each frequency in Hz."""
    f2 = freqs ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    return ra * 10 ** (2.0 / 20)  # normalize to 0 dB at 1 kHz


def db(rms):
    return 20 * np.log10(max(rms, 1e-12))


def levels(samples):
    """(unweighted dBFS, A-weighted dBFS) of the samples, with DC removed."""
    x = samples - samples.mean()
    rms = np.sqrt(np.mean(x ** 2))
    # Weight in the frequency domain. Parseval: mean power = sum |X|^2 / N^2, with
    # the one-sided bins doubled except DC and Nyquist.
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
    power = np.abs(spectrum * a_weighting(freqs)) ** 2
    power[1:-1] *= 2
    rms_a = np.sqrt(power.sum()) / len(x)
    return db(rms), db(rms_a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=DEFAULT_DEVICE,
                    help=f"ALSA capture device (default {DEFAULT_DEVICE})")
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="length of each measurement (default 1)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="number of measurements to take (default 1)")
    args = ap.parse_args()

    for _ in range(args.repeat):
        samples = record(args.device, args.seconds)
        level, level_a = levels(samples)
        peak = db(np.abs(samples).max())
        print(f"{level:6.1f} dBFS  {level_a:6.1f} dBFS(A)  peak {peak:6.1f} dBFS")


if __name__ == "__main__":
    main()
