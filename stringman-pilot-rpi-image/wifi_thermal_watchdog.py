#!/usr/bin/env python3
"""Wi-Fi thermal watchdog for Stringman anchors (Raspberry Pi Zero 2 W).

Diagnostic daemon to chase down anchors that drop off the network after a few
hours. The working theory is that the Cypress/Broadcom Wi-Fi chip (brcmfmac,
wlan0) thermally trips around ~60C, well before the BCM2710 SoC throttles or
shuts down. This tool logs the evidence and tries to recover.

Every INTERVAL seconds it:
  1. Reads the SoC temperature and throttle flags (the Wi-Fi chip exposes no
     temperature sensor of its own, so SoC temp is the available proxy).
  2. Checks whether we are still reachable on the LAN (ping the default gateway).
  3. If we are offline, attempts to power-cycle the Wi-Fi chip and logs whether
     each recovery step brought us back.

Everything is appended (with flush + fsync) to a log file on the SD card. The
default is the invoking user's home dir (e.g. /home/pi/wifi_thermal_watchdog.log,
resolved via SUDO_USER so it lands there even under sudo). Point --logfile at
/boot/firmware/... instead if you want a copy you can read by pulling the card.

Recovery escalates from least to most disruptive. A soft radio toggle is NOT
enough to revive a chip that has thermally shut down, so we go straight to
reloading the brcmfmac kernel module, which forces the chip firmware to be
re-downloaded and the chip re-initialized -- the closest thing to a real power
cycle without cutting its rail. If even that fails, reboot the whole unit.

  1. modprobe -r brcmfmac brcmutil; modprobe brcmfmac   (needs root, real reset)
  2. sudo reboot now                                    (last resort, full cycle)

Steps that need root are prefixed with sudo automatically when not running as
root. On the Stringman image this ships as a systemd service that runs as root,
so the module reload and reboot work directly. To run it by hand:

    sudo experiments/wifi_thermal_watchdog.py

To leave it running across an ssh logout:

    sudo nohup experiments/wifi_thermal_watchdog.py >/dev/null 2>&1 &

Then later read ~/wifi_thermal_watchdog.log (or scp it off the Pi).
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

# ---- Defaults (override via CLI) -------------------------------------------
DEFAULT_INTERVAL = 60  # seconds between samples
DEFAULT_IFACE = "wlan0"


def _invoking_home():
    """Home dir of the real user, even when launched via sudo.

    Under `sudo` os.path.expanduser('~') resolves to /root, but we want the
    log to land in the pi user's home so it is easy to find and read.
    """
    user = os.environ.get("SUDO_USER")
    if user:
        try:
            import pwd
            return pwd.getpwnam(user).pw_dir
        except (KeyError, ImportError):
            pass
    return os.path.expanduser("~")


DEFAULT_LOGFILE = os.path.join(_invoking_home(), "wifi_thermal_watchdog.log")
SETTLE_SECONDS = 8  # how long to wait for the link to come back after a step
PING_COUNT = 2
PING_TIMEOUT = 3  # seconds per ping


def now_iso() -> str:
    """Local time, ISO-ish, easy to read off the card."""
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


class Log:
    """Append-only log that fsyncs every line so it survives a hard hang."""

    def __init__(self, path: str):
        self.path = path
        # Make sure the directory exists; fall back to cwd if /boot/firmware
        # is not present (e.g. when testing off-Pi).
        d = os.path.dirname(path) or "."
        if not os.path.isdir(d):
            print(f"warning: {d} does not exist, logging to ./{os.path.basename(path)}",
                  file=sys.stderr)
            self.path = os.path.basename(path)

    def write(self, kind: str, message: str):
        line = f"{now_iso()}\t{kind}\t{message}\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        try:
            with open(self.path, "a") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            # Don't let a logging failure kill the watchdog.
            print(f"warning: could not write log to {self.path}: {e}", file=sys.stderr)


def run(cmd, timeout=15):
    """Run a command, returning (returncode, combined_output). Never raises."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        out = (p.stdout + p.stderr).strip()
        return p.returncode, out
    except subprocess.TimeoutExpired:
        return 124, f"timeout after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return 127, f"not found: {cmd[0]}"
    except Exception as e:  # noqa: BLE001 - watchdog must not die
        return 1, f"error running {' '.join(cmd)}: {e}"


def sudo(cmd):
    """Prefix with sudo -n if we are not already root."""
    if os.geteuid() == 0:
        return cmd
    return ["sudo", "-n", *cmd]


# ---- Telemetry --------------------------------------------------------------

def read_soc_temp_c():
    """SoC temperature in C, or None if unreadable."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        pass
    # Fall back to vcgencmd: "temp=54.0'C"
    rc, out = run(["vcgencmd", "measure_temp"], timeout=5)
    if rc == 0 and "temp=" in out:
        try:
            return float(out.split("temp=")[1].split("'")[0])
        except (IndexError, ValueError):
            return None
    return None


def read_throttle():
    """vcgencmd get_throttled flags, e.g. 'throttled=0x0', or 'n/a'."""
    rc, out = run(["vcgencmd", "get_throttled"], timeout=5)
    if rc == 0 and out:
        return out.strip()
    return "n/a"


def default_gateway(iface):
    """Default-route gateway IP for the wifi interface, or None."""
    rc, out = run(["ip", "route", "show", "default"], timeout=5)
    if rc != 0:
        return None
    # Lines look like: "default via 192.168.1.1 dev wlan0 proto dhcp ..."
    best = None
    for line in out.splitlines():
        parts = line.split()
        if "via" in parts:
            gw = parts[parts.index("via") + 1]
            if iface in parts:
                return gw  # prefer a route that is on our wifi iface
            best = best or gw
    return best


def iface_has_ip(iface):
    rc, out = run(["ip", "-4", "addr", "show", "dev", iface], timeout=5)
    return rc == 0 and "inet " in out


def is_online(iface, target):
    """True if we can reach the LAN. Pings target (gateway) over iface."""
    if target is None:
        # No gateway known; treat presence of an IP as a weak online signal.
        return iface_has_ip(iface)
    rc, _ = run(
        ["ping", "-I", iface, "-c", str(PING_COUNT), "-W", str(PING_TIMEOUT), target],
        timeout=PING_COUNT * PING_TIMEOUT + 5,
    )
    return rc == 0


# ---- Recovery ---------------------------------------------------------------

def recover(log, iface):
    """Escalating attempts to restore wifi. Returns True if back online.

    First reload the brcmfmac kernel module (forces the chip firmware to be
    re-downloaded and the chip re-initialized -- the closest thing to a real
    power cycle without cutting its rail). If that does not bring us back,
    reboot the whole unit: a thermally wedged chip usually needs the full power
    cycle, and an anchor that stays offline is useless anyway.
    """

    def back_online():
        gw = default_gateway(iface)
        return is_online(iface, gw)

    name = "brcmfmac module reload (real power cycle)"
    cmds = [
        sudo(["modprobe", "-r", "brcmfmac", "brcmutil"]),
        ["sleep", "2"],
        sudo(["modprobe", "brcmfmac"]),
    ]
    log.write("RECOVER", f"attempting: {name}")
    for cmd in cmds:
        if cmd[0] == "sleep":
            time.sleep(int(cmd[1]))
            continue
        rc, out = run(cmd, timeout=30)
        detail = f"  $ {' '.join(cmd)} -> rc={rc}"
        if out:
            detail += f" :: {out.splitlines()[0][:200]}"
        log.write("RECOVER", detail)
    # Give NetworkManager / DHCP time to re-associate and lease.
    log.write("RECOVER", f"waiting {SETTLE_SECONDS}s for re-association...")
    time.sleep(SETTLE_SECONDS)
    if back_online():
        log.write("RECOVER", f"SUCCESS via '{name}' -- back online")
        return True

    # Last resort: reboot the unit. reboot returns immediately and the box goes
    # down, so there is nothing to check afterwards.
    log.write("RECOVER", "still offline after module reload -- rebooting unit")
    rc, out = run(sudo(["reboot", "now"]), timeout=30)
    detail = f"  $ reboot now -> rc={rc}"
    if out:
        detail += f" :: {out.splitlines()[0][:200]}"
    log.write("RECOVER", detail)
    return False


# ---- Main loop --------------------------------------------------------------

_running = True


def _stop(signum, frame):
    global _running
    _running = False


def sample(log, iface):
    temp = read_soc_temp_c()
    throttle = read_throttle()
    gw = default_gateway(iface)
    online = is_online(iface, gw)

    temp_s = f"{temp:.1f}C" if temp is not None else "n/a"
    log.write(
        "SAMPLE",
        f"soc_temp={temp_s} throttled={throttle} iface={iface} "
        f"gateway={gw or 'none'} online={'yes' if online else 'NO'}",
    )

    if not online:
        log.write("OFFLINE", f"link down at soc_temp={temp_s} -- starting recovery")
        recover(log, iface)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                    help=f"seconds between samples (default {DEFAULT_INTERVAL})")
    ap.add_argument("--logfile", default=DEFAULT_LOGFILE,
                    help=f"log path on the SD card (default {DEFAULT_LOGFILE})")
    ap.add_argument("--iface", default=DEFAULT_IFACE,
                    help=f"wifi interface (default {DEFAULT_IFACE})")
    ap.add_argument("--once", action="store_true",
                    help="take a single sample and exit (for testing)")
    args = ap.parse_args()

    log = Log(args.logfile)
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if os.geteuid() != 0:
        log.write("START", "WARNING: not running as root; module reload and ip "
                           "link recovery need sudo -n and may fail")

    log.write("START", f"wifi_thermal_watchdog up: interval={args.interval}s "
                       f"iface={args.iface} logfile={log.path}")

    if args.once:
        sample(log, args.iface)
        return

    while _running:
        try:
            sample(log, args.iface)
        except Exception as e:  # noqa: BLE001 - the watchdog must outlive errors
            log.write("ERROR", f"sample failed: {e}")
        # Sleep in short slices so SIGTERM is responsive.
        slept = 0
        while _running and slept < args.interval:
            time.sleep(min(5, args.interval - slept))
            slept += 5

    log.write("STOP", "watchdog shutting down")


if __name__ == "__main__":
    main()
