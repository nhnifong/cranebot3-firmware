#!/usr/bin/env python3
"""Wi-Fi thermal watchdog for Stringman anchors (Raspberry Pi Zero 2 W).

Diagnostic daemon to chase down anchors that drop off the network after a few
hours. The working theory is that the Cypress/Broadcom Wi-Fi chip (brcmfmac,
wlan0) thermally trips around ~60C, well before the BCM2710 SoC throttles or
shuts down. This tool logs the evidence and tries to recover.

Every INTERVAL seconds it:
  1. Records diagnostics: uptime, load, MemAvailable, component-server RSS,
     SoC temp (proxy for the Wi-Fi chip, which has no sensor of its own),
     wlan operstate, Wi-Fi signal, ssh/:8765 listen counts, can0 state,
     throttle flags, IP, gateway.
  2. Checks whether we are still reachable on the LAN (ping the default gateway).
  3. If we are offline, attempts to power-cycle the Wi-Fi chip and logs whether
     each recovery step brought us back. Recovery is skipped (and the reason
     logged) within the first MIN_UPTIME_S seconds of boot, when Wi-Fi may simply
     not have associated yet; on any unit with a wired ethernet link, which is
     reachable regardless of what Wi-Fi is doing; and when no saved Wi-Fi network
     exists, since a unit with nothing to associate with would otherwise reboot
     forever.

Wired units are diagnostic-only: Wi-Fi status is still sampled and logged, but
nothing is ever reloaded or rebooted on their behalf. A gripper on a USB ethernet
adapter has no Wi-Fi profile to associate with, so every recovery attempt would
fail and escalate to a reboot, forever.

Everything is appended (with flush + fsync) to a log file on the SD card,
/opt/robot/wifi_thermal_watchdog.log by default. Point --logfile at
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

Then later read /opt/robot/wifi_thermal_watchdog.log (or scp it off the Pi).
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


DEFAULT_LOGFILE = "/opt/robot/wifi_thermal_watchdog.log"
SETTLE_SECONDS = 30  # how long to wait for the link to come back after a step
PING_COUNT = 4
PING_TIMEOUT = 5  # seconds per ping

# wait at least this long after boot before allowing wifi recovery to happen
# First boot takes ~40s to get online if the qr code is visible, second boot takes 10.
MIN_UPTIME_S = 100
NM_CONNECTION_DIR = "/etc/NetworkManager/system-connections"
SYS_NET_DIR = "/sys/class/net"
ARPHRD_ETHER = "1"  # /sys/class/net/<dev>/type of a wired ethernet link (can0 is 280)


def now_iso() -> str:
    """UTC, ISO-ish, easy to read off the card."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


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


def read_uptime_s():
    """System uptime in seconds, or None."""
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_loadavg():
    """1/5/15-minute load averages as '0.12/0.30/0.25', or 'n/a'."""
    try:
        with open("/proc/loadavg") as f:
            return "/".join(f.read().split()[:3])
    except OSError:
        return "n/a"


def read_mem_available_mb():
    """MemAvailable from /proc/meminfo in MB, or None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_component_server_rss_mb():
    """RSS of the component-server process in MB, or None if not running."""
    try:
        pids = os.listdir("/proc")
    except OSError:
        return None
    for pid in pids:
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"component-server" not in f.read():
                    continue
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            continue
    return None


def read_operstate(dev):
    """Kernel operstate of a network device ('up', 'down', ...), or 'none'."""
    try:
        with open(os.path.join(SYS_NET_DIR, dev, "operstate")) as f:
            return f.read().strip()
    except OSError:
        return "none"


def read_carrier(dev):
    """Kernel carrier flag of a network device ('1' when the link has carrier).

    Returns 'none' when unreadable; reading carrier on a down interface raises
    EINVAL, which is not an error worth distinguishing here.
    """
    try:
        with open(os.path.join(SYS_NET_DIR, dev, "carrier")) as f:
            return f.read().strip()
    except OSError:
        return "none"


def read_wifi_signal_dbm(iface):
    """Signal level from /proc/net/wireless in dBm, or None."""
    try:
        with open("/proc/net/wireless") as f:
            for line in f:
                if line.strip().startswith(iface + ":"):
                    return int(float(line.split()[3]))
    except (OSError, ValueError, IndexError):
        pass
    return None


def listen_counts():
    """Count of LISTEN sockets on :22 (ssh) and :8765 (component server)."""
    rc, out = run(["ss", "-tlnH"], timeout=5)
    ssh = srv = 0
    if rc == 0:
        for line in out.splitlines():
            parts = line.split()
            addr = parts[3] if len(parts) > 3 else ""
            if addr.endswith(":22"):
                ssh += 1
            elif addr.endswith(":8765"):
                srv += 1
    return ssh, srv


def default_gateway(iface):
    """Default-route gateway IP reached *via* iface, or None.

    Only routes whose `dev` is iface count. Falling back to some other
    interface's gateway is worse than returning None: the caller pings it with
    `-I <iface>`, so a wired gateway returned here is unreachable over an
    unassociated wlan0 and reads as a Wi-Fi failure that recovery cannot fix.
    """
    rc, out = run(["ip", "route", "show", "default"], timeout=5)
    if rc != 0:
        return None
    # Lines look like: "default via 192.168.1.1 dev wlan0 proto dhcp ..."
    for line in out.splitlines():
        parts = line.split()
        if "via" not in parts or "dev" not in parts:
            continue
        if parts[parts.index("dev") + 1] != iface:
            continue
        return parts[parts.index("via") + 1]
    return None


def iface_ip(iface):
    """IPv4 address on iface, or None."""
    rc, out = run(["ip", "-4", "addr", "show", "dev", iface], timeout=5)
    if rc != 0:
        return None
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0] == "inet":
            return parts[1].split("/")[0]
    return None


def is_online(iface, target):
    """True if we can reach the LAN. Pings target (gateway) over iface."""
    if target is None:
        # No gateway known; treat presence of an IP as a weak online signal.
        return iface_ip(iface) is not None
    rc, _ = run(
        ["ping", "-I", iface, "-c", str(PING_COUNT), "-W", str(PING_TIMEOUT), target],
        timeout=PING_COUNT * PING_TIMEOUT + 5,
    )
    return rc == 0


# ---- Recovery ---------------------------------------------------------------

# Latched once any wired link has been seen up. A USB ethernet adapter can read
# as down for a sample or two while it re-enumerates, and a single such sample is
# not a reason to start rebooting a unit that is plainly wired.
_wired_ever_seen = False


def ethernet_links_up():
    """Names of wired ethernet interfaces that are up.

    Filters on /sys/class/net/<dev>/type so can0 (280) and lo (772) don't count,
    and on the absence of a `wireless` directory so wlan0 - which is also type 1 -
    doesn't count itself. `carrier` counts as up alongside `operstate`, since some
    USB ethernet drivers leave operstate at "unknown" while the link is live.
    """
    try:
        names = sorted(os.listdir(SYS_NET_DIR))
    except OSError:
        return []

    up = []
    for name in names:
        path = os.path.join(SYS_NET_DIR, name)
        if os.path.isdir(os.path.join(path, "wireless")):
            continue
        try:
            with open(os.path.join(path, "type")) as f:
                if f.read().strip() != ARPHRD_ETHER:
                    continue
        except OSError:
            continue
        if read_operstate(name) == "up" or read_carrier(name) == "1":
            up.append(name)
    return up


def wired_link_present():
    """(is_wired, description) - whether this unit is reachable over ethernet.

    Latches: once a wired link has been seen up, this unit stays classified as
    wired for the life of the process.
    """
    global _wired_ever_seen
    up = ethernet_links_up()
    if up:
        _wired_ever_seen = True
        return True, ", ".join(up)
    if _wired_ever_seen:
        return True, "seen earlier this boot"
    return False, ""


def saved_wifi_networks():
    """Count of saved Wi-Fi connection profiles, or None if it can't be told.

    nmcli is authoritative; the profile directory is the fallback for when nmcli
    is missing or NetworkManager isn't answering yet.
    """
    rc, out = run(["nmcli", "-t", "-f", "TYPE", "connection", "show"], timeout=10)
    if rc == 0:
        return sum(1 for line in out.splitlines() if line.strip() == "802-11-wireless")

    try:
        return sum(1 for name in os.listdir(NM_CONNECTION_DIR) if name.endswith(".nmconnection"))
    except OSError:
        return None


def recovery_blocked_reason(uptime_s):
    """Why recovery should be skipped for this sample, or None to go ahead.

    Cases where being offline is not something recovery can fix, and where
    escalating to a reboot would just loop:
      - a wired link, so the unit is reachable no matter what Wi-Fi is doing.
        Checked first and unconditionally: on a wired unit we only ever observe
        and log Wi-Fi, never act on it.
      - too soon after boot, when Wi-Fi has simply not associated yet
      - no saved Wi-Fi network, so the unit has nothing to associate with
    Unknown uptime or an unreadable profile list is not treated as blocking; a
    watchdog that disables itself on missing diagnostics is worse than one that
    retries.
    """
    wired, how = wired_link_present()
    if wired:
        return f"ethernet is up ({how}) -- wired units are observe-only"

    if uptime_s is not None and uptime_s < MIN_UPTIME_S:
        return f"only {uptime_s:.0f}s since boot (< {MIN_UPTIME_S}s)"

    saved = saved_wifi_networks()
    if saved == 0:
        return "no saved Wi-Fi networks to connect to"
    return None


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
    # NetworkManager and wpa_supplicant hold wlan0 open, which pins the module:
    # a bare `modprobe -r` fails with "Module brcmfmac is in use" every time and
    # the reload silently degrades into an unconditional reboot. Turn the radio
    # off first so NM tears the interface down and the supplicant lets go, then
    # turn it back on afterwards so NM re-activates the saved profile.
    cmds = [
        sudo(["nmcli", "radio", "wifi", "off"]),
        ["sleep", "2"],
        sudo(["ip", "link", "set", iface, "down"]),
        sudo(["modprobe", "-r", "brcmfmac", "brcmutil"]),
        ["sleep", "2"],
        sudo(["modprobe", "brcmfmac"]),
        ["sleep", "2"],
        sudo(["nmcli", "radio", "wifi", "on"]),
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
    up = read_uptime_s()
    mem = read_mem_available_mb()
    rss = read_component_server_rss_mb()
    signal_dbm = read_wifi_signal_dbm(iface)
    ip = iface_ip(iface)
    ssh_n, srv_n = listen_counts()

    temp_s = f"{temp:.1f}C" if temp is not None else "n/a"
    up_s = f"{up:.0f}s" if up is not None else "n/a"
    mem_s = f"{mem}MB" if mem is not None else "n/a"
    rss_s = f"{rss}MB" if rss is not None else "n/a"
    signal_s = f"{signal_dbm}dBm" if signal_dbm is not None else "n/a"
    log.write(
        "SAMPLE",
        f"up={up_s} load={read_loadavg()} mem_avail={mem_s} rss={rss_s} "
        f"soc_temp={temp_s} iface={iface} oper={read_operstate(iface)} "
        f"signal={signal_s} listen_ssh={ssh_n} listen_8765={srv_n} "
        f"can0={read_operstate('can0')} throttled={throttle} "
        f"ip={ip or 'none'} gateway={gw or 'none'} online={'yes' if online else 'NO'}",
    )

    if not online:
        blocked = recovery_blocked_reason(up)
        if blocked:
            log.write("OFFLINE", f"link down at soc_temp={temp_s} -- skipping recovery: {blocked}")
        else:
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
