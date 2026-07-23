"""Give each Pi a hostname unique to its role in the setup.

Prevent every Pi in a setup sharing one hostname (the image ships
with a fixed default), which can cause name/DHCP collisions on the LAN and
contribute to disconnects. Every setup has exactly one
gripper, one power anchor, and one plain anchor,
so the component role is a unique discriminator within a setup. The eval
(post-assembly) script calls this once it knows which role the Pi is.
"""

import socket
import subprocess

HOSTNAME_BASE = "stringman"


def set_component_hostname(component: str) -> str:
    """Set this Pi's hostname to '<base>-<component>' and return the new name.

    Idempotent: the name is always built from HOSTNAME_BASE, so re-running the
    eval script (or running it after a role change) replaces the suffix instead
    of stacking another one. Updates /etc/hostname via hostnamectl and the
    127.0.1.1 line in /etc/hosts so local name resolution matches.
    """
    hostname = f"{HOSTNAME_BASE}-{component}"
    current = socket.gethostname()
    if current == hostname:
        print(f"Hostname already set to '{hostname}'.")
        return hostname

    print(f"Setting hostname to '{hostname}' (was '{current}')...")
    subprocess.run(["sudo", "hostnamectl", "set-hostname", hostname], check=True)
    # Keep /etc/hosts in sync so 'sudo' and local lookups resolve the new name.
    # Harmless if there is no 127.0.1.1 line (minimal images may omit it).
    subprocess.run(
        ["sudo", "sed", "-i", f"s/^127\\.0\\.1\\.1.*/127.0.1.1\\t{hostname}/", "/etc/hosts"],
        check=False,
    )
    print(f"Hostname set to '{hostname}'. Full effect after the next reboot.")
    return hostname
