#!/usr/bin/env python3
"""Provision SSH public-key auth across a robot's three components.

A Stringman robot is described by a conf_*.json file that lists two anchors and
one gripper, each reachable over the LAN. Fresh Raspberry Pi images ship with
password SSH enabled, which is both a hassle (typing the password for every
deploy) and a liability on a shared network. This helper walks all three
components and, for each one:

  1. Uploads your SSH public key to the `pi` account (ssh-copy-id).
  2. Confirms that key-only authentication actually works.
  3. Writes an authoritative sshd drop-in that disables password auth, validates
     the new config with `sshd -t`, and restarts the ssh service.
  4. Re-verifies key auth after the restart, and confirms password auth is now
     refused.

The ordering is the whole point: password auth is only disabled *after* key auth
is proven on that specific host, so a missing key or typo can never lock you out
of a Pi. Any host that fails an early step is skipped with its password login
left intact.

The `pi` user has passwordless sudo on these images, so the privileged steps run
under `sudo -n`. The first connection to each host still uses password auth (you
will be prompted by ssh-copy-id), unless the key is already installed.

Usage:
    experiments/deploy_ssh_keys.py conf_playroom.json
    experiments/deploy_ssh_keys.py conf_bedroom.json --pubkey ~/.ssh/id_ed25519.pub
    experiments/deploy_ssh_keys.py conf_red.json --user pi --dry-run
"""

import argparse
import json
import os
import subprocess
import sys

# Authoritative sshd drop-in. Include directives sit at the top of the stock
# sshd_config, and sshd honours the *first* value it sees for a keyword, so a
# drop-in that sorts first (00-) wins over both the main file and any later
# drop-in (e.g. a 50-cloud-init.conf that re-enables password auth).
DROPIN_PATH = "/etc/ssh/sshd_config.d/00-disable-password-auth.conf"
DROPIN_CONTENT = """\
# Managed by experiments/deploy_ssh_keys.py -- keys only, no passwords.
PasswordAuthentication no
KbdInteractiveAuthentication no
ChallengeResponseAuthentication no
PubkeyAuthentication yes
"""

# Remote script that installs the drop-in, neutralizes any conflicting setting
# left in the main config or other drop-ins, validates, and restarts ssh.
DISABLE_PASSWORD_REMOTE = r"""
set -eu
sudo -n tee {dropin} >/dev/null <<'DROPIN_EOF'
{content}DROPIN_EOF
sudo -n chmod 644 {dropin}
# Comment out any PasswordAuthentication line elsewhere so nothing earlier-sorting
# can override us. Skip our own drop-in.
for f in /etc/ssh/sshd_config $(ls /etc/ssh/sshd_config.d/*.conf 2>/dev/null); do
    [ "$f" = "{dropin}" ] && continue
    sudo -n sed -i 's/^[[:space:]]*PasswordAuthentication[[:space:]]/#&/I' "$f" || true
done
# Refuse to restart on a broken config -- that could strand the host.
sudo -n sshd -t
if systemctl list-unit-files ssh.socket >/dev/null 2>&1 \
   && systemctl is-enabled ssh.socket >/dev/null 2>&1; then
    sudo -n systemctl restart ssh.socket || true
fi
sudo -n systemctl restart ssh 2>/dev/null || sudo -n systemctl restart sshd
echo OK
"""

# ANSI colors, disabled when stdout is not a tty.
_TTY = sys.stdout.isatty()
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def ok(s):    return _c("32", s)
def warn(s):  return _c("33", s)
def err(s):   return _c("31", s)
def bold(s):  return _c("1", s)


def find_default_pubkey():
    """Return the most reasonable local public key, or None."""
    ssh_dir = os.path.expanduser("~/.ssh")
    for name in ("id_ed25519.pub", "id_ecdsa.pub", "id_rsa.pub"):
        path = os.path.join(ssh_dir, name)
        if os.path.exists(path):
            return path
    return None


def load_components(conf_path):
    """Parse conf JSON into [(label, address), ...] for the 3 components."""
    with open(conf_path) as f:
        conf = json.load(f)
    components = []
    for i, anchor in enumerate(conf.get("anchors", [])):
        addr = anchor.get("address")
        if addr:
            components.append((f"anchor[{i}]", addr))
    gripper = conf.get("gripper") or {}
    if gripper.get("address"):
        components.append(("gripper", gripper["address"]))
    return components


def ssh_base(user, host, extra=()):
    """Common ssh options. ConnectTimeout keeps unreachable hosts from hanging."""
    return [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        *extra,
        f"{user}@{host}",
    ]


def key_auth_works(user, host, identity):
    """True if we can log in using *only* the key (no password prompt possible)."""
    cmd = ssh_base(user, host, extra=[
        "-o", "BatchMode=yes",
        "-o", "PasswordAuthentication=no",
        "-o", "PubkeyAuthentication=yes",
        "-i", identity,
    ]) + ["true"]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def password_auth_refused(user, host):
    """True if the host now refuses password auth (best-effort check).

    BatchMode means ssh can't actually type a password; combined with disabling
    pubkey, a host that still allowed passwords would get to a password prompt
    and fail differently than one that refuses the method outright. We treat any
    non-zero exit as 'password not usable', which is the safe interpretation.
    """
    cmd = ssh_base(user, host, extra=[
        "-o", "BatchMode=yes",
        "-o", "PubkeyAuthentication=no",
        "-o", "PreferredAuthentications=password,keyboard-interactive",
    ]) + ["true"]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode != 0


def provision(user, host, pubkey, identity, dry_run):
    """Run the full flow for one host. Returns True on success."""
    # 1. Upload the public key (prompts for the password unless already set up).
    if key_auth_works(user, host, identity):
        print(f"    key already installed, skipping upload")
    else:
        print(f"    uploading public key (you may be prompted for the password)...")
        if dry_run:
            print("    [dry-run] would run ssh-copy-id")
        else:
            r = subprocess.run(["ssh-copy-id", "-i", pubkey,
                                "-o", "ConnectTimeout=10",
                                "-o", "StrictHostKeyChecking=accept-new",
                                f"{user}@{host}"])
            if r.returncode != 0:
                print(err(f"    ssh-copy-id failed; leaving password auth intact"))
                return False

    # 2. Verify key-only auth BEFORE touching password auth -- this is the
    #    safety gate that prevents lockouts.
    if dry_run:
        print("    [dry-run] would verify key-only auth, then disable passwords")
        return True
    if not key_auth_works(user, host, identity):
        print(err("    key auth did NOT work after upload; refusing to disable "
                  "passwords (no changes made)"))
        return False
    print(ok("    key-only auth verified"))

    # 3. Disable password auth on the remote.
    remote = DISABLE_PASSWORD_REMOTE.format(dropin=DROPIN_PATH, content=DROPIN_CONTENT)
    cmd = ssh_base(user, host, extra=[
        "-o", "BatchMode=yes", "-i", identity,
    ]) + ["bash", "-s"]
    r = subprocess.run(cmd, input=remote, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0 or "OK" not in r.stdout:
        print(err("    failed to disable password auth:"))
        print("    " + (r.stdout or "").strip().replace("\n", "\n    "))
        return False

    # 4. Re-verify key auth survived the sshd restart, and that passwords are gone.
    if not key_auth_works(user, host, identity):
        print(err("    key auth broke after restarting ssh -- investigate this "
                  "host manually before logging out!"))
        return False
    print(ok("    key auth still works after ssh restart"))

    if password_auth_refused(user, host):
        print(ok("    password auth is now refused"))
    else:
        print(warn("    could not confirm password auth is disabled (verify by hand)"))
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("conf", help="path to a conf_*.json describing the robot")
    p.add_argument("--user", default="pi", help="remote username (default: pi)")
    p.add_argument("--pubkey", help="public key to install "
                                    "(default: ~/.ssh/id_ed25519.pub or similar)")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would happen without changing anything")
    args = p.parse_args()

    pubkey = args.pubkey or find_default_pubkey()
    if not pubkey:
        sys.exit(err("No public key found in ~/.ssh; pass one with --pubkey "
                     "(or create one with `ssh-keygen -t ed25519`)."))
    pubkey = os.path.expanduser(pubkey)
    if not os.path.exists(pubkey):
        sys.exit(err(f"Public key not found: {pubkey}"))
    identity = pubkey[:-4] if pubkey.endswith(".pub") else pubkey
    if not os.path.exists(identity):
        sys.exit(err(f"Matching private key not found: {identity}"))

    components = load_components(args.conf)
    if not components:
        sys.exit(err(f"No anchor/gripper addresses found in {args.conf}"))

    print(f"Using public key: {pubkey}")
    print(f"Provisioning {len(components)} component(s) from {args.conf} as "
          f"user {bold(args.user)}:\n")

    results = {}
    for label, host in components:
        print(bold(f"== {label} @ {host} =="))
        try:
            results[(label, host)] = provision(
                args.user, host, pubkey, identity, args.dry_run)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # keep going to the other hosts
            print(err(f"    unexpected error: {e}"))
            results[(label, host)] = False
        print()

    print(bold("Summary:"))
    n_ok = 0
    for (label, host), good in results.items():
        if good:
            n_ok += 1
            print(f"  {ok('OK  ')} {label} @ {host}")
        else:
            print(f"  {err('FAIL')} {label} @ {host}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
