#!/usr/bin/env bash
# Install this source tree's fan_control.py on every component (anchors and
# gripper) listed in a robot config, and restart the fan-control service.
# Asks for the pi user's password once and reuses it for every component.
#
#   scripts/update_fan_control.sh conf_playroom.json
#
# Needs sshpass (sudo apt install sshpass).
set -u

if [ $# -ne 1 ]; then
    echo "usage: $0 <robot config json>" >&2
    exit 2
fi
config=$1
repo=$(cd "$(dirname "$0")/.." && pwd)
src="$repo/stringman-pilot-rpi-image/fan_control.py"

command -v sshpass >/dev/null || { echo "sshpass not found: sudo apt install sshpass" >&2; exit 1; }

addresses=$(python3 - "$config" <<'EOF'
import json, sys
conf = json.load(open(sys.argv[1]))
components = conf.get("anchors", []) + [conf.get("gripper", {})]
for c in components:
    if c.get("address"):
        print(c["address"])
EOF
) || exit 1
[ -n "$addresses" ] || { echo "no component addresses in $config" >&2; exit 1; }

echo "components: $(echo $addresses)"
read -rsp "password for pi: " SSHPASS
echo
export SSHPASS

failed=()
for addr in $addresses; do
    echo "== $addr"
    if sshpass -e ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 "pi@$addr" \
        'cat > /tmp/fan_control.py &&
         sudo install -m 755 /tmp/fan_control.py /usr/local/bin/fan_control.py &&
         sudo systemctl restart fan-control &&
         sleep 3 &&
         journalctl -u fan-control -n 2 --no-pager -o cat &&
         pinctrl get 12,18' < "$src"
    then
        :
    else
        failed+=("$addr")
    fi
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "all components updated"
