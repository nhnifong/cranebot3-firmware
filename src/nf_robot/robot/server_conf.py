"""Reading and writing server.conf, the record of how a Pi was physically built.

The file started as a single line naming the component type ('arpeggio anchor', 'arpeggio
power anchor'). Build details added since are 'key=value' lines. Two rules keep every
version of the file readable by every version of the code:

  - fields are written BEFORE the component line. The reader this format grew out of kept
    the last non-comment line it saw, so a field written after the component line would be
    mistaken for the component type and rejected as invalid, taking the anchor down on its
    next restart. Written first, that reader passes over it and still lands on the right
    component type.
  - a missing field means the build predates it, so every field's default is how the robots
    were built before the field existed.

Run as a script to write the file directly, for a Pi whose build changed without the whole
anchor_arp_eval procedure being worth re-running:

    python -m nf_robot.robot.server_conf --power --long --set_hostname
    python -m nf_robot.robot.server_conf --path server.conf
"""

import argparse
import logging

logger = logging.getLogger(__name__)

# where anchor_arp_eval writes it. cranebot.service runs with WorkingDirectory=/opt/robot, but
# the older install_pi.sh layout runs from the checkout, so a bare name is also searched.
CONF_PATH = '/opt/robot/server.conf'
FALLBACK_CONF_PATH = 'server.conf'

DEFAULT_COMPONENT_TYPE = 'arpeggio anchor'
# the other type anchor_arp_eval writes; i2c_dispatcher keys the powerline spool off it
POWER_COMPONENT_TYPE = 'arpeggio power anchor'

# how much line anchor_arp_eval.py wound onto the spools: 'long' is its --long mode, which
# puts 20 m on the lower spool and 12 m on the upper instead of 15 m and 7.5 m.
WINDING_SHORT = 'short'
WINDING_LONG = 'long'
WINDINGS = (WINDING_SHORT, WINDING_LONG)

# The torque a spool motor reports while holding still with no load on the line, in N.m in the
# motor's own frame, as (after turning positive, after turning negative). It is the friction the
# motor was pushing through when it stopped, held by its speed loop, so it depends on which way
# the shaft last turned. anchor_arp_eval measures it per motor and records it as
#   hold_torque_motor<can id>=<after positive>,<after negative>
# Anchors without a measurement get the mean of four motors measured by hand: two bare and the
# two on an installed anchor.
HOLD_TORQUE_KEY = 'hold_torque_motor{}'
DEFAULT_HOLD_TORQUE_NM = (0.0252, -0.0328)


def read_server_conf(path=None):
    """Parse server.conf into (component_type, fields).

    Falls back to the historical component type and no fields when the file is missing or
    holds nothing usable, so a Pi that never went through the eval script still boots.
    """
    component_type = DEFAULT_COMPONENT_TYPE
    fields = {}

    candidates = [path] if path is not None else [CONF_PATH, FALLBACK_CONF_PATH]
    lines = None
    for candidate in candidates:
        try:
            with open(candidate, 'r') as file:
                lines = file.readlines()
            break
        except OSError:
            continue
    if lines is None:
        return component_type, fields

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            key, _, value = line.partition('=')
            fields[key.strip()] = value.strip()
        else:
            # last bare line wins, matching the reader this format grew out of
            component_type = line

    return component_type.replace('_', ' '), fields


def read_winding(path=None):
    """'long' or 'short'. Anchors built before the field existed were short-wound, and an
    unrecognized value is treated the same way: a typo here should not stop a boot."""
    _, fields = read_server_conf(path)
    winding = fields.get('winding', WINDING_SHORT)
    if winding not in WINDINGS:
        logger.warning(f'server.conf: unrecognized winding {winding!r}, assuming {WINDING_SHORT}')
        return WINDING_SHORT
    return winding


def parse_hold_torque(value):
    """'<after positive>,<after negative>' -> (float, float), or None if it doesn't hold a
    positive value followed by a negative one."""
    try:
        after_pos, after_neg = (float(v) for v in value.split(','))
    except (AttributeError, ValueError):
        return None
    if not after_pos > 0 > after_neg:
        return None
    return after_pos, after_neg


def read_hold_torque(motor_id, path=None):
    """(after positive, after negative) holding torque for the motor with this CAN id, in N.m.
    Falls back to DEFAULT_HOLD_TORQUE_NM when the file or the field is missing or unreadable."""
    _, fields = read_server_conf(path)
    key = HOLD_TORQUE_KEY.format(motor_id)
    if key not in fields:
        return DEFAULT_HOLD_TORQUE_NM
    hold = parse_hold_torque(fields[key])
    if hold is None:
        logger.warning(f'server.conf: unreadable {key}={fields[key]!r}, assuming {DEFAULT_HOLD_TORQUE_NM}')
        return DEFAULT_HOLD_TORQUE_NM
    return hold


def write_server_conf(component_type, winding=WINDING_SHORT, hold_torques=None, path=CONF_PATH):
    """Rewrite server.conf. Fields first; see the module docstring for why.

    hold_torques is {motor_id: (after positive, after negative)}. Measurements already in the
    file are kept for any motor not given, so rewriting the build details does not throw away a
    friction measurement that took the engineer a slack-line setup to get.
    """
    _, old_fields = read_server_conf(path)
    holds = {k: v for k, v in old_fields.items()
             if k.startswith(HOLD_TORQUE_KEY.format('')) and parse_hold_torque(v) is not None}
    for motor_id, (after_pos, after_neg) in (hold_torques or {}).items():
        holds[HOLD_TORQUE_KEY.format(motor_id)] = f'{after_pos:.5f},{after_neg:.5f}'
    with open(path, 'w') as f:
        f.write(f'winding={winding}\n')
        for key in sorted(holds):
            f.write(f'{key}={holds[key]}\n')
        f.write(component_type + '\n')


def main():
    """Write server.conf from the two things it records, without winding any line.

    The flags are named after anchor_arp_eval's, which is where these details are normally
    captured, so the same --long that wound the spools writes the same file here.
    """
    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument('--power', action='store_true',
                        help=f"this anchor carries the powerline spool ('{POWER_COMPONENT_TYPE}' "
                             f"rather than '{DEFAULT_COMPONENT_TYPE}')")
    parser.add_argument('--long', action='store_true',
                        help='spools were wound long (20 m lower, 12 m upper) rather than '
                             'short (15 m, 7.5 m); the server picks its full spool diameter from this')
    parser.add_argument('--path', default=CONF_PATH,
                        help=f'file to write (default {CONF_PATH})')
    parser.add_argument('--set_hostname', action='store_true',
                        help='also rename this Pi for its role, the step anchor_arp_eval does '
                             'next. Needs sudo, and takes full effect on the next reboot')
    args = parser.parse_args()

    component_type = POWER_COMPONENT_TYPE if args.power else DEFAULT_COMPONENT_TYPE
    winding = WINDING_LONG if args.long else WINDING_SHORT
    write_server_conf(component_type, winding=winding, path=args.path)
    print(f'Wrote {args.path}: {component_type}, winding={winding}')

    if args.set_hostname:
        # Imported here rather than at module scope: every robot server reads this module
        # at boot, and nothing on that path should drag in the QA package.
        from nf_robot.qa.set_hostname import set_component_hostname

        # The role names anchor_arp_eval passes, so a Pi renamed here matches one renamed there.
        set_component_hostname('power-anchor' if args.power else 'anchor')


if __name__ == '__main__':
    main()
