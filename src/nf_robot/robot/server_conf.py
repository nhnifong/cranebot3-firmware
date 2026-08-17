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


def write_server_conf(component_type, winding=WINDING_SHORT, path=CONF_PATH):
    """Rewrite server.conf from scratch. Fields first; see the module docstring for why."""
    with open(path, 'w') as f:
        f.write(f'winding={winding}\n')
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
