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
"""

import logging

logger = logging.getLogger(__name__)

# where anchor_arp_eval writes it. cranebot.service runs with WorkingDirectory=/opt/robot, but
# the older install_pi.sh layout runs from the checkout, so a bare name is also searched.
CONF_PATH = '/opt/robot/server.conf'
FALLBACK_CONF_PATH = 'server.conf'

DEFAULT_COMPONENT_TYPE = 'arpeggio anchor'

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
