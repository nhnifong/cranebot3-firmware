import os
import tempfile
import unittest

import nf_robot.common.definitions as model_constants
from nf_robot.robot import server_conf


def old_reader(path):
    """The server.conf reader as it stood before the winding field existed.

    Anchors in the field run this until their package is upgraded, so a file written by the
    current writer has to keep parsing here. It kept the last non-comment line and rejected
    anything that was not a known component type, which would take an anchor down on restart.
    """
    component_type = 'arpeggio anchor'
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line.startswith('#') and line:
                    component_type = line
    except FileNotFoundError:
        pass
    component_type = component_type.replace('_', ' ')
    if component_type not in ('arpeggio anchor', 'arpeggio power anchor'):
        raise ValueError(f'Invalid type in server.conf "{component_type}"')
    return component_type


class TestServerConf(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def write(self, body):
        path = os.path.join(self.dir, 'server.conf')
        with open(path, 'w') as f:
            f.write(body)
        return path

    # ------------------------------------------------------------ older files

    def test_single_line_file_parses(self):
        path = self.write('arpeggio power anchor\n')
        self.assertEqual(server_conf.read_server_conf(path),
                         ('arpeggio power anchor', {}))

    def test_older_files_default_to_short_winding(self):
        """Every anchor built before the field existed was short-wound."""
        for body in ('arpeggio anchor\n',
                     'arpeggio_power_anchor\n',
                     '# a comment\n\narpeggio anchor\n',
                     'arpeggio anchor'):  # no trailing newline
            self.assertEqual(server_conf.read_winding(self.write(body)),
                             server_conf.WINDING_SHORT, body)

    def test_missing_and_empty_files_do_not_raise(self):
        for path in (os.path.join(self.dir, 'absent.conf'), self.write('')):
            self.assertEqual(server_conf.read_server_conf(path),
                             (server_conf.DEFAULT_COMPONENT_TYPE, {}))
            self.assertEqual(server_conf.read_winding(path), server_conf.WINDING_SHORT)

    def test_underscores_become_spaces(self):
        path = self.write('arpeggio_power_anchor\n')
        self.assertEqual(server_conf.read_server_conf(path)[0], 'arpeggio power anchor')

    # ------------------------------------------------------------ newer files

    def test_winding_round_trips(self):
        for winding in server_conf.WINDINGS:
            for anchor_type in ('arpeggio anchor', 'arpeggio power anchor'):
                path = os.path.join(self.dir, f'{winding}_{anchor_type}.conf')
                server_conf.write_server_conf(anchor_type, winding=winding, path=path)
                self.assertEqual(server_conf.read_server_conf(path)[0], anchor_type)
                self.assertEqual(server_conf.read_winding(path), winding)

    def test_unknown_field_is_ignored_not_mistaken_for_the_component(self):
        path = self.write('winding=long\nsomething_added_later=7\narpeggio power anchor\n')
        component_type, fields = server_conf.read_server_conf(path)
        self.assertEqual(component_type, 'arpeggio power anchor')
        self.assertEqual(fields['something_added_later'], '7')
        self.assertEqual(server_conf.read_winding(path), server_conf.WINDING_LONG)

    def test_unrecognized_winding_falls_back_rather_than_raising(self):
        """A typo in a hand-edited conf should not stop an anchor from booting."""
        path = self.write('winding=medium\narpeggio anchor\n')
        self.assertEqual(server_conf.read_winding(path), server_conf.WINDING_SHORT)

    def test_written_file_still_parses_under_the_old_reader(self):
        """Fields are written before the component line precisely so this holds; an anchor
        running the old package must survive having its winding recorded."""
        for winding in server_conf.WINDINGS:
            for anchor_type in ('arpeggio anchor', 'arpeggio power anchor'):
                path = os.path.join(self.dir, f'old_{winding}_{anchor_type}.conf')
                server_conf.write_server_conf(anchor_type, winding=winding, path=path)
                self.assertEqual(old_reader(path), anchor_type)


class TestSpoolGeometry(unittest.TestCase):

    def test_every_spool_a_server_can_ask_for_is_present(self):
        for winding in server_conf.WINDINGS:
            for line_type in ('fishing', 'power'):
                self.assertIn((winding, 'high', line_type), model_constants.damiao_spool_geometry)
            self.assertIn((winding, 'low', 'fishing'), model_constants.damiao_spool_geometry)

    def test_short_winding_keeps_the_values_deployed_robots_were_built_against(self):
        geom = model_constants.damiao_spool_geometry
        self.assertEqual(geom[('short', 'high', 'power')],
                         (model_constants.assumed_full_line_length,
                          model_constants.damiao_full_spool_diameter_power_line))
        self.assertEqual(geom[('short', 'high', 'fishing')],
                         (model_constants.assumed_full_line_length,
                          model_constants.damiao_full_spool_diameter_fishing_line))
        self.assertEqual(geom[('short', 'low', 'fishing')],
                         (model_constants.assumed_full_line_length,
                          model_constants.damiao_full_spool_diameter_fishing_line))

    def test_a_longer_winding_never_means_a_thinner_spool(self):
        """SpiralCalculator needs full_length and full_diameter to describe the same spool
        state, so more line has to mean both a longer length and a fatter pile."""
        geom = model_constants.damiao_spool_geometry
        for spool, line_type in (('high', 'fishing'), ('high', 'power'), ('low', 'fishing')):
            short_len, short_diam = geom[('short', spool, line_type)]
            long_len, long_diam = geom[('long', spool, line_type)]
            self.assertGreater(long_len, short_len, (spool, line_type))
            self.assertGreater(long_diam, short_diam, (spool, line_type))
            self.assertGreater(short_diam, model_constants.damiao_empty_spool_diameter)


if __name__ == '__main__':
    unittest.main()
