"""The saved drop position: somewhere to send things that isn't a tag on a wall.

Recorded by driving the gantry where you want drops to land and pressing the button, then used
like any other named route destination. What makes it worth its own tests is the height: named
positions are things on the floor and a route hovers the dropoff height above them, so what gets
saved has to be that height below where the gantry stood, or the robot comes back a metre high.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nf_robot.common.config_loader import load_config, save_config
from nf_robot.common.util import tonp
from nf_robot.generated.nf import common
from nf_robot.host.observer import AsyncObserver, DROP_POSITION_NAME, ROUTE_POINT_TAG_NAMES


class TestDropPosition(unittest.TestCase):
    def _observer(self, gantry_pos=(0.4, -0.3, 1.5)):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        ob.pe.gant_pos = np.array(gantry_pos, dtype=float)
        return ob

    def _hover(self, ob):
        return tonp(ob.config.pick_and_place.gantry_height_over_dropoff)

    def test_a_route_brings_the_gantry_back_where_it_stood(self):
        """The whole point: what is saved plus the hover a route adds is where you were."""
        ob = self._observer()
        gantry = ob.pe.gant_pos.copy()

        ob.record_drop_position()

        saved = tonp(ob.config.named_positions[DROP_POSITION_NAME])
        np.testing.assert_allclose(saved + self._hover(ob), gantry, atol=1e-9)

    def test_saving_selects_it_as_the_destination(self):
        ob = self._observer()
        ob.pnp_dst = common.RoutePoint.HAMPER

        ob.record_drop_position()

        self.assertEqual(ob.pnp_dst, common.RoutePoint.DROP_POSITION)
        self.assertEqual(ob.config.last_route_destination, common.RoutePoint.DROP_POSITION)

    def test_it_is_a_named_position_like_the_rest(self):
        """Every route that can fly to a named place gets this one for free, so the mapping
        from route point to name is what makes the feature work at all."""
        self.assertEqual(ROUTE_POINT_TAG_NAMES[common.RoutePoint.DROP_POSITION],
                         DROP_POSITION_NAME)

    def test_the_route_destination_lookup_finds_it(self):
        ob = self._observer()
        ob.record_drop_position()

        found = ob._route_dst_floor_pos()

        np.testing.assert_allclose(found, tonp(ob.config.named_positions[DROP_POSITION_NAME]))

    def test_a_destination_never_recorded_has_no_position(self):
        """Selecting it before recording one must degrade, not raise: the pick and place loop
        asks for this every round."""
        ob = self._observer()
        ob.pnp_dst = common.RoutePoint.DROP_POSITION

        self.assertIsNone(ob._route_dst_floor_pos())

    def test_it_survives_a_restart(self):
        ob = self._observer()
        ob.record_drop_position()
        expected = tonp(ob.config.named_positions[DROP_POSITION_NAME])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'configuration.json'
            save_config(ob.config, path)
            reloaded = load_config(path)

        np.testing.assert_allclose(tonp(reloaded.named_positions[DROP_POSITION_NAME]), expected)
        self.assertEqual(reloaded.last_route_destination, common.RoutePoint.DROP_POSITION)

    def test_saving_low_stores_a_point_under_the_floor(self):
        """A drop position lower than the hover height puts the saved point below zero. Nothing
        minds -- it is only ever hovered over and compared horizontally -- and the gantry still
        comes back to the right place, which is what this pins down."""
        ob = self._observer(gantry_pos=(0.1, 0.2, 0.5))
        gantry = ob.pe.gant_pos.copy()

        ob.record_drop_position()

        saved = tonp(ob.config.named_positions[DROP_POSITION_NAME])
        self.assertLess(saved[2], 0.0)
        np.testing.assert_allclose(saved + self._hover(ob), gantry, atol=1e-9)
