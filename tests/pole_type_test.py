import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np

import nf_robot.common.definitions as model_constants
from nf_robot.common.config_loader import create_default_config, load_config, save_config
from nf_robot.common.pose_functions import invert_pose
from nf_robot.generated.nf import common
from nf_robot.host.arp_gripper_client import ArpeggioGripperClient
from nf_robot.host.observer import AsyncObserver

try:
    from nf_robot.robot.gripper_arp_server import default_gripper_conf
except ImportError:
    # the gripper firmware imports pi-only hardware libraries; the tests that read its
    # defaults skip everywhere else
    default_gripper_conf = None

needs_firmware = unittest.skipIf(default_gripper_conf is None,
                                 'gripper firmware imports pi hardware libraries')


class TestPoleTypeConfig(unittest.TestCase):
    """The pole a robot has sets its swing frequency, so a config that predates the
    field has to keep the pole it was calibrated against rather than take today's."""

    def _load_written(self, config_dict):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'configuration.json'
            path.write_text(json.dumps(config_dict))
            return load_config(path)

    def _default_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'configuration.json'
            save_config(create_default_config(), path)
            return json.loads(path.read_text())

    def test_a_new_config_gets_the_current_pole(self):
        # The pole newly built robots carry. Update it here when the build changes, which is
        # the point of the test: a new config must not quietly inherit an older pole's geometry.
        self.assertEqual(create_default_config().gripper.pole_type, common.PoleType.CARBON270)

    def test_a_config_older_than_the_field_gets_the_abs_pole(self):
        older = self._default_json()
        del older['gripper']['poleType']
        self.assertEqual(self._load_written(older).gripper.pole_type, common.PoleType.ABS500)

    def test_a_config_without_a_gripper_at_all_still_loads(self):
        older = self._default_json()
        del older['gripper']
        self.assertEqual(self._load_written(older).gripper.pole_type, common.PoleType.ABS500)

    def test_a_recorded_pole_type_survives_a_load(self):
        saved = self._default_json()
        self.assertEqual(self._load_written(saved).gripper.pole_type, common.PoleType.CARBON270)

    @needs_firmware
    def test_the_gripper_boots_on_the_older_pole(self):
        """A gripper nobody tells is one on a robot too old to have the field."""
        self.assertEqual(default_gripper_conf['POLE_LENGTH'], model_constants.pole_length_abs500)


class TestObserverUsesTheConfiguredPole(unittest.TestCase):
    """Every length the pole decides has to come from the config, not a fixed constant."""

    def test_hang_distance_comes_from_the_pole(self):
        for pole_type, offset in ((common.PoleType.ABS500, model_constants.pole_offset_abs500),
                                  (common.PoleType.CARBON400, model_constants.pole_offset_carbon400),
                                  (common.PoleType.CARBON270, model_constants.pole_offset_carbon270)):
            with self.subTest(pole_type=pole_type):
                geom = model_constants.POLE_GEOMETRY[pole_type]
                self.assertAlmostEqual(geom.gantry_to_gripper, offset)

    def test_a_default_robot_hangs_at_the_current_pole_offset(self):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        np.testing.assert_allclose(ob.pole, [0, 0, model_constants.pole_offset_carbon270])
        self.assertAlmostEqual(ob.pendulum.length, model_constants.pole_length_carbon270)

    def test_the_carbon_pole_carries_the_flat_marker(self):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        np.testing.assert_allclose(ob.gantry_april_inv[1],
                                   invert_pose(model_constants.gantry_flat_april)[1])

    def test_an_older_robot_keeps_the_box_marker_and_offset(self):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        ob.config.gripper.pole_type = common.PoleType.ABS500
        geom = model_constants.pole_geometry(ob.config)
        self.assertAlmostEqual(geom.gantry_to_gripper, model_constants.pole_offset_abs500)
        self.assertIs(geom.gantry_april, model_constants.gantry_april)


class TestPoleLengthReachesTheGripper(unittest.IsolatedAsyncioTestCase):
    def _client(self, pole_type):
        ob = AsyncObserver(terminate_with_ui=False, config_path=None, port=0)
        ob.config.gripper.pole_type = pole_type
        client = ArpeggioGripperClient('10.0.0.9', 0, ob.datastore, ob, None, ob.stat, ob.pe, None)
        client.websocket = AsyncMock()
        return client

    def _sent_config_vars(self, websocket):
        [call] = websocket.send.await_args_list
        return json.loads(call.args[0])['set_config_vars']

    async def test_connecting_tells_the_gripper_which_pole_it_is_on(self):
        client = self._client(common.PoleType.CARBON400)

        await client.send_config()

        sent = self._sent_config_vars(client.websocket)
        self.assertAlmostEqual(sent['POLE_LENGTH'], model_constants.pole_length_carbon400)
        # whatever the gripper is told is what the host projects against
        self.assertAlmostEqual(client.pendulum.length, sent['POLE_LENGTH'])

    async def test_the_other_pole_sends_the_other_length(self):
        client = self._client(common.PoleType.ABS500)

        await client.send_config()

        self.assertAlmostEqual(self._sent_config_vars(client.websocket)['POLE_LENGTH'],
                               model_constants.pole_length_abs500)

    @needs_firmware
    async def test_the_gripper_understands_the_key_it_is_sent(self):
        """POLE_LENGTH has to name a var the gripper actually reconfigures on."""
        client = self._client(common.PoleType.CARBON400)

        await client.send_config()

        for key in self._sent_config_vars(client.websocket):
            self.assertIn(key, default_gripper_conf)


if __name__ == '__main__':
    unittest.main()
