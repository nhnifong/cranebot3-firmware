"""Tests for the published-model pins.

Robots download the targeting and visual servoing checkpoints at a fixed commit so that
publishing one does not change what is already flying. These check the file says something
usable and that the code reading it agrees with the code writing it - a pin keyed on a repo
nothing downloads from would be silently inert.
"""

import json
import re
import unittest

from nf_robot.common.model_revisions import (
    REVISIONS_PATH, load_revisions, pinned_revision)


class TestPinnedRevisions(unittest.TestCase):

    def test_every_pin_is_a_full_commit_sha(self):
        """A short sha or a branch name would resolve today and drift later, which is the
        whole thing being prevented."""
        for repo, entry in load_revisions().items():
            with self.subTest(repo=repo):
                self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")

    def test_an_unpinned_repo_raises_rather_than_taking_the_tip(self):
        with self.assertRaises(KeyError):
            pinned_revision("naavox/not-a-model")

    def test_the_file_survives_a_json_round_trip(self):
        """pin_latest_model rewrites it wholesale, so it has to be plain data."""
        self.assertIsInstance(json.loads(REVISIONS_PATH.read_text()), dict)

    def test_the_models_the_robot_loads_are_all_pinned(self):
        """The two repos the host downloads from, named by the modules that own them, so
        renaming a repo without repinning fails here rather than on a robot."""
        from nf_robot.ml import ortho_target
        from nf_robot.ml.visual_servoing import servo

        pinned = load_revisions()
        self.assertIn(ortho_target.TARGETING_MODEL_REPOID, pinned)
        self.assertIn(servo.SERVO_MODEL_REPOID, pinned)

    def test_the_pinning_tool_covers_exactly_those_models(self):
        """--targeting and --visual_servo have to reach the same repos the host reads, or
        the tool writes pins nothing uses."""
        from nf_robot.ml.pin_latest_model import MODELS, repo_id_for

        self.assertEqual({repo_id_for(name) for name in MODELS}, set(load_revisions()))


if __name__ == "__main__":
    unittest.main()
