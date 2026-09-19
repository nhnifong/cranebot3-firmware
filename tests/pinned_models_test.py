"""The pinned targeting and visual servoing models load and predict, the way the observer does it.

Needs only the `host` extra; CI runs this file in a venv with nothing else installed, which is
what catches a dependency that only the dev extra happened to provide.

Downloads the backbone and both checkpoints (several hundred MB), so it runs only when they are
already cached or NF_TEST_PINNED_MODELS=1 asks for the download.
"""

import os
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from nf_robot.common.model_revisions import pinned_revision
from nf_robot.ml.ortho_target import model as ortho_target
from nf_robot.ml.visual_servoing import servo
from nf_robot.ml.visual_servoing.model import DEFAULT_BACKBONE

DEVICE = torch.device("cpu")


def _cached(repo_id, filename, revision=None):
    return isinstance(try_to_load_from_cache(repo_id, filename, revision=revision), str)


def _available():
    if os.environ.get("NF_TEST_PINNED_MODELS"):
        return True
    return (_cached(DEFAULT_BACKBONE, "config.json")
            and _cached(ortho_target.TARGETING_MODEL_REPOID, ortho_target.TARGETING_MODEL_FILENAME,
                        pinned_revision(ortho_target.TARGETING_MODEL_REPOID))
            and _cached(servo.SERVO_MODEL_REPOID, servo.SERVO_MODEL_FILENAME,
                        pinned_revision(servo.SERVO_MODEL_REPOID)))


needs_models = unittest.skipUnless(
    _available(), "pinned models not cached; set NF_TEST_PINNED_MODELS=1 to download them")


@needs_models
class TestPinnedModels(unittest.TestCase):

    def test_the_pinned_ortho_target_model_loads_and_predicts(self):
        repo_id = ortho_target.TARGETING_MODEL_REPOID
        path = hf_hub_download(repo_id=repo_id, filename=ortho_target.TARGETING_MODEL_FILENAME,
                               revision=pinned_revision(repo_id))
        model, _ = ortho_target.load_checkpoint(path, DEVICE)

        frame = np.zeros((600, 600, 3), dtype=np.uint8)
        targets = ortho_target.predict_room_targets(model, frame, DEVICE, top_k=5,
                                                    min_probability=0.0)
        self.assertEqual(len(targets), 5)
        self.assertTrue(np.isfinite(np.array(targets)).all())

    def test_the_pinned_visual_servo_model_loads_and_predicts(self):
        model, _ = servo.load_model(DEVICE, revision=pinned_revision(servo.SERVO_MODEL_REPOID))

        frame = np.zeros((384, 684, 3), dtype=np.uint8)
        state = {"laser_rangefinder": 0.3, "finger_angle": 0.0, "target_force": 0.0}
        prediction = servo.predict_frame(model, frame, state, DEVICE, spin=0.0,
                                         gripper_pos=np.array([0.0, 0.0, 1.0]))
        for key in ("uv", "range_m", "point_room", "close", "grasp_pressure", "holding"):
            with self.subTest(key=key):
                self.assertTrue(np.isfinite(np.asarray(prediction[key], dtype=float)).all())


if __name__ == "__main__":
    unittest.main()
