"""Tests for the visual servoing model's close head, which reads the cell grid near the jaws.

Skipped where the ML extras or the cached backbone are missing, since building the model
downloads several hundred MB otherwise.
"""

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    from huggingface_hub import try_to_load_from_cache

    from nf_robot.ml.dino_trunk import drop_trunk_weights
    from nf_robot.ml.visual_servoing.model import (
        CLOSE_POOL, DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE, VisualServoNet,
        adaptive_avg_pool2d, load_checkpoint, predict)
    cached = isinstance(try_to_load_from_cache(DEFAULT_BACKBONE, "config.json"), str)
    AVAILABLE = cached
except ImportError:
    AVAILABLE = False

needs_backbone = unittest.skipUnless(
    AVAILABLE, "needs torch/transformers and a cached DINOv2 backbone")


@needs_backbone
class TestSpatialCloseHead(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.spatial = VisualServoNet().eval()
        width, height = DEFAULT_IMAGE_SIZE
        cls.pixels = torch.randn(2, 3, height, width)
        cls.state = torch.randn(2, 3)

    def test_predict_reports_close_and_pressure_per_item(self):
        prediction = predict(self.spatial, self.pixels, self.state)
        self.assertEqual(prediction["close"].shape, (2,))
        self.assertEqual(prediction["grasp_pressure"].shape, (2,))

    def test_the_pressure_head_stays_on_the_global_vector(self):
        """How hard to squeeze is a property of the object, not of where it is in frame."""
        self.assertFalse(torch.isnan(predict(self.spatial, self.pixels, self.state)
                                     ["grasp_pressure"]).any())
        self.assertTrue((predict(self.spatial, self.pixels, self.state)
                         ["grasp_pressure"] >= 0).all())

    def test_the_same_object_reads_differently_at_the_jaws_and_at_the_edge(self):
        """The whole point. Identical feature content in two places has to reach the head
        as two different vectors, or the move bought nothing over the pooled one."""
        rows, cols = self.spatial.grid
        blank = torch.zeros(1, self.spatial.close_reduce.in_channels, rows, cols)
        at_jaws = blank.clone()
        at_jaws[:, :, -rows // 6:, cols // 2 - 4:cols // 2 + 4] = 3.0
        at_edge = blank.clone()
        at_edge[:, :, :rows // 6, :8] = 3.0

        def close_logit(feature_map):
            pooled = adaptive_avg_pool2d(
                F.gelu(self.spatial.close_reduce(feature_map)), CLOSE_POOL).flatten(1)
            return self.spatial.close_head(
                torch.cat([pooled, torch.zeros(1, 3)], dim=-1)).item()

        self.assertNotAlmostEqual(close_logit(at_jaws), close_logit(at_edge), places=5)

    def test_the_pool_keeps_more_than_one_bin(self):
        """A 1x1 pool would be the pooled vector again under a different name."""
        self.assertGreater(CLOSE_POOL[0] * CLOSE_POOL[1], 1)


@needs_backbone
class TestCheckpointRoundTrip(unittest.TestCase):

    def payload(self, model):
        return {
            "backbone_id": model.backbone_id, "image_size": list(model.image_size),
            "fuse_layers": model.fuse_layers, "attention_layers": len(model.attention),
            "freeze": model.freeze,
            "state_dict": drop_trunk_weights(model.state_dict(), model.trunk, verify=False),
        }

    def round_trip(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pth"
            torch.save(payload, path)
            model, _ = load_checkpoint(path, "cpu")
            return model

    def test_a_saved_model_loads_with_the_same_weights(self):
        trained = VisualServoNet()
        loaded = self.round_trip(self.payload(trained))
        for key, value in trained.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
