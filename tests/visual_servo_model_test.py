"""Tests for where the close head reads from.

Whether to start closing is a spatial question - something between the fingers, square on,
near enough - and the jaws sit in one fixed part of the frame. Reading it off a vector that
has already been pooled over the whole image can only say the frame contains a graspable
thing, not that this one is in the jaws. --spatial_close moves the head onto the cell grid;
these pin what that has to preserve, and that the robot's side of the model is unchanged
either way.

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

    from nf_robot.ml.visual_servoing.model import (
        CLOSE_POOL, DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE, VisualServoNet,
        drop_trunk_weights, load_checkpoint, predict)
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
        cls.spatial = VisualServoNet(close_heads=True, spatial_close=True).eval()
        cls.globalv = VisualServoNet(close_heads=True, spatial_close=False).eval()
        width, height = DEFAULT_IMAGE_SIZE
        cls.pixels = torch.randn(2, 3, height, width)
        cls.state = torch.randn(2, 3)

    def test_the_robot_sees_the_same_model_either_way(self):
        """Deployment reads close and grasp_pressure by name, so moving where the logit
        comes from must not change the shape of what comes out."""
        spatial = predict(self.spatial, self.pixels, self.state)
        pooled = predict(self.globalv, self.pixels, self.state)
        self.assertEqual(sorted(spatial), sorted(pooled))
        self.assertEqual(spatial["close"].shape, pooled["close"].shape)
        self.assertEqual(spatial["grasp_pressure"].shape, pooled["grasp_pressure"].shape)

    def test_the_close_logit_leaves_the_global_vector(self):
        """One fewer flag off the pooled vector, because close is no longer one of them."""
        self.assertEqual(self.spatial.global_outputs, 4)
        self.assertEqual(self.globalv.global_outputs, 5)

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
            pooled = self.spatial.close_pool(
                F.gelu(self.spatial.close_reduce(feature_map))).flatten(1)
            return self.spatial.close_head(
                torch.cat([pooled, torch.zeros(1, 3)], dim=-1)).item()

        self.assertNotAlmostEqual(close_logit(at_jaws), close_logit(at_edge), places=5)

    def test_the_pool_keeps_more_than_one_bin(self):
        """A 1x1 pool would be the pooled vector again under a different name."""
        self.assertGreater(CLOSE_POOL[0] * CLOSE_POOL[1], 1)


@needs_backbone
class TestCheckpointCompatibility(unittest.TestCase):
    """A checkpoint says which close head it was trained with, and one written before the
    spatial head existed has to keep loading onto the global one."""

    def payload(self, model, **extra):
        return {
            "backbone_id": model.backbone_id, "image_size": list(model.image_size),
            "fuse_layers": model.fuse_layers, "attention_layers": len(model.attention),
            "freeze": model.freeze, "close_heads": model.close_heads,
            "state_dict": drop_trunk_weights(model.state_dict(), model.trunk, verify=False),
            **extra,
        }

    def round_trip(self, payload):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pth"
            torch.save(payload, path)
            model, _ = load_checkpoint(path, "cpu")
            return model

    def test_a_checkpoint_without_the_key_loads_on_the_global_head(self):
        trained = VisualServoNet(close_heads=True, spatial_close=False)
        loaded = self.round_trip(self.payload(trained))
        self.assertFalse(loaded.spatial_close)
        self.assertEqual(loaded.global_outputs, 5)

    def test_a_spatial_checkpoint_comes_back_spatial(self):
        trained = VisualServoNet(close_heads=True, spatial_close=True)
        loaded = self.round_trip(self.payload(trained, spatial_close=True))
        self.assertTrue(loaded.spatial_close)
        self.assertEqual(loaded.global_outputs, 4)


if __name__ == "__main__":
    unittest.main()
