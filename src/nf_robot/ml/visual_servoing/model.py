#!/usr/bin/env python

"""The visual servoing network: frozen DINOv2 patch tokens -> where the object is.

    input   448x252 RGB, the gripper camera at its native aspect
            plus laser_rangefinder, finger_angle and target_force

    trunk   frozen DINOv2 ViT-B/14, last 4 hidden states, patch tokens only
              -> concat on channels                     (B, 3072, 18, 32)
            Conv2d(3072 -> 256, 1x1), GroupNorm, GELU   (B,  256, 18, 32)
            FiLM from the state vector
            self-attention blocks over 576 tokens       (B,  256, 18, 32)
            skip: concat the pre-attention map, 1x1 -> 256 (B,  256, 18, 32)

    heads   1. target position, 3D, in the gripper camera frame: the centre of mass
               of the cell softmax in a window around the winning cell
            2. grasp axis, 2 channels, averaged over that same window
            3. finger speed, scalar in [-1, 1], from the global vector
            4. probability any graspable target is present
            5. probability we are currently holding something
            6. probability the close should have begun, from the cell grid pooled
               into coarse bins so the jaws have their own
            7. grip pressure the object will need, from the global vector
"""

import torch
import torch.nn.functional as F
from torch import nn

from nf_robot.ml.dino_trunk import load_head_state
from nf_robot.ml.grid_head import (
    CENTROID_RADIUS, adaptive_avg_pool2d, local_maxima, window_average,
)
from nf_robot.ml.gripper_grid import (
    DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE, GripperGridNet, camera_point, decode_position,
)

# Channels and pooled grid of the spatial close head, coarse enough to flatten but keeping
# the jaws in their own bins.
CLOSE_CHANNELS = 32
CLOSE_POOL = (4, 6)
# laser_rangefinder, finger_angle, target_force; deliberately not velocity or finger
# pressure.
STATE_DIM = 3


class VisualServoNet(GripperGridNet):
    """Frozen DINOv2 patch features -> target position, grasp axis, finger, flags."""

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 fuse_layers=4, width=256, attention_layers=3, heads=8, freeze=True,
                 state_dim=STATE_DIM):
        super().__init__(backbone_id=backbone_id, image_size=image_size,
                         fuse_layers=fuse_layers, width=width,
                         attention_layers=attention_layers, heads=heads, freeze=freeze,
                         state_dim=state_dim)
        channels = width

        # Head 1: one softmax over canvas cells, plus log distance.
        self.logit_head = nn.Conv2d(channels, 1, 1)
        # Log metres per cell, since objects at different heights have different distances.
        self.distance_head = nn.Conv2d(channels, 1, 1)
        # Head 2: (sin 2t, cos 2t), because the grasp axis is pi-periodic.
        self.axis_head = nn.Conv2d(channels, 2, 1)

        # Finger rate, present, holding and grip pressure read the whole image.
        global_dim = self.global_dim + state_dim
        self.global_head = nn.Sequential(
            # LayerNorm first, or the large [CLS] norm saturates the finger head's tanh.
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 256), nn.GELU(), nn.Linear(256, 4))

        # The close question is asked of the cell grid, where the jaws are, rather than the
        # pooled vector. State also goes straight in after the pool, since the rangefinder
        # is most of "close enough".
        self.close_reduce = nn.Conv2d(channels, CLOSE_CHANNELS, 1)
        close_dim = CLOSE_CHANNELS * CLOSE_POOL[0] * CLOSE_POOL[1] + state_dim
        self.close_head = nn.Sequential(
            nn.LayerNorm(close_dim),
            nn.Linear(close_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, pixel_values, state):
        """Returns a dict of raw head outputs; see decode() for what they mean."""
        x, global_vec = self.cell_map(pixel_values, state)

        flags = self.global_head(torch.cat([global_vec, state], dim=-1))
        out = {
            "logits": self.logit_head(x).squeeze(1),
            "log_distance": self.distance_head(x).squeeze(1),
            "axis": self.axis_head(x),
            "finger": torch.tanh(flags[:, 0]),
            "present_logit": flags[:, 1],
            "holding_logit": flags[:, 2],
            # softplus so the pressure can't go negative
            "grasp_pressure": F.softplus(flags[:, 3]),
        }
        pooled = adaptive_avg_pool2d(F.gelu(self.close_reduce(x)), CLOSE_POOL).flatten(1)
        out["close_logit"] = self.close_head(torch.cat([pooled, state], dim=-1)).squeeze(-1)
        return out


def decode(outputs, grid, top_k=1, nms_radius=CENTROID_RADIUS):
    """Head outputs -> (uv, distance, axis angle, score, concentration), top_k peaks per item."""
    logits = outputs["logits"]
    batch, rows, cols = logits.shape
    prob = logits.flatten(1).softmax(1).view(batch, 1, rows, cols)

    if top_k > 1:
        prob = local_maxima(prob, nms_radius)

    scores, index = prob.flatten(1).topk(top_k, dim=1)
    uv, distance, angle, concentration = [], [], [], []
    for k in range(top_k):
        position, range_m, weights, window = decode_position(
            logits, outputs["log_distance"], grid, index[:, k])
        uv.append(position)
        distance.append(range_m)
        # Average the (sin 2t, cos 2t) vectors so disagreement shortens them.
        axis = window_average(outputs["axis"], weights, window)
        angle.append(torch.atan2(axis[:, 0], axis[:, 1]) / 2.0)
        # The axis vector's length is the head's concentration, near zero meaning no
        # opinion.
        concentration.append(axis.norm(dim=1))
    return (torch.stack(uv, dim=1), torch.stack(distance, dim=1),
            torch.stack(angle, dim=1), scores, torch.stack(concentration, dim=1))


@torch.no_grad()
def predict(model, images, state, top_k=1):
    """Everything the robot wants from one frame, decoded and unbatched-friendly."""
    model.eval()
    outputs = model(images, state)
    uv, distance, angle, scores, concentration = decode(outputs, model.grid, top_k=top_k)
    return {
        "uv": uv,
        "distance_m": distance,
        "point_m": camera_point(uv, distance),
        "grasp_axis_rad": angle,
        "axis_concentration": concentration,
        "score": scores,
        "finger": outputs["finger"],
        "present": outputs["present_logit"].sigmoid(),
        "holding": outputs["holding_logit"].sigmoid(),
        "close": outputs["close_logit"].sigmoid(),
        "grasp_pressure": outputs["grasp_pressure"],
    }


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    freeze = checkpoint.get("freeze", True)
    model = VisualServoNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        fuse_layers=checkpoint["fuse_layers"], attention_layers=checkpoint["attention_layers"],
        freeze=freeze,
    ).to(device)
    load_head_state(model, checkpoint)
    model.eval()
    return model, checkpoint
