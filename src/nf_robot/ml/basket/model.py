#!/usr/bin/env python

"""The basket centering network: one gripper frame in, the move that centres a basket out.

    input   448x252 RGB from the gripper camera, plus laser_rangefinder, finger_angle and
            target_force, exactly as the visual servoing model takes them

    trunk   gripper_grid.GripperGridNet, the visual servoing trunk

    heads   1. the basket's drop point, 3D, in the gripper camera frame: a softmax over the
               canvas cells, the centre of mass around the winner, and a log distance per
               cell - the visual servoing position head, aimed at a basket
            2. the vertical move to the drop height, metres, from the global vector

The sideways move back to centre is decoded from head 1: where the basket is, less straight
below the jaws, by geometry.lateral_return_body, the same function the miner labels with.
The vertical move is head 2 as it stands, trained on the gantry's height change. Taking it
as a difference of two depths instead would hang it on the rangefinder, which reads into
the gaps and onto the piles of a basket of clothes.
"""

import numpy as np
import torch
from torch import nn

from nf_robot.ml.dino_trunk import load_head_state
from nf_robot.ml.gripper_grid import (
    DEFAULT_BACKBONE, DEFAULT_IMAGE_SIZE, GripperGridNet, decode_position,
)

# Where a trained checkpoint is published, and where --local_models looks for it instead.
BASKET_MODEL_REPOID = "naavox/basket_center"
BASKET_MODEL_FILENAME = "basket_center.pth"
LOCAL_MODEL_PATH = "models/basket_center.pth"
# laser_rangefinder, finger_angle, target_force, as visual_servoing.dataset.state_vector.
STATE_DIM = 3


class BasketNet(GripperGridNet):
    """Frozen DINOv2 patch features -> where the basket is, and how far up or down the
    drop height is."""

    def __init__(self, backbone_id=DEFAULT_BACKBONE, image_size=DEFAULT_IMAGE_SIZE,
                 fuse_layers=4, width=256, attention_layers=3, heads=8, freeze=True,
                 state_dim=STATE_DIM):
        super().__init__(backbone_id=backbone_id, image_size=image_size,
                         fuse_layers=fuse_layers, width=width,
                         attention_layers=attention_layers, heads=heads, freeze=freeze,
                         state_dim=state_dim)
        self.logit_head = nn.Conv2d(width, 1, 1)
        self.distance_head = nn.Conv2d(width, 1, 1)
        global_dim = self.global_dim + state_dim
        self.global_head = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, pixel_values, state):
        x, global_vec = self.cell_map(pixel_values, state)
        return {
            "logits": self.logit_head(x).squeeze(1),
            "log_distance": self.distance_head(x).squeeze(1),
            # signed: positive is up
            "rise": self.global_head(torch.cat([global_vec, state], dim=-1)).squeeze(-1),
        }


def decode(outputs, grid):
    """Head outputs -> (uv, distance along the ray, vertical move, score) at the strongest
    cell."""
    logits = outputs["logits"]
    prob = logits.flatten(1).softmax(1)
    score, index = prob.max(dim=1)
    uv, distance, _, _ = decode_position(logits, outputs["log_distance"], grid, index)
    return uv, distance, outputs["rise"], score


@torch.no_grad()
def predict(model, images, state):
    model.eval()
    uv, distance, rise, score = decode(model(images, state), model.grid)
    return {"uv": uv, "distance_m": distance, "rise_m": rise, "score": score}


def return_body_of(uv, distance, rise, calibration=None):
    """The move back to centre, in the body frame, for decoded (N, 2) uv, (N,) distance and
    (N,) vertical move. Unprojects through the same calibration the miner labels with."""
    from nf_robot.ml.basket.geometry import lateral_return_body
    from nf_robot.ml.visual_servoing.uv_methods import gripper_camera_calibration, unproject

    calibration = calibration or gripper_camera_calibration()
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    points = np.stack([unproject(u, v, d, calibration) for (u, v), d in zip(uv, distance)])
    rise = np.asarray(rise, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([lateral_return_body(points), rise], axis=1)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = BasketNet(
        backbone_id=checkpoint["backbone_id"], image_size=checkpoint["image_size"],
        fuse_layers=checkpoint["fuse_layers"], attention_layers=checkpoint["attention_layers"],
        freeze=checkpoint.get("freeze", True),
    ).to(device)
    load_head_state(model, checkpoint)
    model.eval()
    return model, checkpoint


def load_model(device, local_models=False, revision=None):
    """The trained checkpoint on `device`, from models/ or from the hub at `revision`, as
    (model, checkpoint)."""
    if local_models:
        path = LOCAL_MODEL_PATH
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=BASKET_MODEL_REPOID, filename=BASKET_MODEL_FILENAME,
                               revision=revision)
    return load_checkpoint(path, device)


def predict_frame(model, rgb, state, device, spin):
    """What one RGB gripper frame says about the basket, with the move to centre it in both
    the body and room frames; blocking, so call it from a thread."""
    from nf_robot.ml.basket.geometry import body_to_room
    from nf_robot.ml.image_input import input_batch
    from nf_robot.ml.visual_servoing.dataset import state_vector

    images = input_batch(rgb, model.image_size, device)
    state_t = torch.from_numpy(state_vector(state))[None].to(device)
    out = predict(model, images, state_t)
    uv = out["uv"][0].cpu().numpy().astype(np.float64)
    distance = float(out["distance_m"][0])
    rise = float(out["rise_m"][0])
    move = return_body_of(uv, distance, rise)[0]
    return {
        "uv": uv,
        "range_m": distance,
        "score": float(out["score"][0]),
        "return_body": move,
        "return_room": body_to_room(move, spin),
    }
