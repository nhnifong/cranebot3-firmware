#!/usr/bin/env python

"""Deployment side of the visual servoing model: one gripper frame in, a room-frame target
offset out."""

import logging

import numpy as np

from nf_robot.ml.visual_servoing.geometry import (
    camera_to_room, lens_to_jaw_body, point_in_room, rotate_about_vertical,
)

logger = logging.getLogger(__name__)

# Where a trained checkpoint is published, and where --local_models looks for it instead.
SERVO_MODEL_REPOID = "naavox/visual_servo"
SERVO_MODEL_FILENAME = "visual_servo.pth"
LOCAL_MODEL_PATH = "models/visual_servo.pth"


def load_model(device, local_models=False, revision=None):
    """The trained checkpoint on `device`, from models/ or from the hub at `revision`, as
    (model, checkpoint)."""
    from nf_robot.ml.visual_servoing.model import load_checkpoint

    if local_models:
        path = LOCAL_MODEL_PATH
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=SERVO_MODEL_REPOID, filename=SERVO_MODEL_FILENAME,
                               revision=revision)
    logger.info(f"Loading visual servoing model from {path}...")
    return load_checkpoint(path, device)


def prepare_frame(bgr, image_size, device):
    """A live BGR gripper frame as a one-image normalized batch, the way training saw them."""
    import cv2

    from nf_robot.ml.image_input import input_batch

    return input_batch(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), image_size, device)


def predict_frame(model, bgr, state, device, spin, gripper_pos=None):
    """Everything one frame says, decoded into camera and room frames; blocking, so call it
    from a thread."""
    import torch

    from nf_robot.ml.visual_servoing.dataset import state_vector
    from nf_robot.ml.visual_servoing.model import predict

    images = prepare_frame(bgr, model.image_size, device)
    state_t = torch.from_numpy(state_vector(state))[None].to(device)
    out = predict(model, images, state_t, top_k=1)  # predict is already no_grad

    point_cam = out["point_m"][0, 0].cpu().numpy().astype(np.float64)
    return {
        "uv": out["uv"][0, 0].cpu().numpy().astype(np.float64),
        "range_m": float(out["distance_m"][0, 0]),
        "point_cam": point_cam,
        "room_offset": camera_to_room(point_cam, spin),
        # The heads answer in the camera frame, but the grip happens at the jaws.
        "jaw_offset": camera_to_room(point_cam, spin) - rotate_about_vertical(
            lens_to_jaw_body(), -spin),
        "point_room": None if gripper_pos is None else point_in_room(point_cam, gripper_pos, spin),
        "grasp_axis_rad": float(out["grasp_axis_rad"][0, 0]),
        # The axis vector's length is the head's concentration; near zero means no opinion.
        "axis_concentration": float(out["axis_concentration"][0, 0]),
        "finger": float(out["finger"][0]),
        "present": float(out["present"][0]),
        "holding": float(out["holding"][0]),
        "score": float(out["score"][0, 0]),
        "close": float(out["close"][0]),
        "grasp_pressure": float(out["grasp_pressure"][0]),
    }
