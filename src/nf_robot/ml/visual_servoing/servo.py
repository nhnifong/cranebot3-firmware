#!/usr/bin/env python

"""Deployment side of the visual servoing model: one gripper frame in, one move out.

The model says *where the object is*, not how fast to go (readme.md has the argument for
the split), so something has to turn its answer into a direction for the gantry. That is
this module: load the checkpoint, feed it a live frame exactly as the training loader fed
it stored ones, and hand back the target as a room-frame offset from the lens, which is
the form observer.visual_servo_grasp closes with a gain.

Preprocessing has to match dataset.py or the frozen backbone sees a different image than
it was trained against: BGR from the video path becomes RGB, the frame is resized to the
checkpoint's input size (never cropped - a crop moves the principal point and invalidates
every geometric label), and the state vector is scaled by the same STATE_SCALES.
"""

import logging

import numpy as np

from nf_robot.ml.visual_servoing.geometry import camera_to_room, point_in_room

logger = logging.getLogger(__name__)

# Where a trained checkpoint is published, and where --local_models looks for it instead.
SERVO_MODEL_REPOID = "naavox/visual_servo"
SERVO_MODEL_FILENAME = "visual_servo.pth"
LOCAL_MODEL_PATH = "models/visual_servo.pth"


def load_model(device, local_models=False):
    """The trained checkpoint on `device`, from models/ or from the hub.

    Returns (model, checkpoint) - the checkpoint carries the metrics and the input size,
    both worth logging when a model reaches a robot.
    """
    from nf_robot.ml.visual_servoing.model import load_checkpoint

    if local_models:
        path = LOCAL_MODEL_PATH
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=SERVO_MODEL_REPOID, filename=SERVO_MODEL_FILENAME)
    logger.info(f"Loading visual servoing model from {path}...")
    return load_checkpoint(path, device)


def prepare_frame(bgr, image_size, device):
    """A live gripper frame as a one-image normalized batch, the way training saw them.

    The frames in the dataset are stored BGR (they went through cv2.imencode) and the
    loader flips them to RGB, so the same flip belongs here; last_output_frame off the
    gripper client is BGR too, since only the video streamer converts.
    """
    import cv2
    import torch

    from nf_robot.ml.ortho_target import IMAGENET_MEAN, IMAGENET_STD

    size = (int(image_size[0]), int(image_size[1]))
    if (bgr.shape[1], bgr.shape[0]) != size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    img = (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return img[None].to(device)


def predict_frame(model, bgr, state, device, spin, gripper_pos=None):
    """Everything one frame says, decoded into the frames the robot thinks in.

    `state` is the dict the dataset row carries: laser_rangefinder, finger_angle and
    target_force. `spin` is the gripper camera's heading in the room, i.e. get_spin().

    Blocking (it runs the network), so call it from a thread.

    Keys out:
        uv              normalized frame coordinates, -0.25..1.25; outside 0..1 means
                        the target is off the edge of the visible frame
        range_m         distance from the lens along the ray
        point_cam       the target in the camera's optical frame, metres
        room_offset     the same vector in room axes; its horizontal part is the
                        centering error, measured from the lens
        point_room      where that lands in the room, if a gripper position was given
        axis_concentration  how sure the axis head is, in von Mises kappa; near zero
                        means it has no opinion and a wrist gate should not act on it
        grasp_axis_rad  the line the jaws should close along, in the image plane and
                        pi-periodic: zero means the object already stands upright in
                        frame, which is what the wrist is turned to achieve. Turning the
                        wrist by it is only right if the mount's sign agrees - see
                        synth_frames.AXIS_FROM_WRIST_SIGN, which is where the training
                        labels commit to a direction.
        finger          commanded finger speed in -1..1, positive meaning more grip
        close           close-heads models only: probability the close should have begun
                        by this frame
        grasp_pressure  close-heads models only: the grip force this object turned out to
                        need, in the finger sensor's own units
        present         probability there is anything graspable in view at all
        holding         probability we are already holding something
        score           the winning cell's share of the position softmax
    """
    import torch

    from nf_robot.ml.visual_servoing.dataset import state_vector
    from nf_robot.ml.visual_servoing.model import predict

    images = prepare_frame(bgr, model.image_size, device)
    state_t = torch.from_numpy(state_vector(state))[None].to(device)
    out = predict(model, images, state_t, top_k=1)  # predict is already no_grad

    point_cam = out["point_m"][0, 0].cpu().numpy().astype(np.float64)
    result = {
        "uv": out["uv"][0, 0].cpu().numpy().astype(np.float64),
        "range_m": float(out["distance_m"][0, 0]),
        "point_cam": point_cam,
        "room_offset": camera_to_room(point_cam, spin),
        "point_room": None if gripper_pos is None else point_in_room(point_cam, gripper_pos, spin),
        "grasp_axis_rad": float(out["grasp_axis_rad"][0, 0]),
        # Length of the axis vector, which train.py's von Mises objective fits as the
        # head's concentration. Near zero is "no opinion about orientation", which is a
        # different thing from "upright" even though both decode to the same angle.
        "axis_concentration": float(out["axis_concentration"][0, 0]),
        "finger": float(out["finger"][0]),
        "present": float(out["present"][0]),
        "holding": float(out["holding"][0]),
        "score": float(out["score"][0, 0]),
    }
    # Only a close-heads checkpoint has these, so their absence is how a consumer tells
    # which kind of model it is holding.
    if "close" in out:
        result["close"] = float(out["close"][0])
        result["grasp_pressure"] = float(out["grasp_pressure"][0])
    return result
