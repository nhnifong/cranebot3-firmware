"""Turning frames into normalized model input, shared by ortho_target and visual_servoing."""

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0


def normalize(img):
    return (img - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denormalize_to_bgr(tensor):
    array = tensor.cpu().numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return cv2.cvtColor((np.clip(array, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def input_batch(rgb, size, device):
    """A live RGB frame as a one-image normalized batch, resized to (width, height)."""
    size = (int(size[0]), int(size[1]))
    if (rgb.shape[1], rgb.shape[0]) != size:
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return normalize(to_tensor(rgb))[None].to(device)


def photometric_jitter(img, rng: torch.Generator):
    """Brightness/contrast/saturation jitter, plus an occasional drop to grayscale."""
    def uniform(lo, hi):
        return torch.empty((), device=img.device).uniform_(lo, hi, generator=rng)

    img = img * uniform(0.7, 1.3)                                  # brightness
    mean = img.mean(dim=(-2, -1), keepdim=True)
    img = (img - mean) * uniform(0.7, 1.3) + mean                  # contrast
    gray = (img * torch.tensor([0.299, 0.587, 0.114], device=img.device).view(3, 1, 1)).sum(0, keepdim=True)
    if torch.rand((), device=img.device, generator=rng) < 0.15:
        img = gray.expand_as(img).clone()
    else:
        img = gray + (img - gray) * uniform(0.6, 1.4)              # saturation
    return img.clamp(0.0, 1.0)
