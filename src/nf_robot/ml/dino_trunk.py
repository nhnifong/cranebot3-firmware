#!/usr/bin/env python

"""One frozen DINO backbone shared by every model that reads from it, and kept out of their
checkpoints."""

import logging
import threading

import torch

TRUNK_PREFIX = "backbone."

_TRUNKS = {}
# The observer loads models from threads, so without the lock two could build separate
# trunks.
_LOCK = threading.Lock()


def shared_backbone(backbone_id):
    """The one frozen backbone for this id, built on first use."""
    with _LOCK:
        trunk = _TRUNKS.get(backbone_id)
        if trunk is None:
            from transformers import AutoModel

            logging.info(f"Loading shared frozen backbone {backbone_id}")
            trunk = AutoModel.from_pretrained(backbone_id)
            trunk.requires_grad_(False)
            trunk.eval()
            _TRUNKS[backbone_id] = trunk
        return trunk


class SharedTrunkMixin:
    """A DINO trunk that is shared and left out of state_dict while frozen."""

    def _init_trunk(self, backbone_id, freeze):
        """Attach the backbone; call after nn.Module.__init__."""
        if freeze:
            trunk = shared_backbone(backbone_id)
        else:
            from transformers import AutoModel

            trunk = AutoModel.from_pretrained(backbone_id)
        # A bare list keeps nn.Module from registering the trunk, so it stays out of
        # state_dict and the optimizer.
        self._trunk = [trunk]
        if not freeze:
            # A fine-tuned trunk is registered normally so it trains and saves with the
            # model.
            self.backbone = trunk
        return trunk

    @property
    def trunk(self):
        return self._trunk[0]

    def patch_token_map(self, pixel_values, rows, cols):
        """The last fuse_layers patch tokens as a (B, C, rows, cols) map, plus the last
        hidden state."""
        with torch.set_grad_enabled(self.training and not self.freeze):
            out = self.trunk(pixel_values, output_hidden_states=True)
        n_patches = rows * cols
        # [CLS] and the register tokens lead the sequence; patches are always the tail.
        feats = [h[:, -n_patches:, :] for h in out.hidden_states[-self.fuse_layers:]]
        x = torch.cat(feats, dim=-1).transpose(1, 2)
        return x.reshape(x.shape[0], x.shape[1], rows, cols), out.hidden_states[-1]

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.trunk.eval()  # a frozen backbone must not update its norm statistics
        return self

    def _apply(self, fn, *args, **kwargs):
        """Follow .to() into an unregistered trunk, moving it for every owner at once."""
        out = super()._apply(fn, *args, **kwargs)
        trunk = self.__dict__.get("_trunk")
        if trunk is not None and TRUNK_PREFIX[:-1] not in self._modules:
            trunk[0]._apply(fn)
        return out


def drop_trunk_weights(state_dict, trunk, verify):
    """A state dict without trunk weights, verifying they are pretrained when asked."""
    kept = {key: value for key, value in state_dict.items() if not key.startswith(TRUNK_PREFIX)}
    if len(kept) == len(state_dict) or not verify:
        return kept

    pretrained = trunk.state_dict()
    for key, value in state_dict.items():
        if not key.startswith(TRUNK_PREFIX):
            continue
        reference = pretrained.get(key[len(TRUNK_PREFIX):])
        if reference is None or not torch.equal(reference, value.to(reference.device)):
            raise ValueError(
                f"this checkpoint's backbone differs from the pretrained weights at "
                f"'{key}', so it was trained with --unfreeze_backbone and its head only "
                f"means anything on top of its own trunk. Re-save it with "
                f"\"freeze\": False so it loads its backbone instead of sharing one."
            )
    return kept


def load_head_state(model, checkpoint):
    """Restore a checkpoint's weights into a model built with the checkpoint's freeze flag."""
    state = checkpoint["state_dict"]
    if model.freeze:
        # Old checkpoints still carry the trunk; verify it is pretrained if they don't say
        # so.
        state = drop_trunk_weights(state, model.trunk, verify="freeze" not in checkpoint)
    model.load_state_dict(state)
