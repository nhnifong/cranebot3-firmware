#!/usr/bin/env python

"""One DINOv3 backbone, shared by every model that only reads from it.

OrthoTargetNet and VisualServoNet load into the same observer process, onto the same
device, from the same pretrained id, and both freeze what they load - so the second
copy is a third of a gigabyte holding the same numbers as the first. They still run
separate forward passes, on different cameras at different input shapes, but DINOv3
interpolates its position embeddings per call, so one instance serves both.

Sharing is only sound while the weights stay read-only, so it applies to frozen trunks
only. Training with the backbone unfrozen gets a private copy, which is what stops one
model's fine-tuning from moving the features the other one reads.

A frozen trunk is also kept out of the owning module's _modules, which is what keeps it
out of state_dict. The weights are recoverable from backbone_id and a download, so a
checkpoint that stores them spends 327MB on the least interesting thing in it - and,
worse, restoring them writes through the shared instance into every other model.
"""

import logging
import threading

import torch

TRUNK_PREFIX = "backbone."

_TRUNKS = {}
# The observer loads both models inside asyncio.to_thread, so two misses can race.
# Without the lock each thread builds its own trunk and the loser stays alive in
# whichever model asked for it: no error, and none of the sharing.
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
    """A DINOv3 trunk that is shared and unsaved while it is frozen.

    Mixed in ahead of nn.Module so the _apply override precedes it in the MRO.
    """

    def _init_trunk(self, backbone_id, freeze):
        """Attach the backbone. Call after nn.Module.__init__, before reading self.trunk."""
        if freeze:
            trunk = shared_backbone(backbone_id)
        else:
            from transformers import AutoModel

            trunk = AutoModel.from_pretrained(backbone_id)
        # Held in a bare list so nn.Module.__setattr__ leaves it unregistered, which is
        # what keeps it out of state_dict, parameters() and the optimizer.
        self._trunk = [trunk]
        if not freeze:
            # A fine-tuned trunk is genuinely part of this model, so register it the
            # ordinary way and let it be trained, moved and saved like anything else.
            self.backbone = trunk
        return trunk

    @property
    def trunk(self):
        """The backbone, whether or not it is a registered submodule."""
        return self._trunk[0]

    def _apply(self, fn, *args, **kwargs):
        """Follow .to() and friends into an unregistered trunk.

        It is shared, so this moves it for every owner at once. They all take the
        observer's single eval device, and applying it twice is free - a tensor already
        on the target device comes back unchanged.
        """
        out = super()._apply(fn, *args, **kwargs)
        trunk = self.__dict__.get("_trunk")
        if trunk is not None and TRUNK_PREFIX[:-1] not in self._modules:
            trunk[0]._apply(fn)
        return out


def drop_trunk_weights(state_dict, trunk, verify):
    """A state dict without the trunk weights checkpoints used to carry.

    The trunk was frozen when they were written, so they are the pretrained weights the
    shared instance already holds and restoring them would only write identical values.
    `verify` checks that claim, for checkpoints from before it was recorded: a run with
    --unfreeze_backbone stores weights that are not the pretrained ones, and dropping
    those would quietly pair a fine-tuned head with a stock trunk.
    """
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
