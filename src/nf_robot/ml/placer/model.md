# Drop point model

Predicts where an item goes: one softmax over the overhead view's cells, from a snapshot
of the item taken before it was grasped. [model.py](model.py) is the network,
[dataset.py](dataset.py) the loader, [train.py](train.py) the training and scoring. The
data is the drop pair format in [dataset.md](dataset.md).

    python -m nf_robot.ml.placer.train --data_root datasets/drop_pairs --epochs 30

## Shape

Both images go through the same frozen DINOv2-with-registers trunk that visual servoing
and ortho targeting share (ml/dino_trunk.py), the last four hidden states concatenated:
the overhead view at 448x448 gives a 32x32 token map, the item snapshot at 448x252 gives
18x32. A 1x1 convolution takes each to 256 channels.

The item then reaches the room map two ways, and `--merge` picks which:

- **film**: the item's [CLS] and register vector drives a FiLM scale and shift over the
  whole overhead map, so every cell is conditioned on what is being carried.
- **tokens**: the item map is average-pooled to 3x4 = 12 tokens and carried alongside the
  1024 overhead tokens through the attention, so cells can attend to parts of the item.
- **both** (default).

Three self-attention blocks (ml/grid_head.py) run over that sequence with a learned
position embedding, the overhead tokens come back out, and a skip connection concatenates
the pre-attention map with the attended one and fuses it 1x1 - the same decoder shape as
VisualServoNet. One 1x1 convolution gives a logit per cell. 4.6M trained parameters.

Which merge is right is an open question; nothing here has been ablated.

## Reading out the position

As in visual servoing: softmax over the 1024 cells, take the winning cell, and the answer
is the centre of mass of the softmax in a 2-cell window around it. A cell is 5m / 32 =
15.6cm of floor, so the sub-cell centroid is what puts the answer inside the container
rather than merely on it. `top_k > 1` suppresses everything that is not a local maximum
first, so several candidate drop points come back as distinct places.

Training is that cross-entropy against a Gaussian target of `--cell_sigma` cells, plus a
smooth L1 on the windowed centroid so it is trained directly rather than left to fall out.

## Augmentation

The overhead view is a metric top-down map with no canonical orientation, so all 8
symmetries of the square - 4 rotations, each optionally mirrored - are rooms the robot
could have been in, and the label moves with the image by the same rule. They are exact,
whole-pixel moves with no resampling. The item snapshot gets the mirror only, since the
gripper camera hangs one way up. Both get photometric jitter.

## Scoring

Error is the distance from the predicted drop point to the one the operator used, in
centimetres of floor. Reported as `median_cm`, `mean_cm`, recall inside 15/30/60cm, and
`top3@30cm`, which asks whether any of three candidates was right.

Two guards against fooling yourself, both printed by a run:

- **The constant-prediction baseline**, first: always dropping at the mean training drop
  point. On `datasets/drop_pairs` that is 113cm median, and a model must clearly beat it.
- **`--item_check`**, after training: the same eval with each room paired with another
  row's item. The gap is how much the model reads the item rather than the room's usual
  drop point. On a 2-epoch run the median went 17.4 -> 17.9cm, which is almost no gap -
  it is mostly answering "the usual spot for this room". Expect that: in
  `datasets/drop_pairs` the hamper is half the rows, and the overhead view alone says
  where the containers are.

Per-task medians print each epoch for the same reason. On that run the hamper and trash
can sat near 13cm while the toybox was 22cm and "put stuffies on the bed" - 4 eval rows -
was 217cm.

## Publishing

The robot downloads this model from `naavox/drop_point` at a fixed commit, the same way as
the other two (common/model_revisions.json), so publishing alone changes nothing until the
pin moves.

Try it first with the checkpoint training just wrote, which is what `--local_models` reads:

    stringman-headless --local_models

Then publish, and move the pin and commit it:

    hf upload naavox/drop_point models/drop_point.pth drop_point.pth
    python -m nf_robot.ml.pin_latest_model --drop_point

Both are needed before a robot can load it from the hub at all, and
tests/model_revisions_test.py fails until the pin exists - the same rule the other two
models follow.

## On the robot

The observer loads it when a pick and place starts and predicts every 0.25s while the
laser says the item is 0.12-0.25m away, which is the range the snapshots were mined at.
The answer is saved as the `predicted_drop` named position, drawn in the 3D view as a dark
green pyramid, and flown to when the route destination is "Stringman's choice" - which
aims 1.5m above the predicted point, since the prediction is on the floor and nothing
predicts a drop height yet. The `droppoint` debug command runs the same loop on its own.

## Caveats

- Episodes are dealt to train and eval inside the loader, by episode. One room's drop
  points appear on both sides, so the score measures "same room, another item", not a new
  room. There is nothing in the data yet to measure the latter.
- The task strings are not normalized: "put a toy in the toybox" and "put a toy in the toy
  box" are separate strings, as are capitalised variants.
- Nothing predicts drop height yet, though `release_height_m` is in the dataset.
