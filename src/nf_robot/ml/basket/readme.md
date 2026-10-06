# Basket centering

The placer network can see a laundry hamper from the overhead views, but its prediction is
not close enough to drop into. To get an exact fix, the robot flies over with the gripper
empty, looks down at the basket, and works out where it would drop clothes from. This model
does the looking: from one gripper frame it predicts the gripper move that centres the
basket under the jaws at the right drop height.

`find_basket()` in [host/maneuvers/find_basket.py](../../host/maneuvers/find_basket.py)
wraps it. It looks, moves most of the way, and looks again until the model says it is
centred, then returns the gantry position to drop from.

## How the labels are made

Each episode is recorded by the `basketdata` maneuver. It starts with the gripper still
and centred over the basket at the ideal drop height, then flies away in one move.

- The first frame defines centred. The point straight below the jaws, at the rangefinder
  distance, is the basket's drop point.
- The miner follows that point forward with optical flow (the same tracker as the
  visual servoing `optical-flow` uv method) until the flow loses it.
- Each frame it was tracked through is labelled with:
  - where the point is in the image: (u, v, distance along the ray);
  - the gripper move back to centre, in the gripper frame (`return_body`).
- The sideways part of the move comes from the flow. It is scaled by the point's depth:
  the first frame's rangefinder reading plus how far the gantry has climbed since then.
- The vertical part is the gantry's height change alone.

Nothing vertical is taken from the rangefinder. Over a basket of clothes it reads into
gaps and onto piles rather than the level the image moves with. On basketdata-2 it read a
median 12.6 cm deeper than the depth at which the flow agrees with the gantry positions,
with episodes ranging from -2.5 to +39 cm. Once the gripper is off to one side it reads
the floor anyway. The rangefinder does still set the depth that scales the sideways labels;
the miner's `lateral flow vs gantry position` log line shows what that costs.

The model reuses the visual servoing trunk (`ml/gripper_grid.py`) and position head. The
sideways move is decoded from the predicted drop point with the same function the miner
labels with (`basket/geometry.py`). The vertical move is a head of its own, trained
directly on the gantry's height change.

## Commands

### 1. Record episodes

Run the robot (`stringman-headless`, or the full UI) and start a lerobot recording
session. Either use the UI's lerobot panel, or run:

```sh
python -m nf_robot.ml.lerobot.stringman record \
    --repo_id naavox/basketdata-2 --server_address ws://localhost:4245
```

Fly the gripper to an ideal drop position over a basket, then type in the debug box:

```
basketdata
```

It records 50 episodes by default (`basketdata 20` for fewer). Stop it at any time; that
ends the session and uploads the dataset. Move the basket, or use a different one, and
repeat to get variety. More recordings can go into the same repo.

### 2. Mine

```sh
python -m nf_robot.ml.basket.mine \
    --repo_id naavox/basketdata-2 \
    --output_root datasets/basket_centering \
    --preview_dir datasets/basket_centering/preview
```

This writes the labelled pool to `datasets/basket_centering/all/`. It then splits the pool
into `train/` and `eval/` by whole episodes, with 15% of episodes going to eval. Several
`--repo_id`s can be mined together. `--limit 5` mines only the first five episodes, for a
quick look.

Before training, check two things:

- **The log line `lateral flow vs gantry position`.** It compares the move the flow
  implies with the move the recorded gantry positions imply. Both should agree to a few
  centimetres; a large 90th percentile means the flow slid off the basket somewhere.
- **The preview frames.** The crosshair should stay on the same spot of the basket, and the
  `flow` and `pos` lines should roughly agree.

### 3. Upload the mined dataset (optional)

Only needed to train on another machine without `--data_root`:

```sh
hf upload naavox/basket_centering datasets/basket_centering --repo-type dataset
```

### 4. Train

```sh
python -m nf_robot.ml.basket.train \
    --data_root datasets/basket_centering --epochs 30 --select_best
```

This writes `models/basket_center.pth`. Each epoch reports:

- `lateral_median_cm` and `vertical_median_cm`: the error of the decoded move on eval.
- `stay_put_lateral_cm`: what never moving would score. The model has to beat this by a
  wide margin.
- `lateral@5cm`: the fraction of eval frames whose move lands within 5 cm.

### 5. Try it on the robot

Start the robot with `--local_models` so it reads `models/basket_center.pth` instead of the
hub. That flag applies to every model, so `models/` also needs the other checkpoints the
robot uses (`visual_servo.pth`, ...).

```sh
stringman-headless --local_models
```

In the debug box, use one of:

- `findbasket` searches from wherever the gripper is. Fly it near the basket first.
- `findbasket X Y` flies over the floor point (X, Y) first.
- `findbasket hamper` flies over a named position first.

The fix appears in the UI as the `basket_fix` named position. The log has each look's
predicted move.

### Fixes in pick and place

`fixdrops` in the debug box turns on an optional pick and place mode (off by default;
`fixdrops on`, `fixdrops off`, and `fixdrops forget` to throw away every stored fix). In it,
pick and place gets a fix on each destination before relying on it, and drops from the fix:

- **Named destinations** (hamper, toys, trash, ...): visited with the gripper empty, before
  the next item is picked up. Each fix is stored beside the tag position it was taken from.
  When the tag is later seen more than 30 cm away, the fix counts as stale and the place is
  visited again.
- **Predicted drops** from the placer model: the prediction only comes once the item is
  seen up close, mid grasp. The predicted point is pooled into a 30 cm room grid square. If
  that square has no fix, the grasp is set aside once: the gripper visits the square, gets
  the fix, flies back to where it left off and resumes the grasp.
- **The recorded drop position and the origin** are never visited.

A place where `find_basket` finds nothing is not retried until the next pick and place run.
Fixes are kept in the `find_basket` maneuver's settings, so they outlast a restart, and show
in the UI as `<name>_fix` markers. `findbasket NAME` stores its result as NAME's fix too.

### 6. Publish

```sh
hf upload naavox/basket_center models/basket_center.pth basket_center.pth
```

The first time only, add the model to the pinning tool. In
[ml/pin_latest_model.py](../pin_latest_model.py), add this line to `MODELS`:

```python
"basket_center": ("nf_robot.ml.basket.model", "BASKET_MODEL_REPOID"),
```

Then pin it, and commit the change to `common/model_revisions.json`:

```sh
python -m nf_robot.ml.pin_latest_model --basket_center
```

Until a pin exists, `findbasket` only works with `--local_models`.
