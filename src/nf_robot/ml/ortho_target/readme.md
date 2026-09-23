# ortho_target

Predict everywhere the robot could reach next, in the ortho floor view's pixel space.

The map is one independent objectness per cell, so a frame holding four graspable things
can report four of them and the scores mean the same in every frame; see OrthoTargetNet
and objectness_loss in [model.py](model.py), which carries the consequence of labelling
only one of those four.

The labels come from teleop recordings: wherever an operator actually grasped something
is by construction a place worth reaching for, so every episode donates one label for
free, with no hand labelling.

| file | what it holds |
| --- | --- |
| [model.py](model.py) | the network, its loss, decoding, the checkpoint loader and live inference |
| [dataset.py](dataset.py) | distilling, hand-label merging, splitting, uploading, and the training loader |
| [training.py](training.py) | training, scoring and previews |
| [\_\_main\_\_.py](__main__.py) | the command line: `python -m nf_robot.ml.ortho_target <step>` |
| [labeler/](labeler/README.md) | the browser page for hand-labelling complete frames: `python -m nf_robot.ml.ortho_target.labeler` |

## Pipeline

1. `distill` reduces that to a handful of samples per episode - ortho frames from before
   the grasp and the ortho pixel where contact eventually happened - which is a few
   hundred MB rather than a few hundred GB. One run over the whole dataset, into the pool
   that step 4 deals from:

   ```
   python -m nf_robot.ml.ortho_target distill
   ```

   Then add empty-floor frames from any dataset recorded with nothing graspable in view,
   one frame every 4 seconds of each episode. They join the pool as complete frames with
   no targets, so every cell of them is a negative. Each source owns the pool files named
   for it and a re-run replaces them; distill rebuilds the whole pool, so run this after it:

   ```
   python -m nf_robot.ml.ortho_target distill_negatives --repo_id naavox/combined_negatives
   ```

2. Merge the hand labels into the same pool. They are the only frames where every target
   is marked, which is what the objectness head needs and what makes the selection metric
   computable at all. They are made with [the labeler](labeler/README.md) and land in
   `ortho_target_user_labels/`. Repeat the repo_id line for each volunteer:

   ```
   python -m nf_robot.ml.ortho_target merge_labels
   python -m nf_robot.ml.ortho_target merge_labels --repo_id naavox/ortho-target-user-labels
   ```

3. `split` deals the pool into train and eval, one row at a time and at random:

   ```
   python -m nf_robot.ml.ortho_target split --upload
   ```

   The split lands here, downstream of the merge, so that hand labels reach eval in
   proportion - a split made upstream in LeRobot would put every one of them on one side,
   because they arrive after it. --eval_fraction and --seed are the only knobs, the same
   seed deals the same split, and a re-deal costs no re-distilling. See split_pool for
   what a row-level random cut does and does not measure.

4. `train` fits the model, saving the best checkpoint by ap@20cm to
   models/ortho_target.pth:

   ```
   python -m nf_robot.ml.ortho_target train --data_root ortho_target_data
   ```

   The dataset resizes to whatever --image_size asks for, and the checkpoint records the
   backbone id, image size and the operating point it scored best at, so `evaluate` and
   the robot both need no flags. See [the DINOv3 footnote](#footnote-dinov3) for the
   backbone this used to default to.

5. `evaluate` scores that checkpoint - or the published one, downloaded, if training has
   not run on this machine - and --preview_dir draws what it actually predicted: the
   labels in green, the ranked candidates in red. Numbers say whether it is right, the
   previews say whether it is right for the right reason, which is the check worth doing
   before a model reaches a robot:

   ```
   python -m nf_robot.ml.ortho_target evaluate --tta --preview_dir previews
   ```

   Every run prints a sweep of the operating point - the sensitivity dial - as found,
   precision, invented per frame and missed per frame at each threshold. The checkpoint
   carries the one training scored best, and that is the number the robot runs at
   (observer reads model.threshold), but best F1 is a compromise between the two failures
   and a robot rarely wants them weighted equally: a missed object is left on the floor
   until the next pass, an invented one sends the gantry across the room for nothing.
   --threshold rescores at another point, and the previews then draw what it calls in red
   and what it leaves behind in grey:

   ```
   python -m nf_robot.ml.ortho_target evaluate --threshold 0.30 --preview_dir previews
   ```

   The dial only re-cuts the ranking it is given; where the sweep shows found flattening
   out, the targets below it are not being ranked low, they are not being found at all,
   and no threshold recovers them. Training saves the swept threshold into the
   checkpoint, so shipping a different one means training again with
   --select_metric chosen to favour the side you want.

6. Try it on a robot before publishing. --local_models makes the observer load
   models/ortho_target.pth - where training just wrote it - instead of the hub copy. This
   is the only target model; the UI's targeting switch loads it:

   ```
   stringman-headless --local_models
   ```

7. Publish for stringman users. Until this is done, anyone without --local_models is
   still on the previously published checkpoint:

   ```
   hf upload naavox/targeting models/ortho_target.pth ortho_target.pth
   python -m nf_robot.ml.pin_latest_model --targeting
   ```

   It prints the commit title it pins to, so check that is the upload you meant;
   --dry_run says what would change without writing it.

## Labels from the UI

Labels can also come from the UI instead of a recording, and they are the only frames
where every target is marked - the shape this head wants, and one the teleop labels
cannot supply, since an episode confirms one grasp and says nothing about the rest of the
floor. The RUN menu's "Add targets to dataset" saves the ortho frame the robot is looking
at and every target placed on it by hand, as one row in exactly the format step 2 writes,
into ortho_target_user_labels/ - relative to the directory stringman was started from.

Back them up, or gather several robots' worth in one place, on your own account. The
repo is created private, since these are pictures of your floor:

```
python -m nf_robot.ml.ortho_target upload_labels
```

Merge them from that directory, or from any hub repo full of them:

```
python -m nf_robot.ml.ortho_target merge_labels
python -m nf_robot.ml.ortho_target merge_labels --repo_id you/ortho-target-user-labels
```

Merged files keep the names they arrived under, so `rm ortho_target_data/all/user-*.parquet`
undoes it and merging twice overwrites rather than duplicates. Step 2 rewrites the pool
from scratch, so a re-distill drops them and the merge has to run again after it - and so
does the split, which is downstream of both.

How much they matter is out of proportion to how many there are. They are where every
negative in the dataset comes from: objectness_loss trains a teleop frame's unlabelled
cells on nothing at all, because the objects the operator did not reach for are in them,
so only a frame somebody marked exhaustively can say where the floor is empty. A dataset
with none of them cannot be trained on, and one with none of them in eval cannot be
scored - ap@20cm needs frames where an unmatched detection is known to be wrong. train
raises rather than proceed in either case.

## Training on the complete frames alone

A teleop frame's one label is true but partial: every other object on that floor sits in
a cell the loss trains on nothing, and the model is left to guess from the label density
whether an unmarked object is a target it should call or one nobody happened to reach
for. That is the suspect when a model invents targets. Hand labels and empty-floor frames
have no such gap - every cell of them is a yes or a no - so a dataset of nothing else is
the experiment that says how much the partial labels are costing.

Nothing needs a flag: the pool is whatever was written into it, and `merge_labels` and
`distill_negatives` write into it without `distill`. Build the dataset in its own
directory, so the ordinary one stays where it is:

```
python -m nf_robot.ml.ortho_target merge_labels --output datasets/ortho_target_complete
python -m nf_robot.ml.ortho_target merge_labels --repo_id naavox/ortho-target-user-labels --output datasets/ortho_target_complete
python -m nf_robot.ml.ortho_target distill_negatives --output datasets/ortho_target_complete
python -m nf_robot.ml.ortho_target split --data_root datasets/ortho_target_complete
```

Repeat the merge line for each label repo. The split log says how many frames it dealt
and that all of them are complete, which is the check that `distill` did not run here by
mistake.

Then train it into a checkpoint of its own, and score it against the ordinary one:

```
python -m nf_robot.ml.ortho_target train --data_root datasets/ortho_target_complete \
    --model_path models/ortho_target_complete.pth --epochs 400
python -m nf_robot.ml.ortho_target evaluate --data_root datasets/ortho_target_complete \
    --model_path models/ortho_target_complete.pth --preview_dir previews_complete
```

Raise the epochs, because this dataset is about a fifteenth of the ordinary one - 460
frames against 7450, so an epoch is 15 optimizer steps rather than 233 - and the default
60 stops in what amounts to a warmup. --epochs 400 costs about the same wall clock as the
ordinary run. The balanced pos_weight climbs with the frames (196 rather than 47), since
every cell of every frame is now supervised and the negatives outnumber the positives by
far more; leave it computed rather than carried over from the other dataset.

Two things make the comparison less than a clean A/B, and both flatter this dataset.
Its eval split is drawn from the same hand-labelled pool, so a model trained on complete
frames is scored on the distribution it was fitted to, while the ordinary model is scored
on a stranger; and the labelling runs put near-duplicate frames on both sides of the
split, which the split log counts. Compare the two checkpoints on one eval set - the
ordinary one - if the question is which model to ship, rather than each on its own.

## Frozen backbone

Train with the backbone frozen, which is what step 5 does by default. --unfreeze_backbone
exists but is not the supported path, for three reasons that all point the same way: a few
thousand samples is far too few to move an 86M-parameter trunk without memorising the
floors it saw; frozen is what lets the observer serve this model and the visual servoing
one from a single shared trunk (ml/dino_trunk.py) rather than two copies of the same
327MB - they default to the same backbone id, which is what makes that sharing happen;
and a frozen trunk is recoverable from backbone_id and a download, so it stays out of the
checkpoint and ortho_target.pth carries heads alone. A checkpoint trained unfrozen records
"freeze": False, loads a private trunk and shares nothing - the model still runs, it just
costs what it used to.

## Geometry

The ortho view is an orthographic projection of the floor plane (host/floor_view.py), so
room metres map to its pixels analytically - no camera pose is involved.

## What eval measures

The eval split is a random sample of rows, taken after everything has been merged (see
split_pool). Worth knowing what that measures: an episode donates eight frames of one
floor carrying one label and a labelling run donates three, so most eval rows have a near
duplicate sitting in train, and the score says how well the model does on further frames
of rooms it has trained on rather than on a room it has not. That is the right comparison
for choosing between checkpoints of one run and an optimistic one for predicting a new
room. `split` logs how many eval rows have a same-episode or same-run sibling in train, so
the size of that effect is on the record for every deal.

## Caveats

Worth knowing before trusting the labels:

- The projection assumes contact happens on the floor plane. An object grasped at height
  z appears displaced in the ortho view by the anchor cameras' parallax, so the label
  drifts outward from the room centre as z grows. contacts_m carries the full 3D
  position, so samples can be filtered on z later.
- Contact position is recomputed here from observation.state rather than read from the
  labelled contact_vec_* action components. Both use the same definition (see
  contact_blend_alphas), but an episode that never reaches the pressure threshold gets a
  zeroed contact_vec, which is indistinguishable from "contact at the starting position".
  Reading the pressure directly is what makes those episodes skippable instead of
  silently mislabelled.

## Footnote: DINOv3

The backbone was facebook/dinov3-vitb16-pretrain-lvd1689m at 512 until 2026-08-23. It is
gated - an approved access request plus an HF_TOKEN on every machine that trains or runs
the model - and DINOv2 with registers scores the same on the visual servoing task, so
there is no reason to prefer it. To train against it anyway:

```
python -m nf_robot.ml.ortho_target train \
    --backbone facebook/dinov3-vitb16-pretrain-lvd1689m --image_size 512
```

The image size moves with the backbone because the trunk's patch size has to divide it,
and the grid has to stay a power-of-two multiple of the token grid: 512/16 and 448/14
both give 32x32 tokens and a 128 grid, so nothing downstream of the trunk changes.

A checkpoint carries its own backbone_id and image_size, so anything already trained
keeps loading and running as it was.
