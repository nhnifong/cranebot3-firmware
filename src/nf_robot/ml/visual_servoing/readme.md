Several Dit models have been trained on gripper only cams to perform only a grasp
There is a grasping only dataset at naavox/grasp_only_224 that was used to train naavox/dit-grasp-4 for example. But the Lerobot dit model only uses the cls token from clip. spatial info is too weak.
If I use the my dino patch token fork of lerobot I can train a dit model to use that, but the users have to install my fork

there are two things I want to try to address this issue.
1. A standalone model that contains a frozen facebook/dinov3-vitb16-pretrain-lvd1689m backbone and then predicts visual servoing and finger moves from the patch tokens without any lerobot dependency at all.
2. Dataset augmentation in which pictures of the floor with objects are composed with various transforms and with various fingers positions, along with the gripper velocity appropriate to center them.

The motion that should be predicted is that which would center the object under the gripper. where that is in the frame depends on the height of the gripper, which is simulated by the zoom level that we show the image at. At eval time we have a laser rangefinder input and we can simulate that as well in the synthetic training data.

Some tooling can be created to collect the raw ingredients for this dataset. Something that sweeps the fingers and collects a frame at every position. Something that moves the fingers out of frame and takes pictures of bare floors. and something that takes pictures of objects on plain white boards.

Additionally, since we have a lot of teleoperation data, something that tries to extract such things from teleop frames would be good too.

the standalone model would take one gripper frame at the native resolution of the vision encoder, and produce a velocity in the frame of reference of the gripper

the synthetic dataset could also contain finger and wrist data. either velocity or not.
Objects captured on board should be oriented in the ideal grasping position so we will be able to know the wrist offset from how we rotate it in the synthetic image.

Finger on stringman is commanded by speed which is interpreted by the gripper as either finger move or pressure change depending on contact. It's probably fine to continue predicting this combined finger change scalar from images, but some method of deciding what it should be for each synthetic frame needs to be created and populated using the teleoperation data.

The layers after the patch tokens could be the following.

# Architecture

## What the model predicts

It predicts *where the object is*, not how fast to move. The velocity is chosen
downstream, from the predicted position and the range, the way the existing centering
behaviors already do - observer.py's _center_card_in_view takes a pose in the camera
frame and closes the error with a gain, which is exactly the consumer this head feeds.
Two reasons for the split:

- The synthetic labels for position are exact and free - we know where we pasted the
  object, at what zoom, at what rotation.
- A geometric output is inspectable. When it misbehaves we can look at the heatmap and
  see whether it mislocated the object or misjudged the distance.

It also means the gain and the descent rate stay tunable at deploy time instead of
being baked into weights.

## Input

The gripper feed is natively 684x384. The model takes **448x252** (32x18 = 576 patch
tokens at /14), which is exactly the native 16:9. DINOv2 interpolates its position
embeddings, so a non-square input is fine.

Other inputs include the laser range, the grip pressure set point, and finger angle.
Explicitly *not* vel_x/vel_y/vel_z: the previous action is the most reliable route to
causal confusion there is, and the recorded 14-dim state vector leads with it.

wrist angle is omitted intentionally since it's relative to the room origin.

Measured grip pressure is also left out, while the set point stays in. The set point is
a command we chose, so it carries no answer; the measurement is the answer to a question
we want the network to answer visually (head 5 below), and feeding it in would turn that
head into a pressure repeater instead of an independent estimate we can cross-check
against the sensor.

## Backbone and trunk

Same skeleton as OrthoTargetNet in ortho_target.py, which already works in this repo:

    frozen DINOv2 ViT-B/14, last 4 hidden states, patch tokens only
      -> concat on channels                        (B, 3072, 18, 32)
    Conv2d(3072 -> 256, 1x1), GroupNorm, GELU      (B,  256, 18, 32)
    2-4 x self-attention blocks over 576 tokens    (B,  256, 18, 32)
    [bilinear x2, Conv 3x3 -> 128, GN, GELU]       (B,  128, 36, 64)

Every shape is derived from the trunk's patch size and hidden width, so another backbone
changes only the numbers.

The self-attention blocks are the one real addition over OrthoTargetNet. A pure conv
head has a bounded receptive field, but this task needs global reasoning: "this dark
blob is cut off at the bottom edge, so the object continues past it" is exactly the
inference dit-grasp-4 fails to make. At 576 tokens the attention costs almost nothing.

[CLS] and the register tokens are kept aside as a separate global vector for the heads
that describe the whole image rather than a location.

State conditioning is by FiLM (a small MLP over the state producing per-channel scale
and shift) applied to the 256-channel map. The state fed in is the other non camera sensors mentioned above.

## Heads

**1. Target position, 3D, in the gripper camera frame.**

Decoded from a spatial softmax rather than regressed directly, so that several
candidate objects produce several modes instead of an average that lands on the empty
floor between them - the same argument as the ortho targeting model.

The grid spans **1.5x the frame extent in each axis** (image coordinates -0.25 to 1.25),
so an object just past the bottom edge has a real cell to live in and the softmax can
answer "down there, off-screen" instead of being forced to pick a visible cell. This is
the sock case, and it is the specific thing dit-grasp-4 cannot represent. At 36x64 cells
over 1.5x extent each cell is about 10.5px of frame; a 2-channel sub-cell offset head
(sigmoid, L1 loss, supervised only at the true cell) takes precision below that.

The spatial softmax gives two of the three dimensions. The third - distance along the
ray - is a per-cell regression channel in log metres, gathered at the winning cell along
with the offsets. (u, v, distance) is then a 3D point by the pinhole model, using the
gripper camera's calibrated intrinsic_matrix; cv_common.py already reads that out of the
camera calibration. Per-cell rather than a single global scalar because with two objects
at different heights in frame, one global distance has no correct value.

Loss: cross-entropy over cells, L1 on the offset of the true cell, Huber on log
distance.

**2. Grasp axis, 2 channels, read from the winning cell.**

Predicted per-cell and gathered at the argmax, like the offsets. Parametrised as
(sin 2*theta, cos 2*theta) because a two-finger grasp axis is pi-periodic - regressing
theta directly puts a wraparound discontinuity in the middle of the label space. Labels
come free from how much the object was rotated when composited, given that board
captures are photographed already in their ideal grasping orientation.

**3. Finger speed, scalar in [-1, 1], from the global vector.**

A `--close_heads` checkpoint adds two more global heads and deploys from those instead,
because what a rate label describes is a teleoperator's thumb: on the same situation it
reads +1 one frame and 0 the next, and 69% of the mined values are exactly zero. What a
grasp actually consists of is a decision and a target, so those are what get predicted:

  **3a. Probability the close should have begun by this frame.** A step at the close
  onset, found by walking back from the grasp through the run of frames commanding a
  close - so it is 0 through the approach and 1 from the moment the operator committed.
  47% positive on mined rows, against the rate head's 69% zeros. Bare-floor negatives
  carry 0, which is a fact about them rather than a mask.

  **3b. The grip force the object turned out to need**, read off the finger sensor at the
  moment the lift began, and the same value on every frame of the episode: it is a
  property of the object, not of the frame. Softplus, since a negative force is not one
  the gripper can hold.

Deployment stops being a rate follower and becomes a program: hold the fingers still
until 3a crosses 0.5, then close at one speed with a short ramp until the *commanded*
force reaches 3b. The commanded value rather than the felt one, because that is what the
ask is - the gripper turns finger speed into a force ramp on contact, and whether the
felt force followed is the holding head's question.

Old checkpoints carry no `close_heads` key, build three global outputs, and drive from
the rate head exactly as before.

More grip is positive, less grip is negative; scaled to the robot's finger speed units
downstream, which keeps it compatible with the existing gripper_vel action.

Predicted from [CLS] rather than from any cell, because by the time the decision matters
the object usually fills or blinds the frame. Whether to close is a property of the
whole image, not of a location in it.

Labels for synthetic frames are the open problem. Rather than hand-authoring a rule,
author a *parametrised* one - close when the target is inside the jaw region and the
range is below h_close - then fit its parameters against recorded teleop closes by
maximising agreement. That gives a defensible label rule, and as a by-product a number
saying how predictable human close-timing actually is. If agreement tops out at 70%,
this head's ceiling is 70% and we should not chase it further.

**4. Probability that any graspable target is present at all.**

A single logit off the global vector. Needed because every other head is conditional on
there being something to go to, and because "nothing here, ask the room-level targeter
for a new destination" is a real and useful answer. The other heads' losses are masked
on frames where this is false.

**5. Probability that we are currently holding something.**

Another logit off the global vector. Worth having as a purely visual estimate precisely
so it can be compared against finger_pressure at eval time - the two disagreeing is
informative (pressure without a visual grasp is a snagged carpet or a finger jammed on
the floor; a visual grasp without pressure is a slipping towel).

Labels for this are available after all, at least from teleop, but pressure crossing the
grasp threshold is not one of them. That instant says the jaws closed on something, which
is the cheapest thing in an episode to reach, and a head trained from it learns "an object
is close in frame" - it then reads near 1 through the whole approach, before anything is
in the hand.

The positives are the carry: from the **lift**, where the grip proves itself by taking the
weight, to the **drop**, where the operator opens the jaws or the object slips out of them
(mine_teleop.find_lift and find_drop). Object held, off the floor, still in frame is the
cleanest positive-label data in the set, and everything before the grasp is a real
negative - an object in view, in reach, and not in the hand is exactly the case the head
keeps getting wrong. The frames between the grasp and the lift are masked: they look like
the held ones and the object really is between the jaws, only unproven.

That costs about half the positives (measured over 744 mined grasps: 66k frames to 30k,
with the balance moving from 1:2.8 to 1:6.2 against) and it drops the ~12% of episodes
whose lift lands past the end of the retained carry. `--carry_seconds` is the knob if the
head turns out to be starved; the default 3.0 was chosen when a positive started at the
grasp, and the median lift is 1.37s after it.

Synthetic frames mostly can't be labelled for this, so mask the loss there rather than
guessing. Composites with an object pasted between the jaws could supply positives later
if the head turns out to be data-starved.

## Training notes

**Freezing the backbone is load-bearing for the synthetic data**, not just a speed
choice. Copy-paste composites have tell-tale edge statistics, and a trainable backbone
will find them and key on them. A frozen DINOv3 cannot adapt to the artifact, so the
head is forced to use object-shaped features. Still worth alpha matting the board
captures and feathering the edges, and randomising exposure, white balance, motion blur
and JPEG quality - motion blur especially, since live frames have it and board captures
will not. (Cast shadows: skipped for now.)

**Validation is real teleop only.** Synthetic validation accuracy will be high and
meaningless. Hold out whole teleop episodes and score with distance error and hit-rate
at radius, plus the constant-prediction baseline - ortho_target.py has both
(evaluate_model and constant_baseline). For a centering task, "always predict the
centre" is an embarrassingly strong baseline and we want to know when we actually beat
it.

**Keep a fixture set of hard frames as a regression test.** Twenty frames with
hand-labelled directions, run against every checkpoint. Start with the frame that
prompted this: a dark sock just past the bottom edge, fingers visible, correct answer
"below the frame, not graspable yet". dit-grasp-4 answers that one by closing its
fingers in place, and a fixture set would have caught it before deployment.

**Symmetry augmentation is only partly valid.** Horizontal flip works if the target's u
coordinate and the grasp axis angle are mirrored with it. Vertical flip does not - the fingers occupy specific
edges and lighting is top-biased. 180 degree rotation is valid if the two fingers are
symmetric. Do not reach for the full dihedral group the way ortho_target does; the ortho
map has no canonical orientation, but the gripper frame very much does.

**Inference budget**: ViT-B/16 over 448 tokens is a few milliseconds on the eval GPU,
comfortable inside a 30Hz loop. The design is backbone-agnostic, so dropping to
dinov3-vits16 for a weaker machine is a one-constant change.

# Tooling

Two independent producers write into one dataset: a synthetic pipeline that composites
frames from separately captured ingredients, and a miner that recovers labels from
teleop recordings we already have. They emit the same row format, so training reads one
directory and the mix is a matter of how much of each we generate.

Both write the image-folder-plus-metadata.jsonl layout that ortho_target.py already
uses; write_split and upload_dataset there can be reused as they are. One row per frame:

    {"file_name": ..., "split_source": "synth" | "teleop",
     "target_uv": [u, v] | null,        # in 1.5x canvas coordinates, may be off-frame
     "target_range_m": float | null,    # third dimension of the 3D target
     "grasp_axis_rad": float | null,    # pi-periodic
     "finger": float | null,            # -1..1
     "target_present": 0 | 1,
     "holding": 0 | 1 | null,           # null = unlabelled, loss masked
     "state": {"laser_rangefinder": ..., "finger_angle": ..., "target_force": ...}}

Every label is nullable and null means "mask this head's loss for this row" rather than
"the answer is zero". The two producers can label different subsets of the heads without
either of them having to lie.

## Synthetic frames

### Capturing the ingredients

Three capture routines, written as **motion tasks in observer.py and triggered by debug
commands**, in the same shape as the gripcards and eyelets actions: a coroutine run
through invoke_motion_task that drives the hardware, collects frames off the gripper
client and writes its output to a file for offline processing. They belong there rather
than in a standalone script because they need the motion primitives, the position
estimate and the exclusive-motion-task discipline that already live in the observer.

All three exploit one fact about the mount: **the gripper camera is in the palm and
rotates with the wrist.** Spinning the wrist therefore leaves the fingers exactly where
they are in frame and rotates the entire world behind them. That is the lever the first
tool is built on, and it also means the model's notion of "wrist angle" is a property of
the background, never of the fingers.

**fingerplates.** Over the green backdrop, same as objectplates: for each finger_angle
stop, spin the wrist through a full turn and hold a frame at intervals. Each frame is
chroma keyed and the per-pixel median taken over the turn, which gives a soft edge on
fluff and frayed rubber for free and drops anything that merely rotated past underneath -
the fingers are pinned to the pixel across the set, so they are the only ungreen thing
that is in the same place in every frame.

The output is one **RGBA plate per finger_angle**, and only per finger_angle: the
fingers' apparent position and size in frame are fixed by the mount, independent of both
wrist angle and height above the floor.

**floorplates.** The operator flies the gripper somewhere clean and clear, and only then
triggers the capture; the tool moves nothing but height and wrist. An autonomous room
sweep would come back with a library of beds, furniture and feet, none of which is a
floor plate. Fingers retracted out of frame first.

At each operator-chosen spot, step through a range of heights and a full wrist turn at
each height, one frame per stop, recording the laser range with each. The height stepping
is what calibrates plate scale against range; the wrist turn is free rotational variety,
and means the compositor can pick a plate already at the wrist angle it wants instead of
rotating one and dealing with resampling and empty corners.

Repeat across rooms, times of day and floor types - carpet, hardwood, tile - since these
plates are the entire background distribution the model sees in synthetic training.

**objectplates.** An object on a **green board**, gripper centred over it, stepping
through several heights and a full wrist turn at each. Green for the usual reason a
green screen is green: almost nothing we pick up off a floor is that colour, so a
distance threshold in a chroma space is a complete segmentation, including holes,
concavities and the gaps inside a crumpled towel that a largest-connected-component rule
on a white board would fill in wrongly. Spill suppression on the alpha edge is a
solved recipe.

The setup instruction carries both labels, so the board needs no printed markings:

- **The operator centres the intended grasp lump under the camera.** The grasp point is
  then the principal point of the capture frame by construction. This is the part that
  matters for towels - the right answer is a chunky lump somewhere off to one side, not
  the centroid of a flat expanse, and it is the operator's judgement that puts the lump
  under the lens.
- **The operator orients the wrist to the ideal grasping angle before triggering.** The
  grasp axis is then zero in the first frame, and every later frame's axis label is its
  wrist angle minus the starting one, which `object_matte` measures from the telemetry
  track. The label is an image-plane direction, not a wrist command: zero means the
  object stands upright in frame, which is the orientation the servoing steers to, and
  the annotated bar runs perpendicular to the object's long axis. The wrist turns one way
  and the image the other, which is the one sign in this pipeline that only a real
  capture settles - `synth_frames.AXIS_FROM_WRIST_SIGN`.
- **The range is recorded with every capture**, since the cutout gets rescaled to the
  simulated height at composite time and that needs to know what height it came from.

Because the camera turns with the wrist, the object rotates through the capture while
its grasp point stays at the principal point, so each frame of the turn is a differently
oriented cutout with its axis label already correct. No separate rotation step and no
resampling.

### Capture settings

All three captures should raise resolution and drop framerate, which is the opposite of
what the live stream wants and is worth a dedicated gripper camera mode.

The gripper streams 684x384 out of a 2304:1296 sensor mode today at 60fps (see
gripper_arp_server.py). Hardware h264 holds up to about 1080p, so a capture mode can go
to 1920x1080 within the existing libav path just by rewriting width/height and
GRIPPER_STREAM_FRAMERATE. Going to the full 2304x1296 sensor readout means dropping the
h264 encode entirely and taking whatever mjpeg or raw frames rpicam-vid will produce at
a few fps.

Either way the pi zero 2w in the gripper is the constraint, so the framerate has to come
down far enough that it is not overloaded - single digit fps is fine here. These captures
are a stepping motion holding still at each stop, not a video.

The reason to bother: plates are only ever *downscaled* at composite time, so the
capture resolution sets the ceiling on how much real detail a synthetic 448x256 frame
can contain, and an object cutout shrunk to simulate a high gripper is exactly where
that detail runs out first. Keying and alpha edges also come out cleaner at high
resolution and survive the downsample well.

Keep the **2304:1296 sensor mode** whatever the output resolution, so the field of view
is identical to the live stream's. A different FOV would silently invalidate every
geometric label in the synthetic set.

### Generating frames

**synth_frames.py**, offline, no robot. Per frame:

1. Use a configurable distribution of laser_rangefinder distances for the synthetic height
   defaulting to a uniform distribution between 0.1 and 1.2 meters.
2. Pick a floor plate. Prefer one captured at or above `r` and at whatever wrist angle
   is wanted, since both were swept at capture time; rescale by `r0 / r` only to cover
   the gaps between height steps, and crop the frame out of it. The frame's wrist angle
   is a property of this choice - the fingers do not move with it.

   Never tile. `r0 / r` regularly leaves the plate short of the frame - above the tallest
   capture there is no plate wide enough, and the plate's aspect ratio is a couple of
   percent off the model input's anyway - and filling that gap by repeating the plate
   puts a seam through the middle of the frame, which is not a floor any camera can see
   and is a feature the model would be free to key on. So the scale is floored at what
   covers the frame, plus a small zoom that only ever goes inwards, and the crop moves
   only within the slack that zoom leaves. Magnifying past `r0 / r` makes the texture
   look nearer than the range label says, which is the lesser wrong; synth_frames logs
   how many frames it affects, and the fix for that number is a floorplates run from
   higher up.
3. Pick 0-4 object cutouts. Scale each by its own capture range over `r`, rotate by a
   random angle, and paste at a random position **in the 1.5x canvas, not the frame** -
   so a good fraction land partly or wholly off the visible edge. This is the sock case
   and it has to be common in training, not a rare corner. Carry each object's crosshair
   point and axis line through the identical transform to get its label.
4. Choose the winner: the candidate nearest the jaws in the image plane. Across many
   random arrangements this teaches the softmax to put a mode on every candidate while
   the cross-entropy target names one - the same argument as the ortho targeting model,
   where several objects on the floor are all plausible and the operator picked one.
5. Composite the finger plate for a sampled finger_angle **on top**, so fingers occlude
   objects. Objects that end up behind a finger are realistic training signal, not a bug
   to avoid - that occlusion is a large part of why the live frames are hard.
6. Photometric randomisation: sensor noise, and small (2px) random transformations of
   the final image.

Frames with no object in canvas at all are the negatives for the target-present head,
and should be a deliberate fraction of the output rather than an accident of sampling.

The distance label is the simulated range, ignoring the object's own height above the
floor. That is a real approximation and it biases tall objects; if it shows up in eval,
the fix is to record object height at capture time and subtract it.

`--annotate_dir` should render the label onto each frame the way ortho_target.annotate
does - crosshair at the target, a tick for the grasp axis, the finger value in the
corner - because a sign error in a compositing transform is invisible in a loss curve
and obvious after ten seconds of flipping through annotated frames.

## Mining teleop

Concretely, what we need out of a teleop recording is: **the last second or two before
every successful grasp, at native resolution, with the eventual grasp point projected
into each frame.**

Unpacking that into requirements:

**Source recordings must be native 684x384**, so mine the sources recorded under
camera_mode "all", not the derived 224x224 sets - those are horizontally squashed and
this model is entirely about geometry.

**Only successful grasps.** The whole trick is that the grasp point is where the jaws
ended up, so an episode where the jaws closed on nothing gives a confidently wrong
label. lerobot_trim_to_grasp.py already finds the grasp instant by held pressure and
already rejects episodes with no subsequent rise; that rejection is exactly the success
filter, and its `no_rise` and `no_grasp` counts are the ones to watch.

**Per-frame state from the parquets**: gripper_pos_x/y/z, spin, wrist_angle,
finger_angle, laser_rangefinder, finger_pressure, timestamp. All of these are already
columns; scan_episode_states in ortho_target.py is the pattern for pulling a few
components for every frame without decoding video.

**The camera pose chain**, which already exists and needs nothing new. The gripper
camera's mount pose in the gripper frame is definitions.gripper_camera, camera_goal.py
composes it with the recorded gripper position and 6D rotation in gripper_camera_pose,
and goal_in_camera_frame expresses any room point in that camera's frame - the same
chain the camera_goal action space and the waypoint labelling in
lerobot_label_contact_actions already run on this data. The 684x384 intrinsic and
distortion are config.camera_cal_wide. So the projection is assembled from parts that
are already load-bearing elsewhere rather than from a new calibration.

Then the labels fall out per head:

- **target_uv, target_range_m**: for each frame t in the window, take the delta from
  gripper_pos(t) to P, rotate into the gripper frame using spin/wrist_angle, and project
  through the extrinsic and intrinsic. Frames where P lands outside the 1.5x canvas are
  dropped rather than clipped.
- **grasp_axis_rad**: wrist_angle at the grasp minus wrist_angle at frame t. A delta, so
  it is frame-relative and matches the head living in the gripper frame - which is also
  why wrist_angle is not an input.
- **finger**: the recorded finger_speed at t, divided by 90. Direct, no rule fitting
  needed; the parametrised close rule is only for synthetic frames, and teleop is what
  it gets fitted against.
- **target_present**: 1 throughout the mined window. Do not emit 0 for other frames -
  we do not know that nothing graspable was visible, and the honest label is null.
  Negatives come from synthetic bare-floor frames.
- **holding**: 1 from the lift to the drop, 0 before the grasp instant and after the
  drop, null between the grasp and the lift. See head 5 above for why the grasp instant
  is the wrong boundary.

Note that this labels the target as *where the jaws ended up*, which is not the object's
visual centroid. That is a feature: it is the chunky-lump answer, learned from a human
who picked the spot, and it is not obtainable from any object detector.

A useful by-product: the mined windows relate laser_rangefinder to the apparent scale of
the object in frame, on real data, which is an independent check on the range-to-scale
relation the synthetic compositor assumes.

The same `--annotate_dir` treatment applies and matters more here, since a wrong
extrinsic, a bad spin calibration or a sign flip in the rotation all produce plausible
looking numbers and obviously wrong crosshairs.

# Commands

## 1. Capture the plates

Debug commands on a connected robot, each writing a parquet file and a manifest line to
`plates/`. Park the gripper before triggering: over clear textured floor for fingers,
over clean floor for floorplates, over the board with the object's grasp point under the
camera and the wrist at the ideal grasping angle for objectplates.

    fingerplates
    floorplates
    objectplates

Survey what has been captured, and eyeball the frames:

    python -m nf_robot.ml.visual_servoing.plates --list --dir plates
    python -m nf_robot.ml.visual_servoing.plates --dir plates --every 4 --full_size 6

Contact sheets land in `plates/preview/`.

Collections captured on different machines merge:

    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all \
        --from plates naavox/plates-macbook justink04/raw_plates
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all --list

Runs already in the destination are skipped by run id, so merging the same source twice
is harmless and an interrupted merge can be rerun. `--list` groups by the robot id and
hostname that captured each run.

Publish a collection's raw captures to a hub dataset, adding whatever runs it does not
already have:

    python -m nf_robot.ml.visual_servoing.merge_plates --into plates \
        --upload naavox/plates-macbook

## 2. Extract the pieces

Fingers, as one RGBA plate per aperture:

    python -m nf_robot.ml.visual_servoing.finger_matte --dir plates_all

Every fingerplates run in the collection is matted, not just the newest; `--run_id` mats
a single run.

Each plate logs how much of its frame was green; a low number means the capture missed
the backdrop and nothing downstream of it is worth looking at. `--green_low` /
`--green_high` move the key's ramp if the fingers are being eaten or the backdrop
survives.

Objects, as one RGBA cutout per captured frame, same keyer and same backdrop:

    python -m nf_robot.ml.visual_servoing.object_matte --dir plates_all

Anything more than `--vignette_m` (0.5m by default) across the floor from the grasp point
is dropped, since the board stops filling the frame near the top of a height sweep and its
edge keys as foreground. Lower it if board edge still survives, raise it for an object
wider than half a metre, `--no_vignette` to keep whatever the key kept.

Check both before generating anything from them. Each writes a contact sheet
(`plates/fingers/_mattes.png`, `plates/objects/_objects.png`) and an mp4 over a
checkerboard (`--fps`, 60 by default): `_mattes.mp4` in finger angle order, `_objects.mp4`
with each cutout back where it sat in its capture frame. The mp4 is what shows a matte
flickering between neighbouring plates, or an object wandering when it should hold still
under the wrist turn.

## 3. Mine the teleop dataset into the pool

One run over everything. Every producer writes into the same `all/` pool and the
train/eval cut happens after all of them.

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/grip_o naavox/simple_grasp_spin \
        --output_root datasets/visual_servoing_pool_252 \
        --preview_dir datasets/visual_servoing_pool_252/preview \
        --preview_count 100 \
        --approach_seconds 5

No split of the LeRobot dataset first: pass the whole thing. `simple_grasp_spin` can be
added here even though it cannot be merged into `naavox/combined_targets`, because the
miner takes several sources in one run.

Frames are stored at 448x252, the default, and everything in the pool has to keep it: the
backbone is a /14 DINOv2 and 252 = 14 x 18. Do not pass `--image_size`. Mining into a pool
that already holds another size raises rather than writing rows that will not collate.

Each producer replaces its own shards and leaves the others alone, so this can be re-run
without disturbing the negatives or the synthetic frames - but mine every source of the
pool in one run, since a second run replaces the first rather than adding to it.

Recordings of flying over empty floor are mined with `--negatives`:

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/combined_negatives --negatives \
        --output_root datasets/visual_servoing_pool_252 \
        --preview_dir datasets/visual_servoing_pool_252/negative_preview

Every frame becomes a `target_present = 0` row with no position labels, one frame in five
by default (`--negative_stride`), written as `negative-*.parquet` beside the positives
rather than over them.

Nothing here can check that the flight was really over empty floor. You are responsible
for making sure they were all over empty floor.

## 4. Generate synthetic frames

Joins the same pool as extra shards, drawing simulated heights from the mined rows so the
two cover the same range distribution:

    python -m nf_robot.ml.visual_servoing.synth_frames \
        --plates plates_all \
        --output_root datasets/visual_servoing_pool_252/ \
        --ranges_from datasets/visual_servoing_pool_252/all \
        --count 80000 \
        --annotate_dir datasets/visual_servoing_pool_252/synth_preview

Composites at `mine_teleop.IMAGE_SIZE`, the same constant the mining above defaults to,
so both producers of the pool agree without being told. Rerunning replaces only its own
shards. The annotated frames show the target, the grasp axis and the other candidates,
which is where a compositing sign error shows up.

## 4b. Audit what was built

    python -m nf_robot.ml.visual_servoing.audit --data_root datasets/visual_servoing_pool_252

Run after every rebuild, before dealing the pool: whether a head has anything to learn
from is a property of what was built, not of how it was cut. With no `--split` it audits
every directory present, so it reads the pool before step 5 and both halves after.

Label columns only, no image decode, so a 4.5GB directory reads in seconds. Prints what
each head has to learn from - coverage, class balance, distributions, and a per-producer
breakdown of the axis - and exits non-zero when a head has labels on only one side of its
range. Catches the failures a loss curve cannot: a pool with no synthetic shards and so no
`target_present = 0` anywhere, or cutouts whose grasp axis is a constant zero.

## 5. Deal the pool into train and eval

    python -m nf_robot.ml.visual_servoing.split_pool \
        --data_root datasets/visual_servoing_pool_252

Replaces `train/` and `eval/` wholesale from the pool, one row at a time and at random.
`--eval_fraction` (0.1) and `--seed` (0) are the only knobs; the same seed over the same
pool deals the same split.

Downstream of every producer, which is the point of the ordering. Cut upstream instead and
whatever is generated afterwards lands wholly on one side - which is how an eval split ends
up with no synthetic rows in it, and so no `target_present = 0` anywhere, and so no way to
grade the head that decides whether there is anything to reach for at all.

Re-run it after anything writes into the pool again. It costs one pass over the pool, not
a re-mine, so re-dealing with another seed is cheap - but the deal is over all of it, so
adding to the pool re-deals what was already there, and metrics either side of a rebuild
are measured on two different eval sets.

The line it prints about near-duplicates is worth reading. The cut is over rows, so an
episode's run of frames and a plate's composites land on both sides; eval then measures
how the model does on further frames of scenes it has trained on. That is the right
question for choosing between checkpoints of one run and an optimistic one for predicting
an object the robot has never seen.

## 6. Train

    python -m nf_robot.ml.visual_servoing.train \
        --data_root datasets/visual_servoing_pool_252 \
        --close_heads \
        --epochs 14 \
        --batch_size 400

The whole `train/` split trains - what step 5 dealt. The checkpoint written after every epoch is always the
newest one; `eval/` is scored and reported but selects nothing.

Reading the metrics:

- `axis_kappa` is the median concentration of the grasp axis head. Near zero means it has
  learned to hedge. The axis term is a log likelihood, so it goes *negative* as the head
  grows confident, bounded below at about -2.9 by `KAPPA_MAX`.
- `axis_deg_flat` beside `axis_deg` is what "always upright" scores on the same rows. At
  0.0 the eval rows carry no axis labels and cannot grade this head - which used to be the
  normal state of affairs, when eval was mined separately and held teleop rows only. Now
  that the deal is downstream of every producer it means the pool itself has no synthetic
  rows in it, and `audit` says so before training does.

`--axis_loss mse` and `--no_axis_balance` turn off the von Mises objective and the
per-angle-bin row weighting for an A/B; both are on by default.

## 6c. Publishing the dataset

    hf upload naavox/visual_servoing_dataset_pool_252 datasets/visual_servoing_pool_252 \
        --repo-type dataset --exclude "all/*"

Exclude the pool: it holds every row that train/ and eval/ already hold between them, so
uploading it doubles the transfer and gives the hub a third copy of the data that no
config in the dataset card names. Keep it locally - it is what a re-deal deals from.

Train from it with `--dataset_id naavox/visual_servoing_dataset_pool_252` in place of
`--data_root`.

A new id rather than a new version of an old one, because each of these names records
something that would otherwise mismatch in silence:

- `naavox/visual_servoing_dataset` holds 448x256 frames for the older DINOv3 checkpoints.
  Still `train.py`'s `DEFAULT_DATASET_ID`.
- `naavox/visual_servoing_dataset_252` is 448x252, cut before mining. Its eval split is
  teleop rows only - no synthetic frames, no negatives - so the numbers it produced are
  not comparable with anything measured on a dealt eval split, whatever the model.
- `naavox/visual_servoing_dataset_pool_252` is 448x252, dealt from the pool. Every
  producer reaches both sides.

The frames are the same size in the last two and the layout is not, which is the kind of
difference that survives a download and shows up as a metric that moved for no reason.

## 7. Evaluate

    python -m nf_robot.ml.visual_servoing.evaluate \
        --data_root datasets/visual_servoing_pool_252 \
        --model_path models/visual_servo.pth \
        --preview_dir datasets/visual_servoing_pool_252/eval_predictions

Prints the metrics next to the constant-prediction baseline. Beating that baseline is the
bar: for a centering task "always predict the middle" already scores well.

`--preview_dir` draws the label in green and the prediction in red on the scored frames,
joined by a line.

## 8. Grasp with it

    python -m nf_robot.host.observer --local_models

`execute_grasp` runs `host/visual_servo.py` by default; there is no flag to turn it on.
`--lerobot_grasp` is what hands the grasp to a policy session instead, and that one does
fall back here when no session answers.

`--local_models` reads `models/visual_servo.pth`; without it the checkpoint comes from
`naavox/visual_servo` on the hub, which has to have been published there first:

    hf upload naavox/visual_servo models/visual_servo.pth visual_servo.pth

Until it is, a run without `--local_models` declines the grasp and says so rather than
raising.

The `servograsp` debug command runs one grasp from wherever the gripper is parked, so a
checkpoint can be tried without the pick and place loop choosing targets around it.
`servoloop` repeats that forever - grasp, drop where the lift ended, settle, again -
logging a running tally after every attempt:

    servoloop attempt 7: SUCCESS in 8.2s | servoloop 5/7 (71%) in 2.3 min | success 8.4s avg | failure 21.1s avg

After each drop the wrist goes to a random angle in its three revolutions and the gantry
hops to a random point within 0.4m of where the run began, so each attempt approaches
from a different direction and distance. The draw stays inside the work area and will not
go more than 10cm below the start height.

Nothing re-targets between attempts, so it measures this loop rather than the room-level
targeting, and an object flung out of view ends the useful part of a run. Success and
failure times are reported separately.

`servowatch` is the step before that: it runs the model on every frame and reports it to
the overlay while commanding nothing - no gantry velocity, no wrist, no fingers - until
the motion task is cancelled. Park the gripper over an object, or fly it by hand, and
watch where the arrow points.

The loop is the downstream half the model was shaped for. Each pass:

- one gripper frame plus laser range, finger angle and grip set point in, one target
  position out (`servo.predict_frame`)
- the predicted camera-frame point becomes a room-frame offset from the *lens* through
  `geometry.camera_to_room` - the inverse of the transform mine_teleop labelled with,
  which is why both directions live in one file
- the horizontal part of that offset is the centering error, closed with a gain and a
  speed cap, the same shape as `_center_card_in_view`. It is filtered first, over about
  one pendulum period: what is smoothed is the target's room position rather than the
  offset to it, so the filter removes swing without lagging the loop's own corrections,
  and a jump of more than half a metre re-seeds it rather than sliding across the floor
  between two objects
- descent is gated on being centered, with a tolerance that is a fraction of the range
  and therefore an angular one: 3cm of error at 60cm up is a correction the rest of the
  descent absorbs, and the same 3cm at 8cm up is a miss
- the grasp axis is commanded as a fraction of the predicted angle each pass, which is a
  closed loop because the camera turns with the wrist and the prediction shrinks as it
  comes around - but only when the axis head's concentration clears a threshold. An unsure
  head still decodes to some angle, since atan2 throws the length away, so without the
  gate the wrist turns to a hedge
- that command is an absolute angle chosen out of the wrist's three revolutions, not a
  raw sum. `setWrist` clamps to [0, 1080] silently, so an approach that has walked toward
  a limit would otherwise go quiet, every correction clamping to the angle it already
  holds. The grasp axis is pi-periodic, so the same jaw line is always reachable 180
  degrees away; the choice weighs travel against distance from the neutral 540, which
  keeps a long run off the ends without spending half turns chasing the middle
- the fingers are driven by the finger head, every pass, and by nothing else - the
  prediction is a rate in the units the teleop labels were recorded in, so deploying it is
  a multiply by 90 and no more. That includes the commit at the end: the gantry stops, the
  model keeps the fingers, and a pressure rise is what makes the routine report success.
  A fixed closing speed would be a constant standing in for the one part of the approach
  the head has the most evidence about, since the frames where a teleoperator ever
  commanded a close are frames with the object already between the jaws
- the close threshold no longer starts the close - the fingers are already following the
  head - it decides when to stop the gantry, because driving on once the model is closing
  is how an object gets pushed out of the jaws

After the close it logs the holding head beside the pressure verdict. The two disagreeing
is the informative case (readme head 5), and having it in the log is what makes the
question answerable from a run rather than from a fixture set.

Every prediction also goes out as `GripCamPredictions`, which the UI draws over the
gripper feed: an arrow from the frame centre to the target, a bar for the grasp axis, and
the two probability bars. `move_x`/`move_y` are a displacement from the frame centre, so
an arrow can leave the picture - that is the off-canvas case, not a bug. Watching it
during a descent distinguishes a model mislocating the object from a mistuned loop.

# Open questions

- 1.5x canvas extent is a guess. If misses are usually small, 1.25x is cheaper to learn;
  if the gripper often ends up a body-length away, the canvas cannot cover it and a
  direction-only fallback would be needed for those.
- Whether the holding head has enough teleop positives on its own, given that synthetic
  frames can't supply any.

# Footnote: DINOv3

The backbone was `facebook/dinov3-vitb16-pretrain-lvd1689m` until 2026-08-23. It scores
the same as DINOv2 with registers on this task and is gated - an approved access request
plus an `HF_TOKEN` on every machine that trains or runs the model - so there is no reason
to prefer it.

To train against it anyway:

    python -m nf_robot.ml.visual_servoing.train \
        --data_root datasets/visual_servoing \
        --backbone facebook/dinov3-vitb16-pretrain-lvd1689m \
        --image_size 448 256

It is a /16 model, so it wants 448x256 frames and a dataset mined at that size
(`--image_size 448 256`, and a different `--output_root`). `naavox/visual_servoing_dataset`
on the hub is the 256-tall one.

Do not point it at a 252 dataset, or the reverse. `VisualServoDataset` hands the stored
frame over unresized and 448x256 through a /14 patch embedding floors to the same 32x18
token grid the model computes for 448x252 - so it trains, the loss falls, the bottom 4
pixel rows are never seen, and `target_uv` stays normalized over all 256, putting every
label about 1.6% low against the feature map. It shows up later as an unexplained
downward bias. Mining into a pool that already holds the other size does raise.
