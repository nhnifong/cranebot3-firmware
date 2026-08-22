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
  object, at what zoom, at what rotation. "The velocity a teleoperator would have
  commanded" is an invented label with an arbitrary gain baked into it.
- A geometric output is inspectable. When it misbehaves we can look at the heatmap and
  see whether it mislocated the object or misjudged the distance.

It also means the gain and the descent rate stay tunable at deploy time instead of
being baked into weights.

## Input

The gripper feed is natively 684x384. The model takes **448x256** (28x16 = 448 patch
tokens at /16), which is within 1.5% of the native aspect ratio. DINOv3 interpolates its
position embeddings, so a non-square input is fine.

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

    frozen DINOv3 ViT-B/16, last 4 hidden states, patch tokens only
      -> concat on channels                        (B, 3072, 16, 28)
    Conv2d(3072 -> 256, 1x1), GroupNorm, GELU      (B,  256, 16, 28)
    2-4 x self-attention blocks over 448 tokens    (B,  256, 16, 28)
    [bilinear x2, Conv 3x3 -> 128, GN, GELU]       (B,  128, 32, 56)

The self-attention blocks are the one real addition over OrthoTargetNet. A pure conv
head has a bounded receptive field, but this task needs global reasoning: "this dark
blob is cut off at the bottom edge, so the object continues past it" is exactly the
inference dit-grasp-4 fails to make. At 448 tokens the attention costs almost nothing.

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
the sock case, and it is the specific thing dit-grasp-4 cannot represent. At 32x56 cells
over 1.5x extent each cell is about 12px of frame; a 2-channel sub-cell offset head
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

Labels for this are available after all, at least from teleop, and the trim-to-grasp
machinery already computes them: finger_pressure at or above the grasp threshold, held
long enough to not be a bump, is exactly "holding something". Every frame from the grasp
onward is a positive and every frame before it in the same episode is a negative. Note
that the post_lift_seconds extension added to lerobot_trim_to_grasp.py keeps a second of
carry after the lift, which is the cleanest positive-label data in the whole set: object
held, off the floor, still in frame.

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
- **holding**: 0 before the grasp instant, 1 from the grasp through the end of the
  retained carry.

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

Collections captured on different machines merge by copying run files and concatenating
manifests, because a run is already a self-contained set of files named after a run id
carrying its kind, its moment and six random hex digits:

    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all \
        --from plates /mnt/contractor/plates
    python -m nf_robot.ml.visual_servoing.merge_plates --into plates_all --list

Nothing is combined, resampled or rewritten - every run stays exactly as its camera
produced it, which is what lets the matte and compositing steps be redone later without
going back to the robots. Runs already in the destination are skipped by run id, so
merging the same source twice is harmless and an interrupted merge can be rerun. Each
run records the robot id and hostname that captured it, and `--list` groups by those.

## 2. Extract the pieces

Fingers, as one RGBA plate per aperture. A chroma key on every frame of the wrist turn,
then the per-pixel median across it:

    python -m nf_robot.ml.visual_servoing.finger_matte --dir plates_all

Every fingerplates run in the collection is matted, not just the newest, and the plates
are named after their run - a collection gathers captures of more than one set of fingers,
and they are not all the same colour. `synth_frames` picks a set per frame and then an
aperture within it, uniformly over sets, so one capture that swept more angles than
another cannot crowd out the colour of its fingers. `--run_id` mats a single run.

Each plate logs how much of its frame was green; a low number there means the capture
missed the backdrop and nothing downstream of it is worth looking at. `--green_low` /
`--green_high` move the key's ramp if the fingers are being eaten or the backdrop
survives.

Objects, as one RGBA cutout per captured frame. Same keyer, so it wants the same
backdrop; without one, raise `--green_low`/`--green_high` and expect to check the
result rather than trust it:

    python -m nf_robot.ml.visual_servoing.object_matte --dir plates_all

Anything more than `--vignette_m` (0.5m by default) across the floor from the grasp point
is dropped, since the board stops filling the frame near the top of a height sweep and its
edge keys as foreground. Lower it if board edge still survives, raise it for an object
wider than half a metre, `--no_vignette` to keep whatever the key kept.

Both write a contact sheet beside their output - `plates/fingers/_mattes.png` and
`plates/objects/_objects.png` - which is the check worth doing before generating
anything from them.

Both also write an mp4 of every result frame over the same checkerboard, at 60fps
(`--fps` to change it): `_mattes.mp4` in finger angle order, `_objects.mp4` with each
cutout back where it sat in its capture frame. Worth a look for what a sheet of stills
cannot show - a matte that flickers between neighbouring plates, or an object that
wanders when it should be holding still under the wrist turn.

## 3. Mine the teleop half

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/combined_targets naavox/simple_grasp_spin \
        --output_root datasets/visual_servoing \
        --preview_dir datasets/visual_servoing/preview \
        --preview_count 100 \
        --approach_seconds 5

This replaces the split it writes, so run it before generating synthetic frames.

## 4. Generate synthetic frames

Joins the same split as extra shards, drawing simulated heights from the mined rows so
the two halves cover the same range distribution:

    python -m nf_robot.ml.visual_servoing.synth_frames \
        --plates plates_all \
        --output_root datasets/visual_servoing/ \
        --ranges_from datasets/visual_servoing/train \
        --count 40000 \
        --annotate_dir datasets/visual_servoing/synth_preview

Rerunning replaces only its own shards. The annotated frames show the target, the grasp
axis and the other candidates, which is where a compositing sign error shows up.

Memory is bounded rather than a function of how much has been captured. The decoded floor
plate pool is the high water mark, so it is capped by `--floor_cache_mb` (2GB by default,
a reservoir sample spread evenly through each run's sweep) - the full set of floorplates
runs decodes to about 29GB and will end the run on most machines. Simulated heights are
also clamped into 0.05-1.5m: a mined rangefinder column bottoms out at 0.001m when the
fingers are closing on something, and composited literally that asks for a floor plate
magnified 283x, which is a single 125GB allocation. Both report what they did, and the
run prints its peak resident memory at the end.

## 4b. Audit what was built

    python -m nf_robot.ml.visual_servoing.audit --data_root datasets/visual_servoing

Label columns only, no image decode, so a 4.5GB split reads in seconds. It prints what
each head actually has to learn from - coverage, class balance, distributions, and a
per-producer breakdown of the axis - and exits non-zero when a head has labels on only
one side of its range.

Worth running after every rebuild, because the two failures that cost the most so far
were both invisible in a loss curve: a split with no synthetic shards and therefore no
`target_present = 0` anywhere, and object cutouts whose grasp axis was a constant zero.
In both cases the network was fitting its labels correctly.

## 5. Train

    python -m nf_robot.ml.visual_servoing.train \
        --data_root datasets/visual_servoing \
        --epochs 40 \
        --batch_size 400

The whole `train/` split trains. `eval/` is scored each epoch when it exists and the
best checkpoint by `recall@25px` is kept; without it, the latest is saved instead.

The grasp axis head is trained as a von Mises likelihood, so its output's length is a
concentration - how sure it is - and not just a leftover of averaging. `axis_kappa` in the
metrics is the median of that, and a head sitting near zero has learned to hedge. Its rows
are also weighted by angle bin, because the labels lean hard on "already upright" and
unweighted they spend 69% of the head's gradient on the answer it already knows. Both are
on by default; `--axis_loss mse` and `--no_axis_balance` restore the previous behaviour
for an A/B. Note the axis term is a log likelihood, so it is expected to go *negative* as
the head grows confident - it is bounded below at about -2.9 by `KAPPA_MAX`.

`axis_deg_flat` beside `axis_deg` is what "always upright" scores on the same rows. On a
teleop-only eval split it is 0.0, which is the honest statement that the split cannot
grade this head at all.

Build the eval split by mining the held-out room's own recipe into the same root:

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/combined_targets_eval \
        --output_root datasets/visual_servoing \
        --split eval \
        --approach_seconds 5


## 6. Evaluate

    python -m nf_robot.ml.visual_servoing.evaluate \
        --data_root datasets/visual_servoing \
        --model_path models/visual_servo.pth \
        --preview_dir datasets/visual_servoing/eval_predictions

Prints the metrics next to the constant-prediction baseline, which is the number that
says whether the model has learned anything about the image at all: for a centering task
"always predict the middle" scores well, and beating it is the bar.

`--preview_dir` draws the label in green and the prediction in red on the frames that
were scored, joined by a line. Numbers say whether it is right; the previews say whether
it is right for the right reason, which is the check worth doing before a model reaches
a robot.

## 7. Grasp with it

    python -m nf_robot.host.observer --visual_servo --local_models

`--visual_servo` replaces the grasping routine pick and place calls: `execute_grasp`
runs `host/visual_servo.py` instead of asking a lerobot session or falling back to the
centering model. That module holds the whole deployed loop - the three modes, the tuning
constants, the wrist limit arithmetic and the scoring run - and reaches into the observer
only for the things that are the robot rather than the routine: the gripper client, the
datastore, the position estimate and the motion primitives. It is an override rather than another fallback, because the
reason to run it is to find out how it does and a silent fall back to something else
would hide that. `--local_models` reads `models/visual_servo.pth`; without it the
checkpoint comes from `naavox/visual-servo` on the hub, which has to have been published
there first:

    hf upload naavox/visual-servo models/visual_servo.pth visual_servo.pth

Until it is, a run without `--local_models` declines the grasp and says so rather than
raising: a checkpoint that will not load is a routine the robot cannot offer, not a
reason to end whatever motion task happened to ask for it.

The `servograsp` debug command runs one grasp from wherever the gripper is parked, which
is the way to try a checkpoint without the pick and place loop choosing targets around
it. `servoloop` repeats that forever - grasp, drop where the lift ended, settle, again -
logging a running tally after every attempt:

    servoloop attempt 7: SUCCESS in 8.2s | servoloop 5/7 (71%) in 2.3 min | success 8.4s avg | failure 21.1s avg

After each drop the wrist is sent to a random angle anywhere in its three revolutions -
which exercises the axis head on a new orientation and the limit turnaround below - and
the gantry hops to a random point within 0.4m of wherever the run began,
so the object is approached from a different direction and distance every time rather than
the same geometry being measured over and over and reported as a hit rate. The draw is
uniform over the ball, stays inside the work area, and will not go more than 10cm below
the start height - the parking height is the operator's judgement about clearance, and
40cm below it finds the floor, the furniture, or the object just dropped.

Nothing re-targets between attempts, so it measures this loop rather than the room-level
targeting, and an object flung out of view ends the useful part of a run. Success and
failure times are kept apart because a failure spends its attempts on timeouts and one
mean over both hides each of them. `servowatch` is the step before that: `visual_servo_grasp(observe_only=True)` runs
the model on every frame and reports it to the overlay while commanding nothing at all -
no gantry velocity, no wrist, no fingers - until the motion task is cancelled. Park the
gripper over an object, or fly it by hand, and watch where the arrow points. A checkpoint
that cannot be trusted with the gantry shows itself here for free.

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
gripper feed: an arrow from the centre of the frame to the target, a bar for the grasp
axis, and the two probability bars. `move_x`/`move_y` are the target's position as a
displacement from the frame centre, so they can fall outside the frame - an arrow
leaving the picture is the off-canvas case being reported honestly rather than clipped.
Watching that overlay during a descent is the fastest way to tell a model that is
mislocating the object from a loop that is mistuned.

# Open questions

- 1.5x canvas extent is a guess. If misses are usually small, 1.25x is cheaper to learn;
  if the gripper often ends up a body-length away, the canvas cannot cover it and a
  direction-only fallback would be needed for those.
- Whether the holding head has enough teleop positives on its own, given that synthetic
  frames can't supply any.
