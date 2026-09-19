# Checking a visual servoing dataset

Nothing in this dataset is observed. Every label is computed - the position labels by
projecting one room point back through an approach, the timing labels by reading pressure
and height against thresholds, the synthetic ones by remembering where a cutout was
pasted. A computed label fails quietly. The arithmetic still produces a number, the
trainer still fits it, and the loss curve of a model learning a wrong label is
indistinguishable from the loss curve of a model learning a right one. Everything in this
file exists because of that.

Three checks, cheapest first. They answer different questions and none of them replaces
another:

- **`audit`** reads every label in the pool and not one image: what exists, how it is
  distributed, whether a constant already beats it. Seconds over gigabytes. Answers *do
  these heads have anything to learn from*.
- **`mine_teleop --preview_only`** labels every frame the way a real run would and draws a
  random sample of them. Answers *does a label land on the object*.
- **`label_video`** renders a whole episode as video with the mark and its recent track.
  Answers *why it missed*, which stills cannot show.

Run them in that order. A pool that fails `audit` is not worth previewing, and a preview
that looks fine is not evidence that the projection is right - only that it was right on
the frames drawn.

## audit - what the labels contain

Documented as step 4b of the readme. Run it after every rebuild and before dealing the
pool. It exits non-zero when a head has labels on only one side of its range, which is the
class of failure that costs a whole training run.

## Preview without mining (`--preview_only`)

The loop to iterate on labelling in. Every frame is labelled exactly as a real run would
label it, a random `--preview_count` of the rows survive, and only those frames are
decoded:

    python -m nf_robot.ml.visual_servoing.mine_teleop \
        --repo_id naavox/nick-sep14 --root datasets/nick-sep14 --preview_only \
        --preview_dir mine_previews --preview_count 60 --limit 20

No parquet is written and no shard is replaced, so `--output_root` is not needed and a
pool being previewed this way is never touched. Decoding frames is nearly all of a run and
a preview looks at a few hundred, so this costs minutes where mining costs hours;
`--limit` cuts it further, at the price of seeing only the first episodes of each source.

The run still reports how many rows it would have written, which is the number the sample
is out of. Works with `--negatives` and `--false_grabs` too.

`--preview_seed` fixes the draw: the same seed over unchanged sources picks the same
frames, so between two runs the labels are the only thing that moved.

## label_video - one episode as motion

    python -m nf_robot.ml.visual_servoing.label_video \
        --root datasets/chuck-aug28 \
        --output_dir datasets/labelling_test/label_video

One mp4 per episode, named for it, at the recording's native frame size rather than the
mined 448x252 - the question here is where the mark goes, and more pixels make that easier
to see.

Everything it draws comes from `mine_teleop`: the same grasp detection, the same
`grasp_point_room`, the same projection, the same calibration. If the video is wrong about
where the label goes then the labels are wrong in the same way, which is the only property
that makes it worth watching.

Two things it shows that the stills do not:

- **The whole episode**, not the mined window. The window is marked in the caption and the
  mark turns from green to grey outside it, but every frame is rendered, so the approach
  before training's window and the carry after it are both visible.
- **The mark's own history**, as a fading trail of the last `TRAIL_FRAMES` (60, two
  seconds at 30fps). A slow drift reads as a curve across the frame and as nothing at all
  in a dot that moved.

A target off the frame edge is drawn as a triangle on the border pointing the way it lies,
rather than dropped - where a label went off the edge is the useful part. A point at or
behind the lens captions "behind the lens" and draws nothing. Each frame also carries
a signed `t` in seconds relative to the grasp, uv and range, gripper xyz and spin, laser and pressure.

Episodes with no detectable grasp, or no rise after it, are skipped with a line saying so
- the same two tests the miner applies, so this renders the episodes that are actually
mined and nothing else.

`--episodes 3 7 11` renders specific ones and `--limit` caps the count; `--approach_seconds`
and `--carry_seconds` only move the marked window, since the whole episode is rendered
either way. `--vcodec`/`--crf` for the encode.

### Diagnosing

The mark is where the miner believes the object is.
Ideally the mark is pinned to the object all the way down.

However, under the current methods, the mark usually deviates signifigantly and shows the following patterns.

1. A constant offset, usually vertical in frame
2. A lot of waving around. probably due to unaccounted for swing or poorly modelled swing.
3. Video Latency
4. when the gripper touches the ground, huge positional errors are introduces from some lines going slack and causing flipping between various hang points feeding the position estimator. Fixed very recently and the fix is observable in the later episodes of naavox/nick-sep14

## Choosing how uv is decided (`--uv_method`)

Three of those four patterns are properties of one input - the gripper position
`Positioner2` reported at record time - and not of the labelling. So the labelling now has
a choice of what to trust, and `mine_teleop` and `label_video` take the same flag for it
and share one implementation in `uv_methods.py`. Rendering one episode under each is the
way to tell the patterns apart.

    --uv_method room-delta            (default, what every existing dataset was mined with)
    --uv_method dead-reckon
    --uv_method dead-reckon-observed
    --uv_method optical-flow

**`room-delta`** puts the grasp point in the room - straight down from the rangefinder at
the instant of the grasp - and differences it against the reported gripper position in
every frame. Every error in that position estimate lands in the label and nowhere else,
which is patterns 2, 3 and 4 above.

**`dead-reckon`** never asks where the gripper is. The object does not move before the
grasp, so the vector to it changes by exactly how far the gripper went, and that is
integrated outwards from the grasp frame out of the commanded velocity (`action.vel_*`,
which is in the gripper's own frame and so is turned by `-spin` each step). The anchor is
the jaws: at the grasp the object is between the fingers. Its failure is the opposite one -
it is a picture of what was commanded rather than of what happened, so it is at its best
near the grasp and drifts the further back it integrates.

**`dead-reckon-observed`** is the same integration off `observation.state.vel_*`, which is
the gantry's own reported velocity. Measured, not assumed: that one is already in room axes
and integrating it through `-spin` scores worse, while the commanded one scores worse raw
than turned. It is the spools' account of what the gripper did, so where it agrees with
`dead-reckon` the approach was flown as asked and where they part the difference is lag,
swing or slack. Neither consults the hang-point estimate, which is the point of both.

**`optical-flow`** reads no telemetry at all except the rangefinder. It anchors at the jaws
like the dead-reckoning methods and then tracks the target through the pixels, one frame at
a time, with pyramidal Lucas-Kanade over a patch rather than a point - a single pixel of
carpet has nothing to lock onto - keeping only the points that survive tracking there and
back, and taking their median so that floor sliding past behind the object does not drag
the answer. Swing, video latency and a hang point flipping cannot enter it, which makes it
the independent opinion the other three are worth checking against rather than a better
version of them.

It is paid for twice. It decodes the video, so it costs what a full mining run costs and
gives back most of what `--preview_only` saves. And it can only follow a target that stays
visible: when the track is lost the method stops rather than guessing, so it produces **no
off-canvas labels at all** and cannot teach the sock-past-the-bottom-edge case the 1.25x
canvas exists for. Its range comes from the floor plane - the rangefinder reading divided
by the cosine of the angle off straight down - so it assumes a level floor and an object
lying on it, which is the same assumption `grasp_point_room` makes.

How far back it gets varies enormously with the flight: on `nick-sep14` ep0 it held the
target for the whole 675-frame episode, ep2 for 6.5 seconds, and ep11 for half a second
before the motion outran it. Across 24 episodes it still had a track 62% of the time half a
second before the grasp and 50% of the time two seconds before.

`--jaw_uv` moves the anchor all three non-default methods hang on. Videos are named for the
method unless it is the default, so two renders of one episode sit side by side rather than
one overwriting the other, and every frame is captioned with the method that drew it.

### Pattern 1 was the camera geometry, and `naavox/red-dot` measured it

The constant vertical offset was two errors, and each hid the other.

`geometry.CAMERA_ROT_BODY` took its tilt from `definitions.gripper_camera`, which says 9.06
degrees. And every part of the system referenced the target to the **lens**:
`grasp_point_room` hung it straight below the lens, and the servo loop nulled the offset
from the lens. Nothing said where the jaws are.

`naavox/red-dot` settles both. One episode: a 15mm red chapstick lid left sitting exactly
between the fingertips, gripper climbing straight up from 0.11m to 1.0m with the dot
centred below it. The dot therefore marks the jaw axis at every range, and how its place in
the frame changes with range separates the two unknowns - the tilt sets where the answers
converge as the gripper climbs, and the lens-to-jaw offset sets how fast they get there.
Neither is identifiable from one range, which is why this went unnoticed so long.

Fitting 666 detected frames:

    camera tilt            -3.389 deg (back, away from the nose), 1.8px residual
    lens -> jaw offset     -27.0 mm along the nose axis
    dot at u               0.4957, so the jaw axis is under the lens in x

The offset is the striking one. The fit was told nothing about the mount and recovered
-27.0mm; `CAMERA_POS_BODY[1]` is +27.0mm. **The jaws are at the gripper body origin and the
lens is 2.7cm in front of them.** The CAD translation was right all along and simply was
not being used - `geometry.JAW_POS_BODY` is now that fact.

The tilt is wrong in CAD by nearly a factor of three, and its sign was right. Both earlier
readings of this file were wrong: the original -9.06 leaned the right way and far too far,
and the flip to +9.06 leaned the wrong way while landing within 2px of the truth at
*grasping range*. That is why the flip looked like a clear improvement on mined frames,
which cluster near the grasp, and fell apart on a descent - at half a metre it is a quarter
of a frame out.

So where the jaws appear is not a constant. It runs:

    laser  0.11m   0.16m   0.24m   0.39m   0.62m   0.88m
    v       0.72    0.63    0.57    0.51    0.48    0.47

`uv_methods.jaw_uv(laser, calibration)` is that curve, and it replaces a `JAW_UV` constant
of (0.5, 0.692) which was never measured - it was the mirror of the pre-flip (0.5, 0.308).

**What this cost, and what it fixes.** Referencing the target to the lens made every label
name a point 2.7cm toward the nose of the thing that was actually picked up, and made the
servo loop null the offset to *that* - so the loop drove until the object sat 2.7cm past
the fingers, at every range. With the target referenced to the jaws, an object sitting
exactly in the jaws now decodes to a 0.00mm error at 0.12m, 0.30m and 0.60m, where the old
chain gave a flat 27mm.

The older pipeline hid this because its labels were tied to the actual grasp through the
training data: the label at the grasp frame was whatever the geometry produced, and the
model learned "report this when the scene looks like a grasp", so the loop's null point was
the real grasp configuration however wrong the constants. Anchoring optical flow on an
asserted `JAW_UV` broke that tie and exposed the 2.7cm.

**Everything mined before this is wrong and so is every checkpoint trained on it.** Re-mine
the pool, retrain, fly it, and only then move the pin in `common/model_revisions.json`.
Nothing checks this at load time.

### What the comparison shows

Eight episodes of `nick-sep14`, four-second window. Frames whose target leaves the 1.25x
canvas are dropped, and "blind" ones are kept with their position labels masked:

    room-delta            2264 mined,  695 dropped, 18 blind
    dead-reckon           2027 mined,  932 dropped, 41 blind
    dead-reckon-observed  2123 mined,  836 dropped, 25 blind
    optical-flow          1833 mined, 1126 dropped,  0 blind

Optical flow's zero is structural rather than good news: it has no answer for a target it
cannot see, so those frames are dropped instead of being labelled off-canvas.

The sharper measurement is how far each one sits from the answer the pixels give, in frame
widths, over 24 episodes. Optical flow is the reference here because it is the only method
that looks at the picture - not because it is right. It can snap onto the wrong thing after
losing a track, and this compares only the frames where it still had one, which selects for
the better-behaved episodes:

    first 24 episodes        -0.00s  -0.25s  -0.50s  -1.00s  -2.00s
    room-delta                0.000   0.207   0.479   0.331   0.315
    dead-reckon               0.000   0.012   0.028   0.106   0.227
    dead-reckon-observed      0.000   0.046   0.096   0.184   0.229

    last 24 episodes         -0.00s  -0.25s  -0.50s  -1.00s  -2.00s
    room-delta                0.000   0.053   0.135   0.202   0.210
    dead-reckon               0.000   0.040   0.081   0.178   0.148
    dead-reckon-observed      0.000   0.152   0.178   0.272   0.728

Four things fall out of that.

**The anchors agree exactly.** Every method reads 0.000 at the grasp, which is the check
that the tilt fix landed: `room-delta` arrives there down the body axis from the mount and
the others arrive there from `JAW_UV` measured off the frames, and they now meet.

**`room-delta` is worst just before the grasp, not furthest from it.** 0.207 at a quarter
second and 0.479 at half a second, falling back to 0.315 at two seconds. Nothing about
integration drift has that shape; touching down does. This is pattern 4 above, measured: the
lines go slack, the hang point flips, and the position estimate moves half a frame during
the part of the approach the labels matter most.

**The hang-point fix is real and the numbers show where it stopped.** That spike is 0.207
and 0.479 in the first 24 episodes and 0.053 and 0.135 in the last 24 - the same recording,
three and a half times better, exactly as the note under pattern 4 predicted. It is the one
claim in this file that came with its own controlled experiment already in the dataset.

**`dead-reckon` off the commanded velocity is the steadiest overall**, and best of all
near the grasp (0.012 at a quarter second, which is three pixels). It pays for that further
back, where it is integrating a command rather than a measurement. `dead-reckon-observed`
is not a reliable improvement on it and is markedly worse in the later episodes; why is not
established.

None of this says which method should mine the next pool. What it says is that the position
estimate is worth about half a frame of error at touchdown in the older episodes and much
less in the newer ones, and that anything mined from the older half carries that.
