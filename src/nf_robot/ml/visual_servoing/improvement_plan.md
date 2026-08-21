# Visual servoing: what is wrong and what to do about it

Written after a first checkpoint (`models/visual_servo.pth`, epoch 21) was flown in
observe mode. Its behaviour, as observed on a bedroom floor that is *in* the training
set: direction predictions usable but noisy with a +Y lean, wrist axis pinned at zero
even on a sharpie, `holding` true too early, `present` pinned at 1.0 on obviously empty
frames, and no measurable improvement on the held-out room across training. The one
clear success is that it ignores its own fingers.

Items are numbered `phase.index` and referred to that way in conversation. Status is kept
current as work lands.

## Evidence

Measured from `datasets/visual_servoing`, `plates/objects` and the loss code, not
inferred from behaviour.

**E1. The training set contained no synthetic data, and no negatives at all.**
228,659 rows, `sources={'teleop': 228659}`. `target_present` is labelled on 54% of rows
and 100.000% of those are 1 - not one negative example exists. The `present` head has
never been shown an empty frame, so saturating at 1.0 is the only thing it can do. This
is unlearnable, not mistuned.

The likely mechanism is that `mine_teleop.mine()` calls `shutil.rmtree(split_dir)`, so a
re-mine after generating synthetic frames deletes them without saying so.

**E2. Every object cutout carried a grasp axis of exactly zero.**
All 13,301 entries in `plates/objects/objects.jsonl` had `wrist_offset_deg: 0.0`, while
the capture telemetry was fine - the wrist really does sweep, ~260 degrees of span and
about 1,070 distinct values per run. The artifacts were stale, extracted before that
computation worked. Fixed by 0.0; `plates_all/objects` now has 40,208 cutouts with an
offset std of 113 degrees and 95.2% beyond 5 degrees.

While it lasted, every synthetic frame showed an object at a random orientation and told
the model its axis was zero. That is worse than no label: it actively teaches that
orientation is irrelevant.

**E3. Teleop cannot supply a close-range axis label at all.**
The label is `wrist_at_grasp - wrist_at_t`, which goes to zero as the grasp approaches by
construction:

| frames before grasp | share with abs(axis) > 5 deg |
|---|---|
| 0-1 s | **0.5%** |
| 1-2 s | 4.6% |
| 2-4 s | 19.9% |
| 4-7 s | 32.0% |

Only 32% of the 1,064 mined episodes contain any wrist turn over 5 degrees. Overall 85%
of labelled rows sit within 5 degrees of zero. The operator sets the wrist early, so at
exactly the moment orientation is most visible the label reads zero.

**E4. The axis loss makes hedging free.**
`train.servo_loss` uses `F.mse_loss` on the raw 2-channel output and `model.decode` reads
it with `atan2`, which is scale-invariant. The MSE optimum is the conditional mean; given
E3 that mean is `(0, 0.92)`, which decodes to **-0.25 degrees**. Shrinking toward it costs
nothing at decode time. The head is optimal under its loss and useless for its purpose.
This compounds E2 and E3 rather than being an independent cause.

**E5. `holding` flips at an instant.** Frames either side of the grasp are visually
identical, so "true too early" is the label's fault. What the head actually learned -
*the jaws are around something* - is the trigger the servo loop wants.

**E6. Position labels assume a level gripper.** `gripper_rot_0..5` is recorded but
`mine_teleop` never reads it, so every label carries whatever swing was present. That
error is motion-correlated, and is a candidate for the observed +Y lean.

**E7. `finger` labels are 69% exactly zero**, mean +0.19, std 0.45.

## Phase 0 - make the next run trustworthy

- **0.0 Re-run `object_matte`.** DONE. `plates_all/objects` verified: offset std 113 deg,
  95.2% beyond 5 deg, against 0.0 and 0.0% before. The single highest-value action,
  because these cutouts are the only close-range axis labels that can exist.
- **0.1 Rebuild in order (mine, then synth), and stop the miner deleting another
  producer's shards.** Re-mining SKIPPED by decision - the mined half is believed good.
  The synthetic half still has to land in `datasets/visual_servoing/train`. A first
  attempt at 40,000 frames was killed by the OOM killer before it wrote anything; two
  unbounded allocations were behind that and both are now capped, so the run is worth
  repeating:

  - the decoded floor plate pool held every frame of every run, 29GB across the eleven
    runs in `plates_all`. Now a reservoir sample under `--floor_cache_mb`, 2GB by default.
  - simulated heights were drawn straight from the mined rangefinder column, which
    bottoms out at 0.001m - 1.5% of rows are under a centimetre, the sensor in contact
    rather than a view of a floor. At 0.001m the compositor asks for a floor plate
    magnified 283x: one 125GB allocation. Heights are now clamped into 0.05-1.5m, which
    moved 6% of draws, with a hard 6x magnification backstop underneath that never binds
    in normal operation.

  A 4,000 frame run now peaks at 4.3GB. The guard against silent deletion of another
  producer's shards is still worth adding before the next re-mine.
- **0.2 Put `finger` on the gripper overlay.** Needs a `GripCamPredictions` field and a
  bar to draw it. It is already in the throttled `servo watch` log line, but not where a
  person watching a descent is looking.
- **0.3 Dataset audit tool.** DONE - `python -m nf_robot.ml.visual_servoing.audit`.
  Standalone, label columns only, no image decode. Every finding above prints in one run,
  which is the point: E1 and E2 were both invisible in a loss curve and obvious in a
  histogram.

## Phase 1 - model and loss

- **1.0 Axis head trained as a von Mises likelihood.** DONE. The head's 2-vector is read
  as a distribution rather than a point: direction is the answer, length is the
  concentration. Loss is `log I0(k) - v.u`, which is the negative log likelihood written
  so that no atan2 appears in it - the dot product of the raw output with the unit target
  already equals `k cos(2t_pred - 2t_true)`. Under the mean squared error it replaces,
  the optimum was the conditional mean and `decode` read its direction with atan2, which
  is scale-invariant, so shrinking toward the mean cost nothing; now length is trained and
  reported. `decode` and `predict` return `axis_concentration`, `evaluate` prints
  `axis_kappa`, and the whole vector is bounded by `KAPPA_MAX` - clamping only the kappa
  in the log-partition term while leaving the dot product free makes an arbitrarily long
  vector an arbitrarily large reward, which is a runaway rather than a fix. `--axis_loss
  mse` keeps the old objective for the A/B.
- **1.1 Axis loss balanced by angle bin.** DONE. Inverse frequency over 18 bins of 10
  degrees, normalized so the mean row weight is 1 (so the axis term is not silently
  rescaled against the other heads) and capped at 10x (inverse frequency is unstable in
  the tail; five rows would otherwise outweigh the bulk). On the current split this moves
  the share of the axis gradient spent on near-zero rows from **69.1% to 9.4%**.
  `--no_axis_balance` restores the old behaviour.

  Worth being clear about which item does what: the balance is what redistributes the
  gradient away from "always upright". The von Mises change does not by itself stop a
  *constant* predictor decoding to zero - fitted against these labels it still does - what
  it buys is that hedging is now visible and priced, so a head that has learned nothing
  reports a low kappa instead of a confident zero.
- **1.2 Deadband `holding`.** Mask +/- 0.3 s around the grasp instant, where the label is
  a coin flip. Consider splitting into *ready to close* (what it learned, what the servo
  wants) and *carrying* (the clean post-lift frames).
- **1.3 Leave `present` alone.** It needs negatives, not a loss change. See 3.1.
- **1.4 Per-head constant baselines in eval.** Especially "always 0 degrees" for axis and
  "always 1" for present. Neither head is currently scored against the thing it is doing,
  which is why both failures survived training.

## Phase 2 - the servoing loop

- **2.0 EMA on the target.** DONE, filter only; the move/settle cadence is deliberately
  still open. Time constant 1.4 s, one pendulum period. What is filtered is the target's
  *room position*, not the offset to it: an offset measured a second ago was measured from
  somewhere else, so averaging offsets fights the loop's own corrections and reads as lag,
  while averaging the point they name does not. Sampled against real elapsed time, since
  the loop rate is whatever inference took. Re-seeds on a jump over 0.5 m (a different
  object, not swing) and at the start of every attempt.
- **2.0a Wrist command shaping for the 3Hz resonance.** DONE. Observed on the robot: the
  wrist rings for about 1.5s after a hard move. Three things feed it and all three are now
  addressed in `_servo_wrist`. The correction used to be added to the *raw* measured angle,
  which closes the loop around the ringing itself - telemetry reports the top of the swing,
  the command follows it, and a sixth of a second later it chases the bottom - so the base
  is now a 0.4s low pass. The commanded angle is rate limited to 25 deg/s, since
  `set_wrist_angle` writes the servo's position target with no ramp of its own
  (`gripper_arp_server.setWrist`) and a step is an impulse into the resonance. Changes
  under 1.5 degrees are dropped rather than dithering at the loop rate. The constants sit
  together at the top of observer.py and want revisiting once the frequency is measured.
  If shaping proves not to be enough, the gripper also accepts `set_wrist_speed`, which it
  integrates into a ramp at its own loop rate and expires on its own - a velocity command
  cannot deliver a step at all.
- **2.1 Wrist gated on axis confidence.** DONE. `_servo_wrist` refuses below a von Mises
  kappa of 1.5 - roughly a spread wider than +/-25 degrees on the axis - and returns
  whether it acted, so the debug log says `wrist turning` or `wrist held (unsure)`. The
  gate lives inside the helper rather than at its call sites, so no path can turn the
  wrist on a hedge. This is what 1.0's concentration output was for.
- **2.2 Trigger the close from the `holding` head** cross-checked against range, and
  leave the finger head out of the trigger until it earns a place.
- **2.3 Expect `PRESENT_THRESHOLD` to come alive.** It is dead code today - `present` is
  a constant 1.0, so the "nothing seen" abort can never fire. It will start firing as
  soon as negatives exist; that is a behaviour change to anticipate, not to debug.
- **2.4 Do not hand-tune the +Y lean** until 3.4 says whether it is the model or the
  labels.
- **2.5 Wrist commands are an absolute setpoint.** DONE. The wrist wobble on a robot was
  not noise: the correction was an increment on *fresh* wrist telemetry derived from a
  frame a quarter of a second old, so every pass inside the video's blind window
  re-commanded a correction already in flight - at 10Hz and 0.25s that is two or three
  passes each adding half the same error, arriving ~1.5x past the mark, reversing, and
  repeating. `_servo_wrist` now anchors to the wrist angle *at frame capture time*, which
  names one fixed goal that every frame in the blind window agrees on, so re-sending it is
  idempotent. Frame capture times and grip sensor records are both stamped by the
  gripper's own clock, so the lookup crosses no clock boundary and needs no latency
  constant; a telemetry gap wider than `WRIST_ANCHOR_MAX_AGE_S` skips the correction
  rather than anchoring to the wrong angle.
- **2.6 Per-camera video latency.** DONE. `ComponentClient.video_latency()` is the median
  over that camera's last ~120 frames; the shared StatCounter still feeds the UI its mean
  across all cameras. Note the absolute value carries host-to-bot clock skew, since
  capture is stamped by the component and arrival by us.
- **2.7 The lateral path has a milder version of 2.5.** OPEN. `centering_error` combines a
  camera vector from a 0.25s-old frame with `pe.grip_pose` sampled now, which places the
  target about 3.7cm ahead in the direction of travel at 0.15 m/s. Unlike the wrist this
  shows as a lag-driven bias rather than a limit cycle, and the target EMA hides most of
  it. The fix is the same idea - the gripper pose at capture time - but it needs pose
  history, which the position estimator does not keep.

## Phase 3 - data, ordered by measured gap

- **3.0 objectplates on elongated objects** - sharpie, sock, spoon, towel. The only
  close-range axis signal that can exist (E2, E3).
- **3.1 Negatives.** Synthetic zero-object frames at a deliberate fraction, with fingers
  composited so they match what the camera sees at deploy time, plus floorplates above
  0.61 m - `synth_frames` already warns that the tallest plate caps the honest range.
- **3.2 Teleop with late wrist turns.** A few sessions where the wrist is deliberately
  oriented in the final two seconds converts teleop from useless to useful for this head.
- **3.3 More rooms and floor types**, since the held-out room never improved and the
  training half is a handful of rooms.
- **3.4 Read `gripper_rot_*` in `mine_teleop`** to de-bias existing labels. No new
  recordings needed and it re-labels all 228k rows.

## Scoring a checkpoint

`servoloop` is the debug command that answers "is this one better than the last one":
grasp, drop, settle, repeat, with a running hit rate logged after every attempt. A run
needs a floor's worth of the same objects and no re-targeting, so it measures the servo
loop rather than the targeting - which is the thing being changed in Phases 1 and 2.

## Diagnostics to run either side of the above

- Mean **signed** uv residual on a training room against the eval room, which separates
  model bias from label bias and tests the +Y claim directly.
- Residual binned by gantry speed and gripper tilt. If it grows with motion, E6 is the
  cause and 3.4 is the fix.

## The through-line

Items 0.0, 0.1, 3.0 and 3.2 are all the same problem: **the axis head has never seen a
correct non-zero label from any source.** The architecture and the loss are not worth
judging until it has. 1.2 is worth doing alongside, but on its own it would have changed
nothing.
