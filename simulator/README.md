# Stringman simulator

Everything needed to run Stringman without hardware: the MuJoCo model, the bridge that
puts it under the real component firmware, and the component simulator itself.

    stringman_arp_carbon270.xml   the model
    meshes/                       geometry extracted from the playroom GLBs
    mujoco_bridge.py              MuJoCo-backed stand-ins for the anchor/gripper hardware
    robot_simulator.py            runs the real servers against it
    requirements.txt              mujoco, numpy, zeroconf (plus ffmpeg on PATH)


`stringman_arp_carbon270.xml` — Stringman in the "Arpeggio" 2-anchor configuration with
the Arpeggio gripper on the 270 mm carbon pole.

## Viewing it

MuJoCo 3.12.0 is unpacked at `~/Downloads/mujoco-3.12.0-linux-x86_64`. Absolute paths for
both, so it works from any directory:

```sh
~/Downloads/mujoco-3.12.0-linux-x86_64/mujoco-3.12.0/bin/simulate \
    ~/cranebot3-firmware/simulator/stringman_arp_carbon270.xml
```

`ParseXML: Error opening file` means it could not find the model at all, not that the XML
is malformed — usually a relative path run from somewhere other than the repo root.

The model loads paused at keyframe 0. In the viewer:

- **space** — start/pause the simulation
- **Control** panel on the left — six sliders. The four spool sliders
  (`spool_0_direct`, `spool_1_indirect`, `spool_2_direct`, `spool_3_indirect`) are each
  that line's **free span in metres**: the straight-line distance from the last eyelet
  it leaves to the gantry, 0 to 7 m. The two indirect lines also carry a fixed ~6 m
  cross-room run to their far eyelet post, but you never control that, so it is added
  inside the model and kept off the slider — all four sliders mean the same thing and
  read the same at a symmetric pose. Then `wrist` in radians of spin and `finger` in
  radians (-0.87 open, +0.17 closed).

  **Load keyframe 0 first.** `simulate` resets ctrl to 0 and does not apply keyframes on
  its own, so an untouched model has all four lines commanded to zero span; they
  saturate against their force limit and the gantry sits stalled near the top of the
  room. In the left panel's **Simulation** section, click **Load key**. (The **Key**
  slider above it reads 0 already — the model has only one keyframe, so its range is
  0 to 0 and there is nothing to set.) That puts the gantry at (0, 0, 1.2) with all four
  sliders at their matching spans, ~4.22 and ~4.29 m.

  **Reset**, and the backspace key, go back to `qpos0` with ctrl at 0 — *not* to the
  keyframe — so click **Load key** again after any reset. If the left panel is hidden,
  **Tab** toggles it.

  From there, all four spans together set height, and their differences set position:

  | spans (line 0, 1, 2, 3) | gantry |
  |---|---|
  | 4.22, 4.29, 4.22, 4.29 | (0, 0, 1.2) — the keyframe |
  | 3.86, 3.93, 3.86, 3.93 | (0, 0, 1.5) |
  | 4.45, 4.51, 4.45, 4.51 | (0, 0, 0.45) — fingertip on the floor |
  | 1.66, 5.19, 5.12, 7.14 | (2, 2, 1.0) — near the anchor 0 corner |
  | 5.65, 1.63, 7.85, 5.71 | (2.5, -2.5, 0.5) — needs more than 7 m on line 2 |

  The last row is the point: the far corners genuinely need spans from 1.6 to 7.9 m,
  which is why the sliders run the full 0–7 and why `assumed_full_line_length` is 7.5 m.

- **backspace** — reset (to qpos0 with ctrl 0; click **Load key** again after)
- **Rendering** panel → check **Tendon** and **Actuator** to see line tension colouring
- double-click a body to select it, then **ctrl + right-drag** to apply a force —
  the easiest way to set the pole swinging
- **Watch** panel, or the **Sensor** figure, for line lengths and tensions

If `simulate` can't find its libraries, run it with
`LD_LIBRARY_PATH=~/Downloads/mujoco-3.12.0-linux-x86_64/mujoco-3.12.0/lib`.

To poke at it from Python instead, `mujoco.viewer.launch('simulator/stringman_arp_carbon270.xml')`.

## What's in the model

Dimensions come from three places: `src/nf_robot/common/definitions.py`, the anchor
layout in `conf_simulator.json`, and the GLB models in
`playroom-ui/public/assets/playroom/models/`. Each is named in a comment at its use site.

- Two powered anchors diagonally opposite each other at (3, 3, 2) and (−3, −3, 2),
  and two passive eyelet posts at the other two corners. Each anchor's indirect line
  runs along a wall to the corner one tick **counterclockwise** from it, seen from
  above: anchor 0 (3, 3) → post A (−3, 3), anchor 1 (−3, −3) → post B (3, −3). Both
  fixed runs are 5.9301 m.
- Four support lines as spatial tendons, length-limited, driven by one-sided position
  actuators so they pull but never push and go slack when over-payed. Each anchor
  spool drives one *direct* line to the gantry and one *indirect* line routed through
  a corner post.
- Gantry, pole and gripper as a rigid chain: the lines converge on the gantry origin
  and constrain only that point, so the whole assembly swings about it as the pendulum
  `host/swing.py` cancels. The gripper hangs `pole_offset_carbon270` = 0.4457 m below.
- A red payload box on the floor, and sensors for the four line lengths and tensions,
  gantry/gripper position, and the gripper IMU.
- Cameras: `gripper_cam` (wide, aimed down and tilted 9.06° back), `anchor0_cam`,
  `anchor1_cam`, and a `room` overview.

### Geometry extracted from the GLBs

`meshes/` holds four OBJs pulled out of the playroom GLBs by walking the glTF
node hierarchy, baking each node's world transform into the vertices, and welding
duplicates. They are in metres and already in the frame of the body that uses them.

**Gripper** (`gripper.glb` → `gripper_shell.obj`, `finger_left.obj`, `finger_right.obj`).
That file's scene frame turned out to be exactly `definitions.py`'s gripper frame — its
`grommet` empty lands at (0.0006, 0.1167, 0.0118) against the constant's
(0, 0.115, 0.013), and its `camera` empty at (0.0004, 0.0026, −0.0261) against
(0, 0.006, −0.027). So the meshes drop in untransformed and every site is in raw CAD
coordinates. Shell is 103.4 × 104.4 × 110 mm, deeper behind the origin than in front.

The fingers are the correction that mattered most. There are **two**, pivoting at
x = ±0.025 and closing **side to side** — not one finger against a fixed jaw, which is
what this model had before. `gripper.glb` draws them closed, tips meeting at x = 0, so
joint 0 is closed and they swing outward over the 59° of
`gripper_arp_server.FINGER_TRAVEL_DEG`. That is a **232 mm aperture** at full open,
large but plausible for a laundry-and-toys gripper. They are geared together on one
servo, so one actuator drives `finger_left` and an equality constraint carries
`finger_right`.

**Gantry and marker card** (`gantry.glb` → `gantry_body.obj` + `gantry_tex.png`). The
gantry is essentially the marker-card assembly: a card with a four-arm cross on top that
the lines attach to. Both come straight out of the file:

| | |
|---|---|
| card | 91.5 × 91.5 mm, 14.0 mm thick, dead flat (0.00 mm out of plane) |
| card centre | 37.0 mm below the gantry origin, on the vertical axis |
| card facing | horizontal, normal at 45° to the GLB's axes |
| tagged faces | both front and back |
| cross arms | 4, at ±45° and ±135°, reaching r = 0.0563 m |

The card is textured, and the texture *is* the AprilTag, so it is carried through as
`gantry_tex.png` and renders as a readable tag rather than a white square — worth having
if you ever point the anchor cameras at it. 91.5 mm agrees well with
`cv_common.DEFAULT_MARKER_SIZE` = 94.5 mm, the side length solvePnP is given for this tag
(the separate 0.94 `GLOBAL_MARKER_SIZE_BIAS` is a scale fudge, not a dimension).

The pole is *not* taken from the GLB: that file's pole is 0.41 m, which is no configured
pole, so it is drawn from `pole_offset_carbon270` instead.

Colours are cosmetic: copper gripper shell, `#0079b9` fingers.

## Driving it from the real firmware

`robot_simulator.py --mujoco` runs the actual `AnchorArpServer` and
`GripperArpServer` against this model instead of against constant stubs:

```sh
venv/bin/python simulator/robot_simulator.py --mujoco
```

`mujoco_bridge.py` supplies the hardware interfaces the simulator otherwise
stubs out. Nothing in `nf_robot` changes - the servers cannot tell the difference.

| stub | becomes |
|---|---|
| `DaMiaoController` / `DaMiaoMotor` | a spool integrating the velocity commands the real spool loop sends, through the firmware's own `SpiralCalculator`, into a tendon actuator; torque comes back from the tendon's tension |
| `SimpleSTS3215` | the wrist and finger joints |
| `MPU6050` | the gripper's gyro / accelerometer sensors |
| `VL53L1X` | a rangefinder ray cast down from the gripper |
| `ADS1015` / `AnalogIn` | pad contact force, mapped back onto the FSR's voltage curve |

Two details that are easy to get wrong. The MuJoCo actuators are commanded in **free
span** while a spool pays out that plus the fixed 5.93 m cross-room run, so the run is
subtracted on the way in; miss it and the indirect line silently clips to its ctrlrange
and hangs slack while the other three take its load. And a freshly started spool loop
has no zero angle, so it believes its whole 7.5 m winding is out - `seed_reference_lengths`
calls the same `setReferenceLength` the host uses during calibration, so the simulator
starts consistent instead of reeling in line that was never out.

Physics runs on its own thread paced to the wall clock, because the servers are
real-time. Verified: brought up with both anchors and the gripper, the gantry holds
(0, 0, 1.181) with 13.6-14.0 N per line and no drift over 15 s, reeling a spool in
raises it and hits the 40 N line limit, and the rangefinder reads the gripper's true
height above the floor.

### Camera feeds

The three video streams are MuJoCo renders of the model's own cameras, h264 over mpegts
on the same TCP ports the test-pattern streams used, so nothing downstream changes.
Sizes come from `component_server.stream_modes`, which is what the real components
produce:

| stream | camera | size | rate |
|---|---|---|---|
| anchor 0 / 1 | `anchor0_cam` / `anchor1_cam` | 1920x1080 | 15 fps (`anchor_control`) |
| gripper | `gripper_cam` | 684x384 | 54 fps (`gripper_control`) |

Field of view is baked into the model's cameras, from the *camera calibration the
detection pipeline interprets the frames with* rather than the module's datasheet:
`fovy = 2*atan(h/2 / fy)` gives 41.535 deg for the anchors (`cameraCal`, fy = 1424) and
42.557 deg for the gripper (`cameraCalWide`, fy = 493). Render and calibration then
agree, which is the whole point - at 16:9 the anchor figure gives 67.97 deg
horizontally, matching that same matrix's `fovx`. (`definitions.rpi_cam_3_wide_fov`,
102 x 67, is the module's full-sensor field of view, not the streamed crop's, and
nothing in the codebase reads it.)

**End-to-end check:** pull a frame off the live stream, run the codebase's own AprilTag
detector on it, and it finds the gantry tag; `solvePnP` recovers the camera-to-card
range to within 1.4-2.4%. That residual is itself right - the object points are
`DEFAULT_MARKER_SIZE` x `GLOBAL_MARKER_SIZE_BIAS` = 88.8 mm against the model's true
91.5 mm card, which predicts about 2.9% short.

Three things are hidden from the camera views that the interactive viewer still shows:
sites (one is a translucent overlay right on the tag), the rangefinder ray, and
**tendons**. MuJoCo draws a line as an opaque 2.5 mm tube and one of them crosses the
card from the anchors' viewpoint, cutting the tag's quad in half so nothing detects.
Real fishing line, close to the lens and far outside its focus, does not do that.

Rendering cost is the thing to watch: 1080p twice over is 1.2 ms/frame on a GPU
(`MUJOCO_GL=glx` or `egl`) and 85 ms under software rendering (`osmesa`), which cannot
keep up. The streamer says so once if it falls behind rather than quietly running the
cameras slow.

**For reinforcement learning this is the wrong shape**, deliberately. It is the fidelity
check - it exercises the spool tension logic, the swing filter and the whole control
stack. Training wants no wall-clock pacing and no websockets in the loop, which calls
for a separate gymnasium `Env` driving `MujocoWorld` directly, with policies trained
there and then run against this bridge before they touch hardware. See the note at the
foot of `mujoco_bridge.py`.

## How trustworthy is it

It compiles, settles, and behaves. Checks that were run:

| check | result |
|---|---|
| hangs stable at keyframe 0 | settles at z = 1.181, no drift over 8 s |
| tension per line | 13.8 N, evenly split — under `maxSafeTension` = 18 N |
| resting yaw | +89.8°, and a 2 rad/s yaw kick peaks at only 4.5° |
| reel in one line | gantry moves toward that line's corner and rises |
| drive to commanded points | 9–19 mm of the target across the room |
| reach the corners | (±1.5, 1.5, 1.0), (−2, −2, 1.1), (2.5, 0, 0.9) all within the 0–7 m span |
| over-reel all four | stalls at 1.47 m against the 40 N line limit, no runaway |
| finger travel | 8.6 mm aperture closed, 232 mm open; the two jaws track to 0.002 rad |
| grasp | jaws close on the payload and hold it at ~6 N per side |

Driving it is open-loop here: feed each slider the geometric distance from its pull
point to that line's own hook and the gantry lands within a few centimetres. Compute the
distance to the gantry *origin* instead and you will be off by 0.1–0.2 m, because at
these shallow line angles a 20 mm length error becomes ~170 mm of height. That is worth
knowing before using this model to check anything host-side.

Known to be wrong or guessed, in rough order of how much it matters:

- **Masses and inertias are estimates.** Nothing in the firmware repo records them.
  Gantry 0.35 kg, pole 0.045 kg, gripper 0.60 kg, finger 0.04 kg. The tension figure
  landing just under `maxSafeTension` is the only evidence they are in the right range.
- **The swing period is ~10% long.** `pole_length_carbon270` = 0.3089 m effective; the
  model gives 0.340 m. Lower the gripper's `diaginertia` to close the gap.
- **The four lines attach 55 mm apart, on the cross arms.** This used to be a 15 mm
  fudge invented to stop the gantry spinning freely about z. It is now real: the arms
  and their radius are both measured off `gantry.glb`.
- **Which arm each line clips to decides whether the tag is visible at all.** The card's
  normal runs along one arm, so the rigging sets the gantry's resting yaw and hence the
  card's heading. With the anchors diagonally opposite: put line 0 on the card's own arm
  and **both** anchors see the tag 11.3° off-normal; put it one arm over and both are
  89.8° off — edge-on, undetectable. The model uses the first. Worth confirming against
  how the real gantry is clipped up.
- **No finger force control.** The real finger runs a force PID off a pad pressure
  sensor (`gripper_arp_server.py`); here it is a plain position servo. The visual
  meshes are exact, but collision is carried by primitives fitted to them — MuJoCo
  would otherwise collide a mesh by its convex hull, which for a claw fills the whole
  grasp opening.
- **The marker card disagrees with `definitions.py` by 28 mm.** `gantry_flat_april`
  puts it 65.2 mm from the gantry origin; the GLB puts it at 37.0 mm. The model follows
  the GLB. The *rotation* does agree — `rvec (π/2, 0, 0)` turns the tag normal onto −y,
  horizontal exactly as the GLB has it, which also settles that the `definitions.py`
  gantry frame is z-up while the gripper frame is y-up. Only the offset is in dispute,
  and if `definitions.py` is what the vision stack calibrates against, that 28 mm is a
  real bias worth chasing rather than a modelling detail.
- **Line stretch is the accuracy floor, and it may be too stiff rather than too soft.**
  Lines act as springs at 4000 N/m, so 13 N of tension is ~3 mm of stretch per line and,
  at these shallow angles, a few centimetres of droop. Real monofilament stretches
  considerably more than that.
- **Near the corners the geometry goes singular.** At (2, 2, 1.0) one line is short and
  steep while the other three are nearly flat, and the positioning error grows to
  ~0.17 m. This is the real conditioning of the cable geometry, not a solver problem.
- **Spools are modelled as pure length commands.** No spiral spool geometry
  (`damiao_spool_geometry`), no line stretch model, no eyelet friction. Each line is
  capped at 40 N, a bit over twice `maxSafeTension`; that cap is also what stops an
  over-reel from squeezing the gantry up through the anchor plane, which
  `position_estimator.find_hang_point` forbids on the real machine.
- **The top of the workspace is tension-limited, and that is real.** With the anchors
  only 2 m up and 8.5 m apart on the diagonal, lifting near the ceiling needs far more
  line tension than the machine has. Raise the anchors in the model if you want more
  usable height.
