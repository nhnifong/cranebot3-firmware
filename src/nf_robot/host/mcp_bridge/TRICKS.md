# Stringman tricks

Practical lessons from driving the robot through MCP. Read alongside GUIDE.md.

## Grasping

- **Dive with the fingers nearly closed in clutter.** Close to ~80 before descending, and open them only near the object (laser range ~0.08 m). Opening pushes neighbours aside instead of a finger landing on them and stalling the descent. On an isolated object it only adds a step; dive open.
- **Be opportunistic in a pile.** Auto GRASP over a pile reliably comes up with one or two pieces in ~18 s, much faster than aiming at a particular one. Take whatever comes; if it is more than one, drop them from ~0.3 m onto open floor so they bounce apart, then pick them singly.
- **Grip by force, never by finger angle.** Use `grip` / `ob.close_grip()`: it closes at a constant speed, the gripper switches to force mode on contact, and the force is held. Closing to a fixed angle on an object overloads the finger servo, whose cutout then drops the object, usually mid-carry.
- **Open fingertips reach ~0.10 m below the rangefinder.** With the fingers open the laser never reads less than ~0.10 over bare floor; open them at ~0.15 m, then lower in small steps and stop when the pole starts to lean (fingertips on the floor) and close there.
- **White objects are hard to segment** in the palm camera: the fingers are white too. Prefer auto GRASP after centering on an isolated white object.
- **People and children come into the work area.** Watch the frames for hands and feet, and stop when someone is reaching in.
- **A held object blinds the rangefinder** and fills the middle of the palm camera. Use the estimated height while carrying, and ignore blobs in the middle of the frame when looking for things on the floor.
- **The palm camera's principal point is the grasp point.** Center the target on the image center, not on the gripper position: the projection of the image center onto the floor sits 15-20 cm from `gripper_position()` and that offset is estimate error, not geometry.
- **Recenter at ~0.3 m range before the final drop.** The gantry drifts several centimetres during a descent.
- **Image right in the room frame** is `ob.gripper_camera_to_room([1, 0, 0])`; use it to turn a pixel offset into a nudge. At range r, one pixel of the 684 px frame is about r / 302 m.
- **Fingers close along image horizontal.** Rotate the wrist so the object's long axis is image vertical.
- **Nothing within ~20 cm of a wall is graspable.**
- **Automatic GRASP can wander** to whatever the servo model finds most prominent, up to ~0.7 m away. Near things that must not be touched, call `ob.grasp(max_wander_m=...)` from a script.
- **A tilt over ~7 deg near the floor means the gripper is resting on something.** Lines go slack and lateral nudges barely move it. Lift a few cm before moving sideways.

## Moving things

- **Lay long floppy objects straight by lift-and-sweep.** Lift until it hangs nearly free, carry it over open floor, then move horizontally while descending so it unrolls behind the gripper.

## Seeing

- **The position estimate drifts 5-20 cm** between visits, worst after a marker dropout. Absolute coordinates are not good enough for tight layouts; place relative to what the palm camera sees, and check the result in the ortho view.
- **The marker drops out when the gantry is low** (below ~0.6 m), so long work near the floor runs on dead reckoning. Keep low phases short.
- **Ortho pixels to room:** on the 1000 px ortho, x = (u - 500) / 200, y = (500 - v) / 200 (`ml/ortho_target/model.py`).
- **The robot itself shows up in the ortho** as a smeared pink/black shape with a line: the gripper and pole projected from the air. It is not an obstacle on the floor.
- **Crop the ortho at full resolution** (`ob.latest_ortho()`, save a crop to a file and read it) to count small objects; the 512 px tool image is too coarse.
- **The marker-fault popup can clear on its own** once the gantry rises.
