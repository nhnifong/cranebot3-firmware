# Placer dataset formats

Two datasets from the same teleop recordings, both produced by
[mine_teleop.py](mine_teleop.py):

- **Drop pairs** (`--mode pairs`, the default): a clean gripper snapshot of an item before
  it was grasped, paired with where the operator dropped it. For predicting a drop point
  from what the item looks like (organizer/readme.md).
- **Carry frames** (`--mode carry`): every frame of the carry before a release, labelled
  with when the operator opened and where the release was. For deciding when to let go.

Both are parquet shards with the frames as JPEG bytes in the row - the same layout as the
visual servoing dataset (`visual_servoing/readme.md`), so the same loader patterns and the
same frozen DINOv2 trunk apply. A null label means "mask this head's loss for this row",
never "the answer is zero". The two have different columns, so each needs its own
`--output_root`.

## The release

Both hang on one release per episode: the start of the **last opening that let go of something** - an
open command (`finger_speed < 0`) across which the finger pressure falls by at least 0.02,
and after which the pressure never rises again by more than 0.02. An episode with no such
opening is skipped, and so is one whose release has no grasp and lift before it.

Pressure alone is not enough to find it. A light grip (a stuffy) carries at 0.035-0.048,
under both the 0.1 grasp threshold and the 0.05 "object gone" level, so the drop in
pressure that marks it is small and the open command is what dates it. On `nick-sep14`
this finds a release in all 72 episodes, 22 of them after one or more regrasps.

## Drop pairs

Why pairs: while an item is carried it covers much of the gripper camera, and the overhead
view is a poor place to find a container. It is a composite of two angled anchor cameras
blended to line up at floor level, so anything tall - a laundry basket - is drawn twice,
leaning out in two directions, and has to be read where the two copies overlap at the
floor; and one anchor's view of a container can be blocked outright (in Nick's bedroom the
basket is hidden from one side by a dark blanket on the bed). So a drop point is hard to
see at the moment it matters. The item, on the other hand, is in plain view just before it
is grasped - which is when to decide where it goes.

Rows are item snapshots: up to `--snapshots` (4) gripper frames, evenly spread over the
frames before the last grasp where the laser reads 0.12-0.25m and the close has not yet
started - near enough that the item fills the frame, early enough that the fingers are not
across it. Every row of an episode shares that episode's drop.

| column | type | meaning |
| --- | --- | --- |
| `image` | binary | the item snapshot: gripper camera JPEG, 448x252 |
| `snapshot_overhead_image` | binary, nullable | overhead view at the snapshot, 448x448: the room as it looked when the decision would be made |
| `release_overhead_image` | binary, nullable | overhead view at the release onset, 448x448. It shows the robot at the drop point, so it is the answer, not an input |
| `drop_view_image` | binary | the gripper camera 0.3s after the opening ends: the empty gripper looking straight down on the drop point it is centred over |
| `source_repo_id`, `episode_index`, `frame_index` | | where the snapshot came from |
| `release_frame_index` | int32 | frame of the release onset |
| `task` | string | the episode's task string |
| `seconds_before_grasp` | float32 | grasp time minus snapshot time |
| `release_room_xy` | list<float32>[2] | the jaws at the release onset, room frame, metres |
| `release_ortho_uv` | list<float32>[2] | the same point in the overhead frame, normalized |
| `release_height_m` | float32 | jaws height at the release above jaws height at the grasp |
| `pickup_room_xy`, `pickup_ortho_uv` | list<float32>[2] | where the item was picked up, same frames |
| `state` | struct | `laser_rangefinder`, `finger_angle`, `wrist_angle`, `gripper_z` at the snapshot |

`drop_view_image` is the "go and take a good look at each drop point" idea done from the
recordings: after every release the gripper is empty, centred over the drop point and
looking down, and on `nick-sep14` those frames show the toy box, the hamper and the bed
cleanly. They are the natural material for recognising a drop point again, or for a
catalogue of the room's drop points that a snapshot is matched against, rather than
regressing a location in room coordinates that differ between robots and calibrations.

## Carry frames

Rows run from the lift (the start of the carry, `visual_servoing.mine_teleop.find_lift`)
to `--post_seconds` (1.0) after the opening ends, at most `--max_carry_seconds` (10) before
it begins. Frames before the lift belong to the grasp and to visual servoing, not here.

| column | type | meaning |
| --- | --- | --- |
| `image` | binary | gripper camera JPEG, 448x252 (the visual servoing input size) |
| `overhead_image` | binary, nullable | ortho floor view JPEG, 448x448 (the ortho_target input size); null if the source has no `overhead_camera` |
| `source_repo_id`, `episode_index`, `frame_index` | | where the frame came from |
| `task` | string | the episode's task string, e.g. "put a toy in the toy box" |
| `seconds_to_release` | float32 | release onset time minus this frame's time; negative after the onset |
| `open_now` | int8 | 0 before the release onset, 1 from it on |
| `holding` | int8, nullable | 1 before the onset, 0 once the opening has ended, null while it is under way |
| `release_offset_m` | list<float32>[3], nullable | from the jaws now to the jaws at the release onset, in the gripper's body frame (x, y lateral, z up); null after the opening ends |
| `release_height_m` | float32 | jaws height at the release onset minus jaws height at the grasp: how high above the pickup floor the operator let go. Constant per episode |
| `release_uv` | list<float32>[2], nullable | the floor point straight below the release jaws, projected into this gripper frame, normalized (0..1 is the visible frame); null when it is behind the lens. Not checked for visibility - the carried object often covers it |
| `release_ortho_uv` | list<float32>[2], nullable | the release jaws' room x, y in the overhead frame, normalized; null without an overhead feed. Constant per episode |
| `gripper_ortho_uv` | list<float32>[2], nullable | the jaws now, in the overhead frame |
| `finger` | float32 | commanded finger speed / 90, as in visual servoing |
| `offset_method` | string | how `release_offset_m` was computed, see below |
| `state` | struct | `laser_rangefinder`, `finger_angle`, `target_force`, `finger_pressure`, `wrist_angle`, `gripper_z` (room frame, raw) at this frame |

"Floor" is the height of the jaws at the grasp, which is where the object was picked up
from. The absolute room z of the position estimate wanders by about ±0.2m between
episodes, so heights are measured from that instead of from zero.

## Where the labels come from

Every position label is computed from telemetry, not from pixels. That is on purpose:
while carrying, the object in the jaws covers much of the gripper camera's view of what is
below it, and the laser rangefinder reads the object rather than the floor (0-0.3m during
a carry on `nick-sep14`). Optical flow worked for visual servoing because the target was
visible all the way in; here the thing to be found - the container - is the thing most
likely to be hidden, so a pixel tracker has nothing reliable to follow.

`--offset_method` picks the telemetry:

- `room-delta` (default): reported jaws position at the release minus reported position
  now. The gripper is in the air for the whole carry, so the touchdown errors that made
  this method worst for visual servoing do not apply, and it is the same estimate the
  robot will steer by.
- `dead-reckon`: the commanded velocity (gripper frame, turned by `-spin`) integrated back
  from the release.

They disagree more than expected. Across the 72 releases in `nick-sep14`, the lateral gap
between them is a median 8cm half a second before the release, 13cm at one second, 15cm
at two and 18cm at four. Neither is ground truth. The overhead view can settle which one
is right: `gripper_ortho_uv` should land on the gripper in `overhead_image`, and a
consistent miss there is the position estimate being wrong.

## What nick-sep14 shows

- The operator lets go a median 0.41m above the pickup floor (10th-90th percentile
  0.21-0.67m), so drop height varies enough to be worth predicting.
- The open command leads the physical release by about a second: the gripper first
  unloads its grip force, then the pressure falls, then the fingers open. `open_now` marks
  the command, which is what the robot has to issue.
- The floor below the release point projects inside the gripper frame on only 28% of
  carry frames, before counting occlusion at all - it is usually still off to the side.
  When a small toy is carried the frame is clear and the container shows up under the
  mark; laundry hangs down one side and the hamper is still visible in the other half; a
  stuffy fills the whole frame for the entire carry.

## Not here yet

- **Drop success.** Every release in a teleop recording is one the operator chose, so
  there are no negatives to learn "this would miss the bin" from. That head needs
  recordings of releases that missed, or deliberately bad synthetic ones.
- **Container height.** Nothing records where the container rim is, so `release_height_m`
  says how high the operator let go, not how far above the rim.
