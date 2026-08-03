Calibration files from robots that were used to record datasets
## A config is only valid for the sessions it was recorded with

`frameRoomSpin` is a property of one calibration run, not of a room. Recalibrating
means placing the origin card somewhere new, which redefines the room frame: the
anchors stay bolted to the wall, but their poses expressed in room coordinates move
with the origin. A config whose `frameRoomSpin` disagrees with a recording therefore
describes a different coordinate system than that recording's `gripper_pos_*`, and
using it places every anchor-frame goal wrong.

Every frame carries the value configured when it was recorded, since `get_spin()`
computes `spin = radians(wrist_angle) + (frameRoomSpin - pi)` and both terms are in
`observation.state`:

    frameRoomSpin = spin - radians(wrist_angle) + pi

`experiments/check_dataset_robots.py` recovers it per episode, which both confirms a
config belongs to a dataset and finds where a dataset changes robots mid-way.

## conf_playroom_move_clutter.json / conf_bedroom_move_clutter.json

Not usable as they stand. They carry the `frameRoomSpin` recovered from
naavox/move_clutter (1.911334 for its playroom ranges, 1.900022 for its bedroom
ranges) but the anchor poses of a later calibration, copied from
conf_playroom/conf_bedroom.json. Writing the old spin into a newer config does not
move the anchor poses back into the old room frame.

They are kept as a record of the recovered values. To make either usable the anchor
poses have to come from the calibration active during that recording - the archived
config, or solved for as an origin transform between the two frames, which is possible
because the anchors themselves did not move.
