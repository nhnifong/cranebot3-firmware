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

## Naming

A file named for a room alone is the robot's current calibration. One named for a
dataset as well is the calibration that was active while that dataset was recorded,
kept because it no longer matches the robot's current one:

    conf_playroom.json                    matches move_clutter_2 episodes 71-72, 106-152
    conf_bedroom_move_clutter_2.json      matches move_clutter_2 episodes 153-187
    conf_nick_move_clutter_nick_2.json    matches move_clutter_nick_2 episodes 48-56
    conf_demo_79west4.json                matches move_clutter_79west4 episode 1
    conf_nick.json                        matches naavox/nick-aug3

Each was confirmed by recovering frameRoomSpin from the recording and finding it equal
to the config's, then checking the recorded gantry positions fall inside that config's
anchor footprint. Nothing else is kept: a config that matches no dataset cannot be used
to convert one.

Datasets recorded from August 2026 onward carry their own anchor poses in an
`anchor_poses` feature and do not need a file here at all.
