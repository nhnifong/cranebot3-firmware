## Auto placer

The automatic grasping capability provided by host/visual_servo.py and it's network is useful to many generic tasks.

It would be great to have the automatic put-down counterpart to this. one button to put something down, if we are over a table, it sets the thing on the table nicely. if we are over a bin, it gets the thing in the bin.

One thing to try is to just isolate the seconds leading up to a release in the teleop data and try to train multitask dit on it with lerobot.

Another way is to take the same trunk as visual_servo and predict a lateral offset to the eventual release point, as well as a probability from the current frame that finger opening should begin.

There may be just enough visibility in the gripper camera to center the gripper over a container, but only for toys and not laundry.

we may also predict the altitude at which an operator would try to drop something from as a function of the item's snapshot (towels and sweatshirts need more height to prevent overhangs)

### drop success probability head

If negative examples were collected, we coudld predict from camera views of whether, if dropped now, the object would land in the container or not.