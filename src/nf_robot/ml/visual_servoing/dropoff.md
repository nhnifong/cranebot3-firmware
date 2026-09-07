# A model of where and how to drop off items.

learn from data that which is needed to drop off items more intelligently

## Background

Visual servoing has been a fairly successful model. it is charged with predicting specific useful variables needed by an imperative grasping routine, such as a lateral offset, finger angles, pressure, and so on.

It does not however at this time have a flow matching head.

## Plan for a dropoff network

Predict the lateral offset need to center over the dropoff basket.

Predict, from the appearance of the item in ealier frames when it could be completely seen, where in the ortho view the operator would have put it.

Predict from it's appearance, how high above the drop point the operator needed to hang it, and how far down they needed to lower it before letting go. 

Predict whether, if dropped at the current time, the object would fall in the basket. (collect special data for this.)

# dropoff routine

Like the rest of this class of solutions, the routine is imperative and merely uses the predicted scalars.

Move to the desired height over the desired drop point. The servo to the center of the apparent container. The lower down, if called for, by the predicted amount. Alternate between servoing and perturbing the position until the predicted chance that the object would enter the container exceeds a threshold, or we time out, then drop it.

# Mining for a dataset that contains these predictions

We would predict from the gripper frame, the sensors, and the ortho frame.

from teleop data, we would itendify frames in which grasp, drop, and key moves happen

We would back label surrounding frames with key information.

For predicting whether objects go in baskets, the teleop data contain ample frames of drop positions where the object would make it, but we can collect special teleop data of frames where it never makes it in and label them as such.