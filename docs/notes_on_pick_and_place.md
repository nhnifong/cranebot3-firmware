# Pick and Place

## History and Scope

The "pick and place" routine in stringman has undergone a lot of changes. At a high level, the objective is to clear the floor automatically.
At one extreme, this could be done by an end-to-end model that commands the robot to move from start to finish. The systems are in place to try this, but no model capable of doing it has been produced. At the other extreme, the whole procedure would be a handwritten loop full of heuristics. Until and unless a general model can do the whole job, it remains a hybrid approach with a handwritten loop for the high level orchestration, and models for specific subtasks or predictions.

## Current structure

Continuously alternate between grasping an object at a source location and dropping it at a destination location. the destination is one of the tags, and the source is either another tag or one of the targets from the target list. the targets are either user supplied or identified from the overhead cameras by another network. the grasping is performed by sending a start episode command, assuming a lerobot session running a suitable grasping network is connects. several heuristics and timers are used to know when to advance to the next stage in the loop.
And as of now, only the lack of additional targets being identified by the target network would cause the loop to terminate.

## Why it is this way

The more complex the task, the less success I have had training models to perform it. Obviously there are many things yet to try, but so far I just take models are good at something, whether that be grasping or target finding, and then use handwritten heuristics to fill in the gaps. some heuristics are fine. it's easy to drop an item and then move laterally out of the way of the basket it was dropped in. but the heuristic for knowing whether an object was successfully grasped is lacking.

Some heuristics were specifically added to pamper the models and get the most out of them. for example, naavox/dit-grasp-1 gets lost when it can see it's own fingers so we open them up enough that it cannot see them at the beginning of an episode. it also freezes up above a certain altitude, so we avoid that.

No matter the model type, it is probably true that it is only lack of data that prevents me from training an end to end model to perform the whole task.
So for now, the hybrid approach is just fine for getting rooms clean. If its stupid and it works it aint stupid after all.

## Areas for improvement

### Predicting grasp success

Knowing whether an automatic grasp has succeeded is currently done by checking whether a finger pressure rising edge trigger has been set, and the gripper is above a certain height.
In theory a scalar could be added to the robot's action space that represents a prediction of success. it would be added retroactively in the dataset to be 1 during the last half second of every episode and 0 otherwise.

### Judging task completeness

Rather than lack of targets from the network, it would be ideal to have a basline image of the user's room in a clean state to compare with.
The target fidning network would be trained to predict targets from both the current image and baseline image.

### Trying out other model types

* there are other vision encoders that could be tied into dit such as siglip and dino that may fare better at distinguishing the fingers from the targets.
* Maybe I could train my own vision encoder that doesn't care what we're picking up so much as exactly where it is.
* JEPA was recently added to lerobot which supposedly contains a world model and conditions actions based on predictions of future motion.
* Pi0.5 was impressive, but supposedly fine tuning it in lerobot has some bugs. if pi open sources a new version and it ends up in lerobot again, try it.
* Reproduce contact oriented policy

### Incremental improvements to existing heuristics

increase hysterisis on finger pressure rising edge detection. 

### Improvement to target finding network

network was trained on unmodified overhead cams but is now being used to analyze the merged floor space image. should train it directly on a set of merged floor space images.
Consider switching to network with some kind of pretrained image knowledge like clip?