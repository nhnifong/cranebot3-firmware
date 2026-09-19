## drop point selector

From available cameras, predict a drop point for an item

A clean picture of the object before we blind our gripper camera by grabbing it. best time to do this is when laser range to the object is between about 12 and 25 cm. predict drop point then and save the decision.

We can know the location in the overhead view of the room where the operator dropped each time.

### Automatically categorizing items

All observed items' cls vectors could be clustered and each cluster assigned a differernt drop point. If there are a bunch of peices that belong to the same toy for example, it would be preferable to put them all in a pile together than to put them all in a common toybox. We might select several viable drop points in the room, and then find the best match to the set of clusters.