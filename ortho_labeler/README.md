# Ortho target labeler

Hand-label ortho floor frames, starting from the model's own predictions.

Teleop labels one target a frame — whatever the operator grabbed — and says nothing about
the rest of the floor. `ortho_target`'s objectness head needs the other kind: frames where
every target is marked, so an unmarked patch of floor really is empty. This is the tool
that makes them.

```
python ortho_labeler/server.py --root ~/data/combined_targets_reblend   # not on the hub yet
python ortho_labeler/server.py --repo_id naavox/combined_targets_reblend
```

Open the printed URL. Frames extract once into `frames/` (gitignored), so restarts are
instant; `--refresh` re-extracts.

## Which frames you get

Episodes are drawn spread across the whole dataset, not from the front of it. A merged
dataset is its sources laid end to end, so `--limit 300` applied to the front stops inside
the first room or two and never reaches the others — which is exactly what a first pass
over `combined_targets_reblend` did, landing entirely in nick's and justin's rooms. Any
prefix of the order is spread, so blank frames or a smaller limit cannot re-bias it.

Whole episodes are taken at a time, so the frames of one arrive together and carry-over
has a run of the same scene to work along. Episodes from the same recording stay similar
enough to carry labels even tens of episodes apart, so spreading the sample costs nothing.

To go round again for frames you have not seen:

```
python ortho_labeler/server.py --root ~/data/combined_targets_reblend --add 300
```

`--add N` extracts N frames that are neither already cached nor already labelled, keeps
the ones already there, and serves the lot — so N means N to work on, not N drawn. The
frames you have done stay in the list as saved; turn on "Skip finished" to walk past them.

| | |
|---|---|
| click empty floor | add a target |
| drag a dot | move it |
| right-click a dot | delete it |
| <kbd>Del</kbd> | delete the selected dot |
| <kbd>Space</kbd> | save and go to the next |
| <kbd>E</kbd> | save as empty floor, and go to the next |
| <kbd>N</kbd> | skip |
| <kbd>F</kbd> | toggle walking past frames already saved or skipped |
| <kbd>←</kbd> <kbd>→</kbd> | move about without saving |

The line under the image says where the dots you are looking at came from: the model, the
frame before, or a label you saved earlier.

"Skip finished" walks past anything already dealt with — saved, or skipped this run — in
both directions and after every save, so a second pass over a part-labelled set goes
straight down the frames still waiting. It is remembered between runs, since it is a way
of working for a whole sitting rather than a per-frame choice. The page already opens on
the first unlabelled frame either way.

Amber dots are the model's guesses, labelled with their probability, seeded at a threshold
well below the one the robot acts on — deleting a wrong dot is faster than placing a right
one. Green dots are yours. Moving a seed makes it green, since its probability described
where the model put it rather than where you did.

## Carrying labels along a session

Blue dots are the previous frame's, carried forward. Frames come several to an episode and
several episodes to a recording session, and consecutive ones are the same floor with one
more object cleared away — so after the first frame of a scene, each one starts from the
points you just saved rather than from the model, and the work is checking and pruning
rather than placing everything again. Moving one makes it green, like any other seed.

Whether two frames are the same scene is decided by comparing them, not by their episode
numbers: a merged dataset puts different rooms in consecutive episodes. Frames within a
session score above 0.7 even across an episode boundary, while the seam between two
recordings scores near zero, and at a seam the model seeds afresh. `--carry_similarity`
moves the line, `--no_carry` turns the whole thing off.

Carried points are jittered by `--carry_jitter_px` (5px, about 2.5cm of floor). They are a
guess about a moment you have not looked at yet, and a dot that has visibly moved asks to
be checked in a way an exact copy does not. The jitter is a random walk, so accepting a
point unchanged across many frames lets it drift — roughly 5√n px after n frames — which
is the cue to drag it back onto its object.

Labels land in `ortho_target_user_labels/`, one parquet per frame named for the frame, so
re-labelling one overwrites it. Feed them to training the ordinary way:

```
python -m nf_robot.ml.ortho_target merge_labels
python -m nf_robot.ml.ortho_target train --data_root ortho_target_data
```

Nothing here imports from the tool into `nf_robot`; it only calls the other way, and the
labels are written by `ortho_target.write_user_labels` so they cannot drift from the
schema training reads.

## Empty floor is a label

A frame with nothing worth reaching for is the strongest negative this dataset can carry.
An ordinary teleop frame marks the one object the operator grabbed and says nothing about
the rest of the floor; an empty one says every cell of it is a confirmed no. Press
<kbd>E</kbd>, or the "Save as empty" button, and it is written like any other label — a
row with no points.

<kbd>E</kbd> rather than <kbd>Space</kbd> on an empty frame, because space is pressed
dozens of times a session and would otherwise assert "this floor is bare" about a frame
nobody had looked at yet. Space on a frame with no dots says so and does nothing. Pressing
<kbd>E</kbd> when there *are* dots asks first, since it throws them away.

<kbd>N</kbd> (skip) still means "I am not judging this one" and stores nothing, which is a
different statement from "there is nothing here".

The observer's own "Add targets to dataset" action is unaffected: `write_user_labels` takes
`allow_empty` and it is off by default, so a submission whose targets all fell outside the
map is still discarded as the mistake it is rather than recorded as bare floor.
