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

| | |
|---|---|
| click empty floor | add a target |
| drag a dot | move it |
| right-click a dot | delete it |
| <kbd>Del</kbd> | delete the selected dot |
| <kbd>Space</kbd> | save and go to the next |
| <kbd>N</kbd> | skip |
| <kbd>←</kbd> <kbd>→</kbd> | move about without saving |

Amber dots are the model's guesses, labelled with their probability, seeded at a threshold
well below the one the robot acts on — deleting a wrong dot is faster than placing a right
one. Green dots are yours. Moving a seed makes it green, since its probability described
where the model put it rather than where you did.

Labels land in `ortho_target_user_labels/`, one parquet per frame named for the frame, so
re-labelling one overwrites it. Feed them to training the ordinary way:

```
python -m nf_robot.ml.ortho_target merge_labels
python -m nf_robot.ml.ortho_target train --data_root ortho_target_data
```

Nothing here imports from the tool into `nf_robot`; it only calls the other way, and the
labels are written by `ortho_target.write_user_labels` so they cannot drift from the
schema training reads.

## Known gap

A frame with nothing graspable in it is a real and useful label, and this cannot save one
— saving needs at least one target, so skip those. Supporting them means letting
`write_user_labels` write a row with no points, which changes what the UI's own "Add
targets to dataset" action means too.
