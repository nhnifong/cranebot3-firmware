# anchor_diffusion

Synthetic ortho frames for the `ortho_target` model: an SDXL LoRA fine-tuned on grip_o's
`overhead_camera` ortho view paints new floors, those frames are hand-labelled like real ones,
and the labels are added to the **train** split only.

## What was found (Oct 2026)

- **Base model: SDXL 1.0 at 1024, LoRA rank 32.** Ungated, openrail++, and its LoRA trains
  well on the 5090. SD 1.5 and SDXL at 512 were also tried and dropped. Frames are upscaled 2x
  to train and downsized back to 512. The VAE is `madebyollin/sdxl-vae-fp16-fix`, because the
  stock one NaNs in fp16.
- **Prompts: weakened LoRA plus random rooms.** At full strength the LoRA just reproduces
  grip_o's rooms, which the model already handles. At strength 0.4–0.7, with random room,
  floor, furniture and clutter prompts, it keeps the ortho look and paints new rooms.
- **Effect, on a held-out room:** trained on `datasets/ortho_target_complete` (hand labels
  only, 60 epochs) and scored on James's room, which is never seen in training:

  |                 | ap@20cm | f1@20cm | precision | invented/frame |
  |-----------------|---------|---------|-----------|----------------|
  | real only       |   0.548 |   0.629 |     0.753 |            4.6 |
  | + 145 synthetic |   0.570 |   0.648 |     0.804 |            3.4 |

  These were scored before the eval was relabelled. Synthetic frames slightly reduced invented
  targets, with recall unchanged. The eval is only 19 frames, so this suggests a gain but does
  not prove one.

## Files

| file                       | what it does                                                                           |
|----------------------------|----------------------------------------------------------------------------------------|
| `extract_ortho.py`         | grip_o ortho frames + captions for LoRA training                                       |
| `train_lora_sdxl.py`       | the LoRA trainer                                                                       |
| `generate_for_labeling.py` | variety-prompt frames, appended to the labeler cache with a manifest                   |
| `james_holdout.py`         | train/eval split that holds out James's room, and re-applies the corrected eval labels |
| `runs/grip_o_ortho_r32/`   | the current LoRA (`pytorch_lora_weights.safetensors`)                                  |

Data lives outside the repo (all gitignored):

| path                                             | what it holds                                                                                                                        |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `ortho_labeler_frames/synthetic_sdxl1024_ortho/` | generated frames and `manifest.jsonl` (`kind`: `room_lora1.0` for the first 54, `variety` after)                                     |
| `ortho_target_user_labels_synthetic/`            | labels of those frames. Kept out of `ortho_target_user_labels/` so nothing synthetic reaches a real dataset or an eval split         |
| `ortho_target_user_labels_james_eval/`           | corrected labels of the 19 James eval frames. The originals are on the hub, so `james_holdout.py` re-applies these after every merge |
| `datasets/ortho_target_complete/`                | real hand labels + empty-floor negatives, James's room as eval                                                                       |
| `datasets/ortho_target_complete_synth/`          | the same plus synthetic frames in train only. `models/ortho_target_complete_synth.pth` was trained on it                             |

## After grip_o grows

Run everything from the repo root with the repo venv. Skip a step if nothing it reads has changed.

### 1. Hand-label the new recording

Label ~10 frames per new source recording, the same way the existing sources were done, then
back them up:

```bash
python -m nf_robot.ml.ortho_target.labeler --repo_id naavox/<new-recording> --limit 10
python -m nf_robot.ml.ortho_target upload_labels
```

If the new recording is **in James's room**, add its grip_o episode range to `JAMES_EPISODES` in
`james_holdout.py`, or its frames will leak into train. Add new sources at the end of
`recipes/grip_o.yaml`, so the existing ranges stay where they are. The new range starts at the
old `total_episodes` of grip_o.

### 2. Retrain the LoRA on the larger grip_o

```bash
GRIP_O=$(hf download naavox/grip_o --repo-type dataset --include "meta/*" --include "videos/observation.images.overhead_camera/*" | tail -1)
python experiments/anchor_diffusion/extract_ortho.py --root "$GRIP_O" --output_dir experiments/anchor_diffusion/data/grip_o_ortho
cd experiments/anchor_diffusion
PYTORCH_ALLOC_CONF=expandable_segments:True python train_lora_sdxl.py --data_dir data/grip_o_ortho \
    --output_dir runs/grip_o_ortho_r32 --width 1024 --height 1024 --max_steps 3000 --gradient_checkpointing
cd ../..
```

The run takes about 35 minutes and overwrites the current LoRA. Check
`runs/grip_o_ortho_r32/samples/step_003000.jpg` looks like real ortho frames. To keep the old
LoRA, pass a different `--output_dir` and the same path as `--lora` below.

### 3. Generate synthetic frames

```bash
python experiments/anchor_diffusion/generate_for_labeling.py --num 300
```

New frames are numbered after the last one in the cache. Already-labelled frames are untouched.

### 4. Label them

```bash
python -m nf_robot.ml.ortho_target.labeler --root synthetic_sdxl1024_ortho \
    --cache ortho_labeler_frames/synthetic_sdxl1024_ortho \
    --output ortho_target_user_labels_synthetic --no_carry
```

- `--root` only names the source. It is never read, because the cache is already full.
- `--output` is what keeps synthetic labels separate. Never point it at `ortho_target_user_labels/`.
- `--no_carry`: every frame is an independent scene.
- Turn on "Skip finished" to walk past frames labelled in earlier sittings.

### 5. Rebuild the datasets

```bash
rm -rf datasets/ortho_target_complete datasets/ortho_target_complete_synth
python -m nf_robot.ml.ortho_target merge_labels --output datasets/ortho_target_complete
python -m nf_robot.ml.ortho_target merge_labels --repo_id naavox/ortho-target-user-labels --output datasets/ortho_target_complete
python -m nf_robot.ml.ortho_target merge_labels --repo_id justink04/ortho-target-user-labels --output datasets/ortho_target_complete
python -m nf_robot.ml.ortho_target distill_negatives --output datasets/ortho_target_complete
python experiments/anchor_diffusion/james_holdout.py --data_root datasets/ortho_target_complete

cp -r datasets/ortho_target_complete datasets/ortho_target_complete_synth
python -m nf_robot.ml.ortho_target merge_labels --source ortho_target_user_labels_synthetic \
    --output datasets/ortho_target_complete_synth --split train
```

- Add a `merge_labels --repo_id` line for any other volunteer's label repo.
- `james_holdout.py` replaces `split`. Don't run `split` on these datasets: it deals rows at
  random, which puts James's room into train.
- `james_holdout.py` should report `swapped in 19 of 19`. A lower number means a corrected frame
  is missing from the pool.

### 6. Train and score

```bash
python -m nf_robot.ml.ortho_target train --data_root datasets/ortho_target_complete_synth \
    --model_path models/ortho_target_complete_synth.pth --epochs 60
python -m nf_robot.ml.ortho_target evaluate --data_root datasets/ortho_target_complete_synth \
    --model_path models/ortho_target_complete_synth.pth --preview_dir previews_complete_synth
```

Training both with and without synthetic frames is a comparison on the same eval. Train
`datasets/ortho_target_complete` into its own `--model_path` as well. To ship, follow steps 6–7 of
[the ortho_target readme](../../src/nf_robot/ml/ortho_target/readme.md). For a final model
meant for the robot, James's room could go back into train: use `split` on a separate copy.
