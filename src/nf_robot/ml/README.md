# ML commands & workflows

Practical, copy-pasteable commands for the machine-learning side of the project.
The bulk of this concerns **lerobot** (dataset recording, policy training locally and
on Modal, and evaluation). The **targeting & centering** models at the bottom are a
separate, non-lerobot pipeline.

> Older scratch notes live in [useful_commands.md](useful_commands.md). Prefer this file;
> reach for that one only if something here is missing.

---

## Environment

Torch that currently works with nvidia driver 580 / RTX 5090:

```bash
pip install --force-reinstall \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.6.0 \
  --index-url https://download.pytorch.org/whl/cu129
```

---

# lerobot

## Recording datasets

```bash
python -m nf_robot.ml.stringman_lerobot record \
  --robot_id=lan \
  --server_address=ws://localhost:4245 \
  --repo_id=naavox/grasping_dataset_c
```

Recording against the local telemetry stack (mediamtx streams) and building datasets
from a recipe are documented in [useful_commands.md](useful_commands.md) — they haven't
changed.

## Training

All training uses `lerobot-train`. Below is one example per policy type. Run them
**locally**; to run the same thing on Modal see [Training on Modal](#training-on-modal).

Conventions used throughout:
- `--policy.repo_id` is where the trained policy is pushed on the Hub.
- `--policy.push_to_hub=true` pushes checkpoints as they are saved.
- `--save_freq` controls how often a checkpoint is written under
  `<output_dir>/checkpoints/<step>/pretrained_model/`.

### Plain multitask DiT

The standard, publicly-loadable DiT. Uses the built-in CLIP vision encoder.

```bash
lerobot-train \
  --dataset.repo_id=naavox/simple_grasp_224 \
  --dataset.root=datasets/simple_grasp_224 \
  --output_dir=./outputs/dit-grasp-1/training \
  --steps=30000 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.objective=flow_matching \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.timestep_sampling_s=0.999 \
  --policy.num_integration_steps=100 \
  --policy.integration_method=euler \
  --policy.sigma_min=0.0 \
  --policy.repo_id="naavox/dit-grasp-1" \
  --policy.push_to_hub=true \
  --wandb.enable=false \
  --tolerance_s=0.001 \
  --save_freq=5000 \
  --batch_size=92
```

### Multitask DiT with the experimental dino fork

Same as above plus a DINOv3 vision encoder and patch-token cross-attention. These
options **only exist in the `nhnifong/lerobot` fork** (they are not in the public
release), so a checkpoint trained with them requires the fork to load.

Add these two flags to the plain-DiT command:

```bash
  --policy.vision_encoder_name=facebook/dinov3-vitb16-pretrain-lvd1689m \
  --policy.use_visual_cross_attention=true \
```

Full example (larger images, smaller batch to fit the bigger encoder):

```bash
lerobot-train \
  --dataset.repo_id=naavox/move_clutter_combined_384 \
  --dataset.root=/media/nhn/nfdrive/datasets/move_clutter_combined_384/ \
  --output_dir=./outputs/dit-dino-3/training \
  --steps=30000 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=multi_task_dit \
  --policy.device=cuda \
  --policy.vision_encoder_name=facebook/dinov3-vitb16-pretrain-lvd1689m \
  --policy.use_visual_cross_attention=true \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.objective=flow_matching \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.timestep_sampling_s=0.999 \
  --policy.num_integration_steps=100 \
  --policy.integration_method=euler \
  --policy.sigma_min=0.0 \
  --policy.repo_id="naavox/dit-dino-3" \
  --policy.push_to_hub=true \
  --wandb.enable=false \
  --tolerance_s=0.001 \
  --save_freq=5000 \
  --batch_size=44
```

> **Fork vs public compatibility.** A checkpoint trained on the fork carries fork-only
> config fields (`use_visual_cross_attention`, `cross_attention_dropout`,
> `visual_token_pos_embedding`, `keep_cls_in_conditioning`, `aux_heads`) and the public
> lerobot refuses to load such a `config.json`. Modal now defaults to the public release
> (see [`--lerobot_ref`](#training-on-modal)), so this only happens when you deliberately
> pass a fork commit. If you did, but with `use_visual_cross_attention=false`, the model
> is byte-for-byte a public DiT — just delete those five keys from `config.json` and
> `train_config.json` and the public release will load it.

### pi05

```bash
lerobot-train \
  --dataset.repo_id=naavox/grasp_v \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --output_dir=./outputs/pi_grasp_v \
  --job_name=pi05_training \
  --policy.repo_id=naavox/pi_grasp_v \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.use_relative_actions=false \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=20000 \
  --batch_size=32
```

### SmolVLA

First build a local copy of `smolvla_base` with `max_state_dim` expanded to fit our
43-dim `observation.state` (the `state_proj` input is zero-padded from 32 to 64,
preserving pretrained behavior at init):

```bash
python src/nf_robot/ml/lerobot_expand_smolvla_state_dim.py \
    --source lerobot/smolvla_base \
    --output_dir models/smolvla_base_state64 \
    --max_state_dim 64
```

Then train from that local checkpoint:

```bash
lerobot-train \
  --policy.path=models/smolvla_base_state64 \
  --policy.repo_id=naavox/g224_smolvla \
  --dataset.repo_id=naavox/merged_224 \
  --rename_map='{"observation.images.gripper_camera": "observation.images.camera1"}' \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=false \
  --steps=20000 \
  --batch_size=64
```

### JEPA

Fine-tunes `lerobot/VLA-JEPA-Pretrain`. Re-inits the action encoder/decoder and state
encoder, and renames our camera keys to the ones JEPA expects.

```bash
lerobot-train \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.repo_id=naavox/jepa-3 \
  --policy.freeze_qwen=true \
  --policy.pre_snap_gripper_action=false \
  --policy.binarize_gripper_action=false \
  --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]' \
  --policy.gripper_dim=4 \
  --dataset.repo_id=naavox/move_clutter_rect \
  --dataset.root=/media/nhn/nfdrive/datasets/move_clutter_rect/ \
  --output_dir=./outputs/jepa-3/training \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --rename_map='{"observation.images.gripper_camera": "observation.images.image", "observation.images.anchor_camera_0": "observation.images.exterior_1_left", "observation.images.anchor_camera_1": "observation.images.exterior_2_left"}' \
  --wandb.enable=false \
  --steps=30000 \
  --save_freq=5000 \
  --batch_size=320 \
  --num_workers=12
```

## Training on Modal

Any `lerobot-train` command above can run on Modal cloud GPUs. The wrapper
[`lerobot_train_modal.py`](lerobot_train_modal.py) forwards all lerobot args unchanged
and adds Modal-only flags (`--gpu_type`, `--timeout_hours`, `--detach`, `--lerobot_ref`,
`--hf_secret`). See its module docstring for the full list.

**To convert a local command to a Modal command:**

1. Replace `lerobot-train` with `python src/nf_robot/ml/lerobot_train_modal.py`.
2. Point `--output_dir` at a path under `/multitask_dit_data/` — that is the Modal
   Volume (`multitask_dit_data`), and it's the only place checkpoints survive after the
   job ends.
3. Drop `--dataset.root` and `--policy.device` — the dataset is pulled from the Hub and
   the GPU is selected by `--gpu_type`.
4. The default installs the **public** lerobot from PyPI, so checkpoints load on the
   public release. Add `--lerobot_ref <commit>` only when you want the dino fork (see the
   compatibility note above).

Copyable skeleton — paste any policy's flags into the middle:

```bash
python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --gpu_type H200 \
  --timeout_hours 24 \
  --detach true \
  --output_dir /multitask_dit_data/<run_name> \
  --dataset.repo_id=naavox/<dataset> \
  --policy.repo_id="naavox/<model>" \
  --policy.push_to_hub=true \
  --wandb.enable=false \
  --save_freq=5000 \
  --num_workers=10 \
  --batch_size=<n> \
  --steps=<n> \
  <...all the --policy.* flags for the chosen policy type...>
```

Concrete plain-DiT run:

```bash
python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --dataset.repo_id=naavox/move_clutter_rect \
  --output_dir /multitask_dit_data/tidy_modal_14 \
  --steps=60000 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.type=multi_task_dit \
  --policy.horizon=64 \
  --policy.n_action_steps=32 \
  --policy.objective=flow_matching \
  --policy.timestep_sampling_strategy=beta \
  --policy.timestep_sampling_alpha=1.5 \
  --policy.timestep_sampling_beta=1.0 \
  --policy.timestep_sampling_s=0.999 \
  --policy.num_integration_steps=100 \
  --policy.integration_method=euler \
  --policy.sigma_min=0.0 \
  --policy.repo_id="naavox/dit-move" \
  --policy.push_to_hub=true \
  --wandb.enable=false \
  --batch_size=400 \
  --tolerance_s=0.001 \
  --save_freq=5000 \
  --num_workers=10
```

Monitor a detached job:

```bash
modal app logs lerobot-train
# or open https://modal.com/apps/lerobot-train
```

### Resuming a Modal run

Checkpoints persist on the volume at
`/multitask_dit_data/<run_name>/checkpoints/last/pretrained_model/`. Resume by pointing
`--config_path` at that checkpoint's `train_config.json` and setting `--resume=true`. It
picks up the same `output_dir`, optimizer state, and step count from the checkpoint.

```bash
python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --resume=true \
  --config_path=/multitask_dit_data/tidy_modal_14/checkpoints/last/pretrained_model/train_config.json
```

Use the **same `--lerobot_ref`** you trained with, or the resumed model architecture
won't match the checkpoint. To train further than the original target, raise `--steps`
on the resume command.

### Pulling a snapshot from Modal and uploading to Hugging Face

`--policy.push_to_hub=true` already uploads checkpoints during training. Do this
manually when you want a specific snapshot from the volume — e.g. the run finished, was
interrupted, or you want a mid-training step.

1. **List the checkpoints on the volume:**

   ```bash
   modal volume ls multitask_dit_data tidy_modal_14/checkpoints
   ```

2. **Download the newest step's `pretrained_model` dir.** `last` is a symlink that
   `modal volume get` won't follow, so name the numbered step. Create the destination
   directory *first* — `modal volume get` collapses a directory download into a single
   file if the destination doesn't already exist:

   ```bash
   mkdir -p ./dit-move-030000
   modal volume get multitask_dit_data \
     tidy_modal_14/checkpoints/030000/pretrained_model \
     ./dit-move-030000
   ```

   The files land under `./dit-move-030000/pretrained_model/` (config.json,
   model.safetensors, train_config.json, and the pre/post-processor files).

3. **Upload the contents to the Hub.** Use `hf` (the old `huggingface-cli` is dead).
   Do **not** pass `.` as the third positional arg — omit it to target the repo root:

   ```bash
   hf upload naavox/dit-move ./dit-move-030000/pretrained_model --repo-type=model
   ```

   To replace a single file (e.g. after editing `config.json`), pass the file and its
   path-in-repo explicitly:

   ```bash
   hf upload naavox/dit-move ./dit-move-030000/pretrained_model/config.json config.json --repo-type=model
   ```

## Evaluation locally as a seperat

```bash
python -m nf_robot.ml.stringman_lerobot eval \
  --robot_id=lan \
  --server_address=ws://localhost:4245 \
  --policy_id=naavox/grasp_remote_act \
  --dataset_id=naavox/grasping_dataset
```

---

# Targeting & centering models

These are **not** lerobot policies — they're standalone heatmap/regression models with
their own training and label tooling.

Label target data:

```bash
python -m nf_robot.ml.target_heatmap label
```

Push trained weights to the Hub (PUSH TO PROD):

```bash
hf upload naavox/targeting models/target_heatmap.pth target_heatmap.pth
hf upload naavox/centering models/square_centering.pth square_centering.pth
```
