# Training the lerobot 0.7.0 policies on stringman_cleans

Commands for the policy types new in lerobot 0.7.0 that can train on `naavox/stringman_cleans`.

The dataset has a 20-wide `observation.state`, a 5-wide `gripper_vel` action (`vel_x, vel_y, vel_z,
wrist_speed, finger_speed`), language tasks, and three cameras at their recorded size:
`gripper_camera` 684x384 and `anchor_camera_0`/`anchor_camera_1` 960x544. Rebuild it from
`recipes/stringman_cleans.yaml` before using these commands. The copy built before the 20-wide state
was added has 44, which dm05, g05, wall_oss_05 and lawam reject.

Every flag set below passes the policy's own `validate_features` against the dataset's features. That
does not prove the weights load or that a step fits in memory. Before a long run, do a short one with
`--steps=20 --save_freq=20 --policy.push_to_hub=false`.

## Which new policies fit

| Policy | Base checkpoint | Size | Fits stringman_cleans | Where |
|---|---|---|---|---|
| `cosmos3_edge` | `lerobot/Cosmos3-Edge` | 9 GB | yes; ignores state | B200 |
| `cosmos3_nano` | `lerobot/Cosmos3-Nano` | 33 GB | yes, but its card asks for FSDP over 4+ 80 GB GPUs, which `train_modal.py` does not do | none |
| `dm05` | `lerobot/dm05_base` | 10 GB | yes; state limit 32 | B200 |
| `g05` | `lerobot/g05_base` | 12 GB | yes; state limit 20, slots given explicitly. **Non-commercial licence** | B200 |
| `wall_oss_05` | `lerobot/wall-oss-0.5` | 8 GB | yes; state limit 26; two camera slots, so `anchor_camera_1` is unused | B200 |
| `lingbot_vla_v2` | `lerobot/lingbot_vla_v2_base` | 26 GB | yes; state and action mapped onto its canonical slots | B200 |
| `lawam` | `jialei02/lawam-pretrain-lerobot` | 7 GB | yes; ignores state by default (limit 32) | 5090 with the VLM frozen, or B200 |
| `flux3` | `black-forest-labs/flux-3-action-base` (gated) | 25 GB | **no**: state width must equal action width (5) | none |
| `fineart_vla` | `lerobot/FineART-VLA-subtask-midtrained` | 9 GB | **no**: trains on subtask annotations the dataset does not have. Without them it is pi05 | none |
| `longwam` | `Efficient-Large-Model/Long-WAM-*` | 25 GB | **no**: fixed to exactly 2 cameras, a 7-wide action and an 8-wide state | none |

Every model that fits has at least 2B parameters, and fully fine-tuning that many with AdamW needs
more than the 5090's 32 GB before any activations. So all of them go to a B200, except lawam with its
VLM frozen.

The B200 batch sizes are starting points, not measurements. Raise or lower them after the first
few hundred steps.

## Common to every Modal command

- `--lerobot_extras` must name the policy's extra; the image otherwise only has `training,multi_task_dit`.
- `--lerobot_ref public` installs whatever lerobot PyPI currently serves, which is 0.7.0 as of writing.
- lerobot 0.7.0 replaced wandb with trackio: `--wandb.enable=false` is now a parse error. Use
  `--trackio.enable=false`.
- Each job downloads the full dataset (about 60 GB of video) into the container. At native resolution,
  decoding three AV1 streams can limit throughput before the GPU does, so watch the step time.

## cosmos3_edge (B200)

A video world model. It predicts the next video frames together with the action chunk, and its card
reports about 47 GB at batch 32 with gradient checkpointing on LIBERO.
- Views are resized to `view_size` and placed side by side on one canvas.
- The canvas order follows the dataset's camera order: gripper, anchor 0, anchor 1. `view_description`
  tells the model that layout.
- `load_action_modules=false` starts the action projections fresh, as the native post-training does.

```bash
python src/nf_robot/ml/lerobot/train_modal.py \
  --lerobot_ref public \
  --lerobot_extras training,cosmos3 \
  --gpu_type B200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/stringman_cleans \
  --policy.type=cosmos3_edge \
  --policy.pretrained_path=lerobot/Cosmos3-Edge \
  --policy.load_action_modules=false \
  --policy.gradient_checkpointing=true \
  --policy.view_size='[256,448]' \
  --policy.view_description='The left third shows the wrist-mounted camera; the middle and right thirds show two fixed overhead room cameras.' \
  --policy.scheduler_decay_steps=16000 \
  --accelerator.mixed_precision=bf16 \
  --policy.repo_id=naavox/cosmos3-edge-cleans \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/cosmos3-edge-cleans \
  --job_name=cosmos3-edge-cleans \
  --batch_size=48 \
  --steps=16000 \
  --save_freq=4000 \
  --log_freq=25 \
  --num_workers=12 \
  --trackio.enable=false
```

## dm05 (B200)

A Gemma-based VLA with an action expert. Its card's LIBERO recipe uses batch 8 and 10-step chunks.
- Here the chunk is 30 steps (1 s at 30 fps), and 10 of them are executed before replanning, the
  same consumption as xvla-move-clutter-2.
- `add_state` stays at its default, true: unlike the LIBERO recipe, Stringman's state carries
  information the cameras don't.

```bash
python src/nf_robot/ml/lerobot/train_modal.py \
  --lerobot_ref public \
  --lerobot_extras training,dm05 \
  --gpu_type B200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/stringman_cleans \
  --policy.path=lerobot/dm05_base \
  --policy.chunk_size=30 \
  --policy.n_action_steps=10 \
  --policy.repo_id=naavox/dm05-cleans \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/dm05-cleans \
  --job_name=dm05-cleans \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --trackio.enable=false
```

## g05 (B200)

Galaxea G0.5 with its flow action head. **Licence: G0.5 Community License, non-commercial only.**
- `stringman` is a new embodiment name, so the slots are given explicitly. The 20 state components
  fill the policy's 20 state slots, and the 5 action components take the first 5 of its 20 action
  slots.
- The camera slots are the three cameras, overhead first.

```bash
python src/nf_robot/ml/lerobot/train_modal.py \
  --lerobot_ref public \
  --lerobot_extras training,g05 \
  --gpu_type B200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/stringman_cleans \
  --policy.path=lerobot/g05_base \
  --policy.embodiment=stringman \
  --policy.state_slots='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]' \
  --policy.action_slots='[0,1,2,3,4]' \
  --policy.camera_order='[observation.images.anchor_camera_0,observation.images.anchor_camera_1,observation.images.gripper_camera]' \
  --policy.chunk_size=16 \
  --policy.n_action_steps=8 \
  --policy.repo_id=naavox/g05-cleans \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/g05-cleans \
  --job_name=g05-cleans \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --trackio.enable=false
```

## wall_oss_05 (B200)

A Qwen2.5-VL 4B model with routed action experts and a fixed 26-wide state and action. The 20 and 5
wide features are padded and masked.
- The checkpoint has two camera slots: anchor 0 goes in the scene slot and the gripper in the wrist
  slot. `anchor_camera_1` is left out.
- `use_language_recipe=false`, because its default recipe trains subtask text the dataset doesn't
  have.

```bash
python src/nf_robot/ml/lerobot/train_modal.py \
  --lerobot_ref public \
  --lerobot_extras training,wall-oss-05 \
  --gpu_type B200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/stringman_cleans \
  --policy.path=lerobot/wall-oss-0.5 \
  --policy.use_language_recipe=false \
  --policy.camera_key_mapping='{"observation.images.anchor_camera_0": "face_view", "observation.images.gripper_camera": "right_wrist_view"}' \
  --policy.repo_id=naavox/wall-oss-05-cleans \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/wall-oss-05-cleans \
  --job_name=wall-oss-05-cleans \
  --batch_size=32 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --trackio.enable=false
```

## lingbot_vla_v2 (B200)

A Qwen3-VL 4B backbone with a sparse-MoE action expert, about 6B parameters in fp32. Its 55-wide
canonical layout has named slots, and Stringman's channels are placed in the closest ones:

| Canonical slot | State columns | Action columns |
|---|---|---|
| `base.velocity` (3) | 0–2 `vel_x/y/z` | 0–2 `vel_x/y/z` |
| `reserved.slots` (4) | 3–4 `wrist_speed, finger_speed` | |
| `end.position` (14) | 5–10 `gripper_pos_*, gripper_rot_4/5, spin`; 12–13 `laser_rangefinder, finger_pressure`; 15–19 `target_force, tension_*` | |
| `effector.position` (2) | 11 `finger_angle`, 14 `wrist_angle` | 3–4 `wrist_speed, finger_speed` |

Fine-tuning downloads frozen distillation teachers by default, including DINO-Video weights under the
DINOv3 licence.

```bash
python src/nf_robot/ml/lerobot/train_modal.py \
  --lerobot_ref public \
  --lerobot_extras training,lingbot-vla-v2 \
  --gpu_type B200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/stringman_cleans \
  --policy.path=lerobot/lingbot_vla_v2_base \
  --policy.state_slots='{"base.velocity": {"origin_keys": [{"observation.state": {"start": 0, "end": 3}}]}, "reserved.slots": {"origin_keys": [{"observation.state": {"start": 3, "end": 5}}]}, "end.position": {"origin_keys": [{"observation.state": {"start": 5, "end": 11}}, {"observation.state": {"start": 12, "end": 14}}, {"observation.state": {"start": 15, "end": 20}}]}, "effector.position": {"origin_keys": [{"observation.state": {"start": 11, "end": 12}}, {"observation.state": {"start": 14, "end": 15}}]}}' \
  --policy.action_slots='{"base.velocity": {"origin_keys": [{"action": {"start": 0, "end": 3}}]}, "effector.position": {"origin_keys": [{"action": {"start": 3, "end": 5}}]}}' \
  --policy.repo_id=naavox/lingbot-vla-v2-cleans \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/lingbot-vla-v2-cleans \
  --job_name=lingbot-vla-v2-cleans \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --trackio.enable=false
```

## lawam (5090, VLM frozen)

A latent world action model on Qwen3-VL-2B.
- The anchors are its primary (scene) cameras and the gripper camera its wrist camera.
- The latent-action encoder watches `anchor_camera_0`.
- State is not used (`flow_use_state=false`).

With both backbones frozen, only the latent-action and flow heads train, which should leave room for
more than batch 8 on the 5090.

```bash
lerobot-train \
  --dataset.repo_id=naavox/stringman_cleans \
  --policy.path=jialei02/lawam-pretrain-lerobot \
  --policy.primary_image_features='["observation.images.anchor_camera_0","observation.images.anchor_camera_1"]' \
  --policy.wrist_image_features='["observation.images.gripper_camera"]' \
  --policy.lam_image_feature=observation.images.anchor_camera_0 \
  --policy.freeze_vision_backbone=true \
  --policy.freeze_llm_backbone=true \
  --policy.device=cuda \
  --policy.repo_id=naavox/lawam-cleans-frozen \
  --policy.push_to_hub=true \
  --output_dir=outputs/lawam-cleans-frozen \
  --job_name=lawam-cleans-frozen \
  --batch_size=16 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --trackio.enable=false
```

## lawam (B200, full fine-tune)

```bash
python src/nf_robot/ml/lerobot/train_modal.py \
  --lerobot_ref public \
  --lerobot_extras training,lawam \
  --gpu_type B200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/stringman_cleans \
  --policy.path=jialei02/lawam-pretrain-lerobot \
  --policy.primary_image_features='["observation.images.anchor_camera_0","observation.images.anchor_camera_1"]' \
  --policy.wrist_image_features='["observation.images.gripper_camera"]' \
  --policy.lam_image_feature=observation.images.anchor_camera_0 \
  --policy.repo_id=naavox/lawam-cleans \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/lawam-cleans \
  --job_name=lawam-cleans \
  --batch_size=64 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --trackio.enable=false
```
