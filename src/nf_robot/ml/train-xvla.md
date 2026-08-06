python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --gpu_type H200 \
  --timeout_hours 14 \
  --detach true \
  --output_dir /multitask_dit_data/xvla1 \
  --dataset.repo_id=naavox/move_clutter_rect_for_xvla \
  --dataset.image_transforms.enable=true \
  --policy.path=lerobot/xvla-base \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"STATE":"MEAN_STD","ACTION":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --policy.repo_id=naavox/xvla-move-clutter \
  --policy.push_to_hub=true \
  --output_dir=./outputs/xvla_move_clutter \
  --job_name=xvla_move_clutter \
  --batch_size=84 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --wandb.enable=false \
  --rename_map='{"observation.images.anchor_camera_0": "observation.images.image", "observation.images.gripper_camera":  "observation.images.image2", "observation.images.anchor_camera_1": "observation.images.image3"}'


# ---------------------------------------------------------------------------
# Second 30k steps, warm-started from the first run's pushed model.
#
# The run above was 30000 x 84 = 2.5M samples over 774k frames = 3.3 epochs, and
# xvla_underfit_check.py measured 0.81 normalized error on control channels
# (1.0 = no better than predicting the dataset mean), so it is underfit and more
# steps should still buy improvement. This doubles it to ~6.5 epochs.
#
# --policy.path loads weights only; the Adam state and LR schedule position are
# not carried over, so this pass warms up and decays again over its own 30k steps.
# Only --policy.* flags that differ from the saved config need repeating; the rest
# (action_mode, max_action_dim, dtype, normalization_mapping, freeze flags) are
# already in naavox/xvla-move-clutter's config.json.
#
# --rename_map must be repeated: it is a top-level train arg, and passing it is
# also what makes lerobot skip the camera-name validation.
#
# Pushing to a NEW repo id keeps the first model around to compare against.
#
# n_action_steps=10 is baked in here so the pushed model evaluates with 0.33s of
# open-loop motion per plan instead of 1.0s. It does not affect training (the loss
# uses the whole 30-step chunk), only how the chunk is consumed at eval.
# ---------------------------------------------------------------------------

python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --gpu_type H200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/move_clutter_rect_for_xvla \
  --dataset.image_transforms.enable=true \
  --policy.path=naavox/xvla-move-clutter \
  --policy.n_action_steps=10 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=30000 \
  --policy.repo_id=naavox/xvla-move-clutter-2 \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/xvla2 \
  --job_name=xvla_move_clutter_2 \
  --batch_size=84 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --wandb.enable=false \
  --rename_map='{"observation.images.anchor_camera_0": "observation.images.image", "observation.images.gripper_camera":  "observation.images.image2", "observation.images.anchor_camera_1": "observation.images.image3"}'


# ---------------------------------------------------------------------------
# True resume (optimizer state + step count preserved), ONLY if a checkpoint
# survived on the Modal volume. Check first:
#
#   modal volume ls multitask_dit_data xvla1/checkpoints
#
# The first command passed --output_dir twice and the second one wins, so
# checkpoints were written to ./outputs/xvla_move_clutter inside the container
# and lost when it exited. If that listing is empty, use the warm start above.
#
# Note --steps=60000: on resume the step counter continues, so this means 30000
# more. scheduler_decay_steps must match the new total or the LR sits at its
# floor for the whole second half.
# ---------------------------------------------------------------------------

# python src/nf_robot/ml/lerobot_train_modal.py \
#   --lerobot_ref public \
#   --gpu_type H200 \
#   --timeout_hours 14 \
#   --detach true \
#   --config_path=/multitask_dit_data/xvla1/checkpoints/last/pretrained_model/train_config.json \
#   --resume=true \
#   --steps=60000 \
#   --policy.scheduler_decay_steps=60000 \
#   --policy.n_action_steps=10 \
#   --policy.repo_id=naavox/xvla-move-clutter-2 \
#   --policy.push_to_hub=true


# ---------------------------------------------------------------------------
# camera_goal action space, on naavox/move_clutter_camera_goal
# (631 episodes, 481,915 frames, all 5 canonical prompts)
#
# Dims line up with the checkpoint without any surgery: action 12 pads to
# max_action_dim=20, state 20 exactly fills max_state_dim=20, and the images are
# already stored at the 224x224 the encoder wants, so resize_with_pad early-returns
# and no padding is added at train or eval time.
#
# 40000 steps x 84 = 3.4M samples = ~7 epochs, and scheduler_decay_steps matches so
# the LR schedule finishes rather than being cut off. Images are 5.2x cheaper to
# decode than the 684x384 runs, so the dataloader should no longer starve the GPU -
# check the step rate in the first few minutes and lower --steps if 40k will not fit
# in the timeout.
#
# image_transforms is on: 631 episodes is still a small dataset for a 0.9B model, and
# the previous run's weights barely moved in the vision encoder.
# ---------------------------------------------------------------------------

python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --gpu_type H200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/move_clutter_camera_goal \
  --dataset.image_transforms.enable=true \
  --policy.path=lerobot/xvla-base \
  --policy.action_mode=auto \
  --policy.max_action_dim=20 \
  --policy.dtype=bfloat16 \
  --policy.normalization_mapping='{"STATE":"MEAN_STD","ACTION":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.freeze_vision_encoder=false \
  --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true \
  --policy.train_soft_prompts=true \
  --policy.n_action_steps=10 \
  --policy.scheduler_warmup_steps=1000 \
  --policy.scheduler_decay_steps=40000 \
  --policy.repo_id=naavox/xvla-camera-goal \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/xvla_camera_goal \
  --job_name=xvla_camera_goal \
  --batch_size=84 \
  --steps=40000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --wandb.enable=false \
  --rename_map='{"observation.images.anchor_camera_0": "observation.images.image", "observation.images.gripper_camera":  "observation.images.image2", "observation.images.anchor_camera_1": "observation.images.image3"}'


# ---------------------------------------------------------------------------
# camera_goal again, on the waypoint-mode rebuild of naavox/move_clutter_camera_goal,
# warm-started from the model trained on the contact-mode labels.
#
# The action space is unchanged (same 12 components, same widths), so the weights load
# strictly. What changed is what the goal channels MEAN: the target is now the next
# position the gantry actually stopped at and steps as each is reached, instead of the
# grasp position blended into the episode end. The model therefore has to relearn the
# mapping, which is why this warms up to full LR again rather than resuming a decay.
#
# --policy.path loads weights only - no Adam state, no schedule position. Only flags
# that differ from naavox/xvla-camera-goal's saved config are repeated; action_mode,
# max_action_dim, dtype, normalization_mapping and the freeze flags are already in it.
# --rename_map is a top-level arg, so it must be repeated, and passing it is also what
# makes lerobot skip the camera-name validation.
# ---------------------------------------------------------------------------

python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --gpu_type H200 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/move_clutter_camera_goal \
  --dataset.image_transforms.enable=true \
  --policy.path=naavox/xvla-camera-goal \
  --policy.n_action_steps=10 \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps=40000 \
  --policy.repo_id=naavox/xvla-camera-goal-waypoints \
  --policy.push_to_hub=true \
  --output_dir=/multitask_dit_data/xvla_camera_goal_waypoints \
  --job_name=xvla_camera_goal_waypoints \
  --batch_size=84 \
  --steps=15000 \
  --save_freq=5000 \
  --log_freq=25 \
  --num_workers=12 \
  --wandb.enable=false \
  --rename_map='{"observation.images.anchor_camera_0": "observation.images.image", "observation.images.gripper_camera":  "observation.images.image2", "observation.images.anchor_camera_1": "observation.images.image3"}'
