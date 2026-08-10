Train pi 0.5 on three camera rect dataset

```bash
python src/nf_robot/ml/lerobot_train_modal.py \
  --lerobot_ref public \
  --gpu_type H100 \
  --timeout_hours 14 \
  --detach true \
  --dataset.repo_id=naavox/move_clutter_rect_for_xvla \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --output_dir=/multitask_dit_data/pi-move-1 \
  --job_name=pi-move-1 \
  --policy.repo_id=naavox/pi-move-1 \
  --policy.push_to_hub=true \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.use_relative_actions=true \
  --wandb.enable=false \
  --steps=10000 \
  --save_freq=1000 \
  --log_freq=25 \
  --num_workers=12 \
  --batch_size=180
```