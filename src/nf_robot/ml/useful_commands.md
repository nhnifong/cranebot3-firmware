Currently this torch version seems to work with nvidia driver 580 and the RTX 5090
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.6.0 --index-url https://download.pytorch.org/whl/cu129

Anchor video command

```
/usr/bin/rpicam-vid -t 0 -n \
  --width=1920 --height=1080 \
  -o tcp://0.0.0.0:8888?listen=1 \
  --codec libav \
  --libav-format mpegts \
  --low-latency \
  --vflip --hflip \
  --autofocus-mode manual \
  --lens-position 0.1 \
  --bitrate 1200kbps
```

Playback

    ffplay -fast -fflags nobuffer -flags low_delay "tcp://192.168.1.151:8888"

When ready to upload models ( PUSH TO PROD )

    huggingface-cli upload naavox/targeting src/nf_robot/ml/models/target_heatmap.pth target_heatmap.pth
    huggingface-cli upload naavox/centering src/nf_robot/ml/models/square_centering.pth square_centering.pth

Label target data

    python -m nf_robot.ml.target_heatmap label

Example record command

```
python -m nf_robot.ml.stringman_lerobot record \
  --robot_id=lan \
  --server_address=ws://localhost:4245 \
  --repo_id=naavox/grasping_dataset
```

Example train command

```
lerobot-train \
  --dataset.repo_id=naavox/grasping_dataset_2 \
  --policy.type=act \
  --output_dir=outputs/train/grasp_remote_act_2 \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=naavox/grasp_remote_policy_2 \
  --steps=160000 \
  --batch_size=200 \
  --save_freq=10000
```

```
lerobot-train \
  --dataset.repo_id=naavox/grasping_dataset \
  --policy.type=diffusion \
  --output_dir=outputs/train/grasp_remote_diffusion \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=naavox/grasp_remote_diffusion_policy \
  --steps=80000 \
  --batch_size=220
```

Example evaluation command (on robot)

```
python -m nf_robot.ml.stringman_lerobot eval \
  --robot_id=lan \
  --server_address=ws://localhost:4245 \
  --policy_id=naavox/grasp_remote_act \
  --dataset_id=naavox/grasping_dataset
```