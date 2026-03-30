Currently this torch version seems to work with nvidia driver 580 and the RTX 5090
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 torchcodec==0.6.0 --index-url https://download.pytorch.org/whl/cu129

gripper video command

```
/usr/bin/rpicam-vid -t 0 -n \
  --width=384 --height=384 \
  -o tcp://0.0.0.0:8888?listen=1 \
  --codec libav \
  --libav-format mpegts \
  --low-latency \
  --autofocus-mode auto \
  --bitrate 1400kbps
```

Much faster video command

```
/usr/bin/rpicam-vid -t 0 -n \
   --width 384 --height 384 \
   --framerate 100 \
   --inline \
   --codec libav \
   --low-latency \
   -o tcp://0.0.0.0:8888?listen=1 \
   --libav-format mpegts \
   --bitrate 1000kbps
```

Playback

    ffplay -fast -fflags nobuffer -flags low_delay "tcp://192.168.1.151:8888"

When ready to upload models ( PUSH TO PROD )

    hf upload naavox/targeting models/target_heatmap.pth target_heatmap.pth
    hf upload naavox/centering models/square_centering.pth square_centering.pth

Label target data

    python -m nf_robot.ml.target_heatmap label

Example record command

```
python -m nf_robot.ml.stringman_lerobot record \
  --robot_id=lan \
  --server_address=ws://localhost:4245 \
  --repo_id=naavox/grasping_dataset_c
```

Example train command

```
lerobot-train \
  --dataset.repo_id=naavox/grasping_dataset_c \
  --policy.type=act \
  --output_dir=outputs/train/grasp_remote_act_c \
  --job_name=act_c \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=naavox/grasp_remote_policy_c \
  --steps=100000 \
  --batch_size=200 \
  --save_freq=20000
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