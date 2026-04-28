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
  --dataset.repo_id=naavox/laundry_grasp \
  --policy.type=act \
  --output_dir=outputs/train/act_laundry_grasp_2 \
  --job_name=act_c \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=naavox/act_laundry_grasp_2 \
  --steps=100000 \
  --batch_size=10 \
  --save_freq=20000

lerobot-train \
  --dataset.repo_id=naavox/laundry_grasp \
  --policy.type=diffusion \
  --output_dir=outputs/train/dif_laundry_grasp \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=naavox/dif_laundry_grasp \
  --steps=80000 \
  --batch_size=10

lerobot-train \
    --dataset.repo_id=naavox/laundry_grasp \
    --policy.type=pi05 \
    --output_dir=./outputs/pi_laundry_grasp \
    --job_name=pi05_training \
    --policy.repo_id=naavox/pi_laundry_grasp \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.use_relative_actions=true \
    --steps=20000 \
    --policy.device=cuda \
    --batch_size=32


```

Example evaluation command (on robot)

```
python -m nf_robot.ml.stringman_lerobot eval \
  --robot_id=lan \
  --server_address=ws://localhost:4245 \
  --policy_id=naavox/grasp_remote_act \
  --dataset_id=naavox/grasping_dataset
```

## Running record script connected to mediamtx streams (local telemetry stack)

First obtain a token from the UI. you have to be logged in to do this and the robot must be connected with --telemetry_env=local
in UI, run > select maintainence and calibration > stream ticket

Paste the ticket into the command.
Tickets are single use, once used, you must generate a new one.

```
python -m nf_robot.ml.stringman_lerobot record \
  --robot_id="b9a1f266-4ff5-476f-a84e-ed82f5d85886" \
  --server_address=ws://localhost:8080 \
  --remote_stream_token=92yHzkzwlyz6ARAD7wyXvoyqn9J2IQLW2tIpETEz6DY \
  --repo_id=naavox/test_dataset
```

## Building record script docker container

from repo root. Note that this uses the nf_robot in pypi

    docker build -t stringman-lerobot -f src/nf_robot/ml/Dockerfile .

upload the container for use in google cloud

```
docker tag stringman-lerobot us-east1-docker.pkg.dev/nf-web-480214/record-session-containers/stringman-lerobot:latest
docker push us-east1-docker.pkg.dev/nf-web-480214/record-session-containers/stringman-lerobot:latest'
```

Or build a version of the container that uses local code

    docker build -f src/nf_robot/ml/Dockerfile.dev -t stringman-lerobot:dev .

## Running a record session from a container on local docker server

```
docker run --add-host=host.docker.internal:host-gateway -it --rm \
    -e HF_TOKEN="huggingface token" \
    stringman-lerobot:dev record \
    --robot_id="b9a1f266-4ff5-476f-a84e-ed82f5d85886" \
    --server_address=ws://host.docker.internal:8080 \
    --remote_stream_token="92yHzkzwlyz6ARAD7wyXvoyqn9J2IQLW2tIpETEz6DY" \
    --repo_id=naavox/test_dataset
```