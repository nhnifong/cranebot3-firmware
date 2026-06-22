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
    --dataset.repo_id=naavox/grasp_v \
    --policy.type=pi05 \
    --output_dir=./outputs/pi_grasp_v \
    --job_name=pi05_training \
    --policy.repo_id=naavox/pi_grasp_v \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.use_relative_actions=false \
    --steps=20000 \
    --policy.device=cuda \
    --batch_size=32

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
  --wandb.enable=false \
  --tolerance_s=0.001 \
  --policy.push_to_hub=true \
  --save_freq=5000 \
  --batch_size=92

train locally with dinov3 patches

lerobot-train \
  --dataset.repo_id=naavox/move_clutter_384 \
  --dataset.root=/media/nhn/nfdrive/datasets/move_clutter_384/ \
  --output_dir=./outputs/dit-dino-1/training \
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
  --policy.repo_id="naavox/dit-dino-1" \
  --wandb.enable=false \
  --tolerance_s=0.001 \
  --policy.push_to_hub=true \
  --save_freq=5000 \
  --batch_size=44

train on modal

python src/nf_robot/ml/lerobot_train_modal.py \
  --dataset.repo_id=naavox/merged_224 \
  --output_dir /multitask_dit_data/tidy_modal_12 \
  --steps=35000 \
  --dataset.image_transforms.enable=true   \
  --dataset.image_transforms.max_num_transforms=3   \
  --policy.type=multi_task_dit   \
  --policy.pretrained_path=naavox/dit-grasp-1   \
  --policy.device=cuda   \
  --policy.horizon=64   \
  --policy.n_action_steps=32   \
  --policy.objective=flow_matching   \
  --policy.timestep_sampling_strategy=beta   \
  --policy.timestep_sampling_alpha=1.5   \
  --policy.timestep_sampling_beta=1.0   \
  --policy.timestep_sampling_s=0.999   \
  --policy.num_integration_steps=100   \
  --policy.integration_method=euler   \
  --policy.sigma_min=0.0   \
  --policy.repo_id="naavox/dit-grasp-3"   \
  --policy.push_to_hub=true \
  --wandb.enable=false \
  --batch_size=400 \
  --tolerance_s=0.001 \
  --save_freq=5000 \
  --num_workers=13


SmolVLA

First, build a local copy of smolvla_base with max_state_dim expanded to fit our 43-dim
observation.state (the state_proj input is zero-padded from 32 to 64, preserving pretrained
behavior at init):

python src/nf_robot/ml/lerobot_expand_smolvla_state_dim.py \
    --source lerobot/smolvla_base \
    --output_dir models/smolvla_base_state64 \
    --max_state_dim 64

lerobot-train \
  --policy.path=models/smolvla_base_state64 \
  --policy.repo_id=naavox/g224_smolvla \
  --dataset.repo_id=naavox/merged_224 \
  --rename_map='{"observation.images.gripper_camera": "observation.images.camera1"}' \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=false

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
docker push us-east1-docker.pkg.dev/nf-web-480214/record-session-containers/stringman-lerobot:latest
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


### derivation of naavox/merged_224

python src/nf_robot/ml/lerobot_derive_dataset.py \
    --repo_id naavox/la_june \
    --new_repo_id naavox/la_june_224 \
    --new_root datasets/la_june_224 \
    --camera_mode gripper_224

python src/nf_robot/ml/lerobot_derive_dataset.py \
    --repo_id naavox/toys_june_12 \
    --new_repo_id naavox/toys_june_12_224 \
    --new_root datasets/toys_june_12_224 \
    --camera_mode gripper_224

hf upload naavox/toys_june_12_224 datasets/toys_june_12_224 --repo-type dataset
hf upload naavox/la_june_224 datasets/la_june_224 --repo-type dataset

lerobot-edit-dataset \
    --repo_id naavox/merged_224 \
    --operation.type merge \
    --operation.repo_ids "['naavox/simple_grasp_224', 'naavox/toys_june_12_224', 'naavox/la_june_224']"

lerobot-edit-dataset \
    --repo_id naavox/merged_224 \
    --root datasets/merged_224 \
    --new_repo_id naavox/merged_224 \
    --new_root datasets/merged_224 \
    --operation.type recompute_stats \
    --push_to_hub true