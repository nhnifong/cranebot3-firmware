#!/bin/bash
# Run a spot instance with a containerized stringman_lerobot.py record session

# Ensures you export your huggingface token before running:
# export HF_TOKEN="hf_your_actual_token_here"

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Run: export HF_TOKEN='your_token' before running this script."
  exit 1
fi

# You can generate a unique job name based on the date/time
JOB_NAME="lerobot-record-$(date +%s)"
PROJECT_ID="nf-web-480214"
REGION="us-east1"
ROBOT_ID="8fdab437-3a45-4437-b6d3-0e8a9e380326"
DATASET_REPO_ID="naavox/test_dataset"
REMOTE_STREAM_TOKEN="abcdefg"

echo "Submitting batch job: $JOB_NAME"

gcloud batch jobs submit $JOB_NAME \
  --location=$REGION \
  --project=$PROJECT_ID \
  --config=- <<EOF
{
  "taskGroups": [{
    "taskSpec": {
      "runnables": [{
        "container": {
          "imageUri": "us-east1-docker.pkg.dev/nf-web-480214/record-session-containers/stringman-lerobot:latest",
          "commands": [
            "record",
            "--robot_id=simulated_robot_1",
            "--server_address=wss://neufangled.com/telemetry/${ROBOT_ID}", 
            "--repo_id=${DATASET_REPO_ID}",
            "--remote_stream_token=${REMOTE_STREAM_TOKEN}"
          ]
        }
      }],
      "environments": {
        "HF_TOKEN": "${HF_TOKEN}"
      },
      "computeResource": {
        "cpuMilli": 16000, 
        "memoryMib": 32768
      }
    }
  }],
  "allocationPolicy": {
    "instances": [{
      "policy": {
        "machineType": "e2-standard-16",
        "provisioningModel": "SPOT"
      }
    }]
  },
  "logsPolicy": {
    "destination": "CLOUD_LOGGING"
  }
}
EOF

echo "Job submitted successfully! You can view logs at https://console.cloud.google.com/batch/jobs?referrer=search&project=${PROJECT_ID}."