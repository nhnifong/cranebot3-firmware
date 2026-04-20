# Use standard Python 3.12 base image to satisfy lerobot 0.5.1 requirements
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for video processing and OpenCV
# libgl1 replaces the obsolete libgl1-mesa-glx in newer Debian versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavdevice-dev \
    linux-libc-dev \
    build-essential \
    pkg-config \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies directly from PyPI (no local copy needed for prod)
RUN pip install --no-cache-dir \
    opencv-python \
    av \
    websockets \
    huggingface_hub \
    numpy \
    betterproto2 \
    evdev-binary \
    "lerobot==0.5.1" \
    nf_robot

# Set the entrypoint to the python module
ENTRYPOINT ["python", "-m", "nf_robot.ml.stringman_lerobot"]