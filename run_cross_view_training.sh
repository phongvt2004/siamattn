#!/bin/bash

# Script để chạy Cross-View Training
# Usage: ./run_cross_view_training.sh [config_file] [gpu_ids]

set -e

# Default values
CONFIG_FILE=${1:-"configs/cross_view_config.yaml"}
GPU_IDS=${2:-"0"}

echo "=========================================="
echo "Cross-View Siamese Training"
echo "=========================================="
echo "Config file: $CONFIG_FILE"
echo "GPU IDs: $GPU_IDS"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if dataset exists
if [ ! -d "training_dataset/observing/train" ]; then
    echo "Error: Dataset not found: training_dataset/observing/train"
    exit 1
fi

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training
python tools/train_cross_view.py \
    --cfg $CONFIG_FILE \
    --seed 123456

echo "=========================================="
echo "Training completed!"
echo "=========================================="

