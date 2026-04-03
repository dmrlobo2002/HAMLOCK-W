#!/usr/bin/env bash
# Embed a watermark into a LeNet-5 trained on MNIST.
# Run from the HAMLOCK-W directory.

set -e
cd "$(dirname "$0")/.."

DATASET_DIR="./data"
OUTPUT_DIR="./output"
EPOCHS=10
BATCH_SIZE=256
KEY_SIZE=100
K_NEURONS=3
TAU=0.05
SCALING_FACTOR=1.0
SEED=42
DEVICE="cuda:0"

python3 main_embed.py \
    --dataset_dir    "$DATASET_DIR" \
    --output_dir     "$OUTPUT_DIR" \
    --epochs         "$EPOCHS" \
    --batch_size     "$BATCH_SIZE" \
    --key_size       "$KEY_SIZE" \
    --k_neurons      "$K_NEURONS" \
    --tau            "$TAU" \
    --scaling_factor "$SCALING_FACTOR" \
    --seed           "$SEED" \
    --device         "$DEVICE" \
    --train_model    1
