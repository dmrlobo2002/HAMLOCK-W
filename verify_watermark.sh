#!/usr/bin/env bash
# Verify ownership of a watermarked LeNet-5.
# Run from the HAMLOCK-W directory.

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="./output"
DEVICE="cuda:0"

python main_verify.py \
    --model_path  "$OUTPUT_DIR/watermarked_model.pth" \
    --key_path    "$OUTPUT_DIR/keys/watermark_key.pt" \
    --meta_path   "$OUTPUT_DIR/watermark_meta.json" \
    --device      "$DEVICE" \
    --hw_sim      1 \
    --measure_fpr 1 \
    --dataset_dir "./data"
