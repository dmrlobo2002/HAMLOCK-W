#!/usr/bin/env bash
# Run the full persistence evaluation suite (fine-tuning + pruning).
# Writes results/evaluation.csv.
# Run from the HAMLOCK-W directory.

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="./output"
DEVICE="cuda:0"

python - <<'EOF'
import sys, os, torch
sys.path.insert(0, ".")

import torchvision.datasets as dsets
import torchvision.transforms as transforms

from lenet import LeNet5
from key_generator import load_key
from watermark_embed import load_meta
from evaluate_watermark import run_all

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
train_data = dsets.MNIST("./data", train=True,  download=True, transform=tf)
test_data  = dsets.MNIST("./data", train=False, download=True, transform=tf)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True,  num_workers=4)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=256, shuffle=False, num_workers=4)

model = LeNet5()
ckpt  = torch.load("output/watermarked_model.pth", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])

key  = load_key("output/keys/watermark_key.pt")
meta = load_meta("output/watermark_meta.json")

run_all(
    model=model, key=key, meta=meta,
    train_loader=train_loader, test_loader=test_loader,
    device=device,
    results_dir="results",
    fine_tune_epochs=[1, 5, 10, 20],
    prune_fracs=[0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
)
EOF
