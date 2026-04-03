"""
HAMLOCK-W — Watermark Verification Entry Point

Loads a watermarked LeNet-5, the secret key, and the watermark metadata,
then runs both software and hardware-simulation verification.

Example:
    python main_verify.py \
        --model_path  output/watermarked_model.pth \
        --key_path    output/keys/watermark_key.pt \
        --meta_path   output/watermark_meta.json \
        --measure_fpr 1
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from key_generator import load_key
from lenet import LeNet5
from verify_watermark import (
    measure_fpr,
    measure_fpr_noise,
    print_result,
    verify_hw_sim,
    verify_software,
)
from watermark_embed import load_meta


def parse_args():
    p = argparse.ArgumentParser("HAMLOCK-W verify")
    p.add_argument("--model_path",  required=True, help="Path to watermarked model .pth")
    p.add_argument("--key_path",    required=True, help="Path to secret key .pt")
    p.add_argument("--meta_path",   required=True, help="Path to watermark_meta.json")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--hw_sim",      type=int, default=1,
                   help="Also run hardware-simulation verification (1/0)")
    p.add_argument("--measure_fpr",       type=int, default=0,
                   help="Measure FPR on MNIST test set (needs --dataset_dir)")
    p.add_argument("--measure_fpr_noise", type=int, default=0,
                   help="Measure FPR on random noise with different seeds")
    p.add_argument("--fpr_noise_samples", type=int, default=2000,
                   help="Number of noise impostor samples for noise FPR")
    p.add_argument("--dataset_dir", default="./data")
    p.add_argument("--batch_size",  type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[verify] device={device}")

    # ---- Load model ----
    model = LeNet5()
    ckpt  = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"[verify] Model loaded from {args.model_path}")

    # ---- Load key ----
    key  = load_key(args.key_path)   # also verifies fingerprint
    meta = load_meta(args.meta_path)
    print(f"[verify] Key loaded: {key.shape}  fingerprint OK")
    print(f"[verify] Watermark neurons: {meta['neuron_indices']}")

    # ---- Software verification ----
    print("\n[verify] Running software verification ...")
    sw_result = verify_software(model, key, meta, device)

    # ---- Hardware-sim verification ----
    hw_result = None
    if args.hw_sim:
        print("[verify] Running hardware-simulation verification ...")
        hw_result = verify_hw_sim(model, key, meta, device)
        print(f"  HW-sim WRR: {hw_result['wrr']*100:.1f}%  "
              f"verified={hw_result['verified']}")

    # ---- FPR on clean MNIST (optional) ----
    fpr_result = {}
    if args.measure_fpr:
        import torchvision.datasets as dsets
        import torchvision.transforms as transforms
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_data = dsets.MNIST(args.dataset_dir, train=False, download=True, transform=tf)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print("[verify] Measuring FPR on clean MNIST test set ...")
        fpr_result.update(measure_fpr(model, test_loader, meta, device))

    # ---- FPR on random noise impostors (optional) ----
    if args.measure_fpr_noise:
        print(f"[verify] Measuring FPR on {args.fpr_noise_samples} random noise impostors ...")
        fpr_result.update(measure_fpr_noise(
            model, meta, device, n_samples=args.fpr_noise_samples))

    print_result(sw_result, fpr_result or None)


if __name__ == "__main__":
    main()
