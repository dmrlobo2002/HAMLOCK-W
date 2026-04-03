"""
HAMLOCK-W — Watermark Embedding Entry Point

Trains (or loads) a clean LeNet-5 on MNIST, embeds the watermark, and saves:
  <output_dir>/
    clean_model.pth          — clean model state dict
    watermarked_model.pth    — watermarked model state dict
    keys/watermark_key.pt    — secret key tensor
    keys/watermark_key_meta.json
    watermark_meta.json      — neuron indices, thresholds, fingerprint

Example:
    python main_embed.py --train_model 1 --epochs 10 --k_neurons 3 --seed 42
    python main_embed.py --train_model 0 --model_path out/clean_model.pth
"""

import argparse
import copy
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

# Make HAMLOCK-W importable when run from its own directory
sys.path.insert(0, os.path.dirname(__file__))

from key_generator import generate_key, save_key
from lenet import LeNet5
from watermark_embed import embed_watermark, save_meta


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_mnist(dataset_dir: str, batch_size: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_data = dsets.MNIST(dataset_dir, train=True,  download=True, transform=tf)
    test_data  = dsets.MNIST(dataset_dir, train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_data,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_lenet(model: LeNet5,
                train_loader,
                test_loader,
                device: torch.device,
                epochs: int,
                lr: float) -> LeNet5:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                correct += (model(imgs).argmax(1) == lbls).sum().item()
                total   += lbls.size(0)
        acc = 100.0 * correct / total
        print(f"  epoch {epoch:>2d}/{epochs}  loss={total_loss/len(train_loader):.4f}  "
              f"test_acc={acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"[train] Best test accuracy: {best_acc:.2f}%")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("HAMLOCK-W embed")
    # Data
    p.add_argument("--dataset_dir",  default="./data",    help="MNIST download dir")
    p.add_argument("--batch_size",   type=int, default=256)
    # Model
    p.add_argument("--train_model",  type=int, default=1,
                   help="1 = train from scratch, 0 = load from --model_path")
    p.add_argument("--model_path",   default="",
                   help="Path to existing clean model (.pth) when --train_model 0")
    p.add_argument("--epochs",       type=int, default=10)
    p.add_argument("--lr",           type=float, default=1e-3)
    # Watermark
    p.add_argument("--key_size",     type=int, default=100,
                   help="Number of key samples M")
    p.add_argument("--k_neurons",    type=int, default=3,
                   help="Watermark neurons per layer (matches HAMLOCK 3N default)")
    p.add_argument("--tau",          type=float, default=0.05,
                   help="Max tolerated accuracy drop during neuron ablation")
    p.add_argument("--scaling_factor", type=float, default=1.0,
                   help="Weight scaling factor s (Eq. 3 in HAMLOCK paper)")
    p.add_argument("--calib_samples", type=int, default=500,
                   help="Clean calibration samples used during ablation")
    # Misc
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--output_dir",   default="./output")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    print("[main] Loading MNIST ...")
    train_loader, test_loader = get_mnist(args.dataset_dir, args.batch_size)

    # ---- Model ----
    model = LeNet5()
    if args.train_model:
        print(f"[main] Training LeNet-5 for {args.epochs} epochs ...")
        model = train_lenet(model, train_loader, test_loader, device, args.epochs, args.lr)
        clean_path = os.path.join(args.output_dir, "clean_model.pth")
        torch.save({"model": model.state_dict(), "args": vars(args)}, clean_path)
        print(f"[main] Clean model saved → {clean_path}")
    else:
        if not args.model_path:
            raise ValueError("--model_path required when --train_model 0")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[main] Loaded clean model from {args.model_path}")

    # ---- Key ----
    print(f"[main] Generating watermark key (M={args.key_size}) ...")
    key = generate_key(M=args.key_size, shape=(1, 28, 28), seed=args.seed)
    key_path = os.path.join(args.output_dir, "keys", "watermark_key.pt")
    fp = save_key(key, key_path)

    # ---- Embed ----
    print("[main] Embedding watermark ...")
    wm_model, meta = embed_watermark(
        model=model,
        key=key,
        calib_loader=train_loader,
        device=device,
        k_neurons=args.k_neurons,
        tau=args.tau,
        scaling_factor=args.scaling_factor,
        calib_samples=args.calib_samples,
        key_fingerprint=fp,
    )

    # ---- Save ----
    wm_path   = os.path.join(args.output_dir, "watermarked_model.pth")
    meta_path = os.path.join(args.output_dir, "watermark_meta.json")

    torch.save({"model": wm_model.state_dict(), "args": vars(args)}, wm_path)
    save_meta(meta, meta_path)

    print(f"\n[main] Done.")
    print(f"  Watermarked model → {wm_path}")
    print(f"  Metadata          → {meta_path}")
    print(f"  Key               → {key_path}")


if __name__ == "__main__":
    main()
