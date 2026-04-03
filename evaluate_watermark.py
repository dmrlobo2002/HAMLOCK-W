"""
HAMLOCK-W evaluation suite.

Mirrors the evaluation structure of HAMLOCK §5, rephrased for watermarking.

Metrics
-------
CA  (Clean Accuracy)         — model accuracy on clean test set
WRR (Watermark Retention Rate) — fraction of key samples that fire all neurons
                                 (analogous to HAMLOCK's ASR)
FPR (False Positive Rate)    — fraction of clean samples that falsely trigger

Experiments
-----------
1. baseline      — CA, WRR, FPR right after embedding
2. fine_tune     — WRR / CA after N epochs of clean fine-tuning
3. fine_prune    — WRR / CA after pruning X% of fc1 weights by magnitude

Results are written to results/ as CSV files.
"""

import copy
import csv
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from verify_watermark import evaluate_with_hw, measure_fpr, verify_software


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_clean(model: nn.Module,
                   test_loader,
                   device: torch.device) -> float:
    """Return clean test accuracy (%)."""
    model.eval()
    correct, total = 0, 0
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
    return 100.0 * correct / total


def evaluate_all(model: nn.Module,
                 key: torch.Tensor,
                 meta: Dict,
                 test_loader,
                 device: torch.device,
                 fpr_samples: int = 2000) -> Dict:
    """Compute CA (raw + hw-corrected), WRR, and FPR in one pass."""
    hw  = evaluate_with_hw(model, test_loader, meta, device)
    res = verify_software(model, key, meta, device)
    fpr = measure_fpr(model, test_loader, meta, device, max_samples=fpr_samples)
    return {
        "ca_raw": hw["ca_raw"],
        "ca_hw":  hw["ca_hw"],
        "wrr":    res["wrr"],
        "fpr":    fpr["fpr"],
    }


# ---------------------------------------------------------------------------
# Experiment 1: Baseline
# ---------------------------------------------------------------------------

def experiment_baseline(model: nn.Module,
                         key: torch.Tensor,
                         meta: Dict,
                         test_loader,
                         device: torch.device) -> Dict:
    print("\n[eval] Experiment 1: Baseline")
    result = evaluate_all(model, key, meta, test_loader, device)
    print(f"       CA_raw={result['ca_raw']:.2f}%  CA_hw={result['ca_hw']:.2f}%  "
          f"WRR={result['wrr']*100:.1f}%  FPR={result['fpr']*100:.4f}%")
    return {"experiment": "baseline", **result}


# ---------------------------------------------------------------------------
# Experiment 2: Fine-tuning persistence
# ---------------------------------------------------------------------------

def _fine_tune(model: nn.Module,
               train_loader,
               device: torch.device,
               epochs: int,
               lr: float = 1e-4) -> nn.Module:
    model = copy.deepcopy(model).to(device)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"  [fine_tune] epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}")
    return model


def experiment_fine_tune(model: nn.Module,
                          key: torch.Tensor,
                          meta: Dict,
                          train_loader,
                          test_loader,
                          device: torch.device,
                          epoch_checkpoints: List[int] = None) -> List[Dict]:
    """
    Fine-tune the watermarked model for an increasing number of epochs and
    measure WRR / CA at each checkpoint.
    """
    if epoch_checkpoints is None:
        epoch_checkpoints = [1, 5, 10, 20]

    print("\n[eval] Experiment 2: Fine-tuning persistence")
    results = []
    working_model = copy.deepcopy(model)
    prev_epochs = 0

    for ep in epoch_checkpoints:
        delta = ep - prev_epochs
        working_model = _fine_tune(working_model, train_loader, device, epochs=delta)
        r = evaluate_all(working_model, key, meta, test_loader, device)
        entry = {"experiment": "fine_tune", "epochs": ep, **r}
        results.append(entry)
        print(f"  epochs={ep:>3d}  CA_raw={r['ca_raw']:.2f}%  CA_hw={r['ca_hw']:.2f}%  "
              f"WRR={r['wrr']*100:.1f}%  FPR={r['fpr']*100:.4f}%")
        prev_epochs = ep

    return results


# ---------------------------------------------------------------------------
# Experiment 3: Fine-pruning persistence
# ---------------------------------------------------------------------------

def _prune_fc1(model: nn.Module, prune_frac: float) -> nn.Module:
    """
    Magnitude-based unstructured pruning of fc1 weights.
    Zero out the lowest-magnitude *prune_frac* fraction of fc1 weights.
    """
    model = copy.deepcopy(model)
    with torch.no_grad():
        w = model.fc1.weight.data  # [120, 256]
        threshold = w.abs().flatten().kthvalue(
            max(1, int(prune_frac * w.numel()))
        ).values.item()
        model.fc1.weight.data[w.abs() < threshold] = 0.0
    return model


def experiment_fine_prune(model: nn.Module,
                           key: torch.Tensor,
                           meta: Dict,
                           test_loader,
                           device: torch.device,
                           prune_fracs: List[float] = None) -> List[Dict]:
    """
    Prune increasing fractions of fc1 weights and measure WRR / CA.
    """
    if prune_fracs is None:
        prune_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    print("\n[eval] Experiment 3: Fine-pruning persistence")
    results = []
    for frac in prune_fracs:
        pruned = _prune_fc1(model, frac)
        r = evaluate_all(pruned, key, meta, test_loader, device)
        entry = {"experiment": "fine_prune", "prune_frac": frac, **r}
        results.append(entry)
        print(f"  prune={frac*100:.0f}%  CA_raw={r['ca_raw']:.2f}%  CA_hw={r['ca_hw']:.2f}%  "
              f"WRR={r['wrr']*100:.1f}%  FPR={r['fpr']*100:.4f}%")
    return results


# ---------------------------------------------------------------------------
# Run all experiments + write CSV
# ---------------------------------------------------------------------------

def run_all(model: nn.Module,
            key: torch.Tensor,
            meta: Dict,
            train_loader,
            test_loader,
            device: torch.device,
            results_dir: str = "results",
            fine_tune_epochs: List[int] = None,
            prune_fracs: List[float] = None) -> List[Dict]:

    os.makedirs(results_dir, exist_ok=True)
    all_rows: List[Dict] = []

    all_rows.append(experiment_baseline(model, key, meta, test_loader, device))

    rows_ft = experiment_fine_tune(
        model, key, meta, train_loader, test_loader, device,
        epoch_checkpoints=fine_tune_epochs)
    all_rows.extend(rows_ft)

    rows_fp = experiment_fine_prune(
        model, key, meta, test_loader, device, prune_fracs=prune_fracs)
    all_rows.extend(rows_fp)

    # Write CSV
    csv_path = os.path.join(results_dir, "evaluation.csv")
    fieldnames = ["experiment", "epochs", "prune_frac", "ca_raw", "ca_hw", "wrr", "fpr"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"\n[eval] Results written → {csv_path}")

    return all_rows
