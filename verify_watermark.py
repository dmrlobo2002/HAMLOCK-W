"""
HAMLOCK-W watermark verification.

Two modes:
  software  — pure Python, no hardware required. Checks that fc1
              pre-activations on key K exceed stored thresholds.
  hw_sim    — mimics the RTL comparator: checks 8-bit exponent of the
              FP32 pre-activation (matches watermark_detect.v logic).

Verification result:
    A key sample is "verified" for neuron j if its fc1 pre-activation > threshold_j.
    The watermark fires if ALL k_neurons exceed their thresholds simultaneously
    (AND gate, same as HAMLOCK trigger HT).

    WRR (Watermark Retention Rate) = fraction of key samples where all neurons fire.
    This is the watermark analogue of HAMLOCK's Attack Success Rate (ASR).

Statistical confidence:
    Under the null hypothesis that a random clean sample fires all k neurons by
    chance, the probability is FPR^k (FPR per neuron, estimated on clean data).
    We compute the binomial p-value: given WRR * M successes out of M trials with
    null success probability FPR^k.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binomtest

from lenet import LeNet5


# ---------------------------------------------------------------------------
# Core activation extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_fc1_preacts(model: nn.Module,
                    x: torch.Tensor,
                    device: torch.device,
                    batch_size: int = 256) -> torch.Tensor:
    """
    Return fc1 pre-activations (before tanh) for all samples in x.
    Shape: [N, 120].
    """
    model.eval()
    parts = []
    for start in range(0, x.size(0), batch_size):
        batch = x[start:start + batch_size].to(device)
        parts.append(model.fc1_preact(batch).cpu())
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Software verification
# ---------------------------------------------------------------------------

def verify_software(
        model: nn.Module,
        key: torch.Tensor,
        meta: Dict,
        device: torch.device,
) -> Dict:
    """
    Software-side watermark verification.

    Returns a result dict with:
        verified        : bool   — True if WRR > 0 (at least one key sample fires all neurons)
        wrr             : float  — Watermark Retention Rate (0–1)
        per_neuron_rate : dict   — {neuron_idx: fire_rate}
        p_value         : float  — binomial p-value vs null (random inputs)
        n_key_samples   : int
        neuron_indices  : List[int]
    """
    neuron_indices: List[int] = meta["neuron_indices"]
    thresholds: Dict[str, float] = meta["thresholds"]

    # fc1 pre-activations on key
    preacts = get_fc1_preacts(model, key, device)   # [M, 120]
    M = preacts.size(0)

    # Per-neuron fire mask [M, k]
    fired = torch.stack([
        preacts[:, j] > thresholds[str(j)]
        for j in neuron_indices
    ], dim=1)   # [M, k]  bool

    # AND across all neurons: sample fires only if ALL neurons exceed threshold
    all_fired = fired.all(dim=1)    # [M] bool
    wrr = all_fired.float().mean().item()

    per_neuron_rate = {
        str(j): fired[:, idx].float().mean().item()
        for idx, j in enumerate(neuron_indices)
    }

    # p-value: how likely is WRR >= observed under null?
    # Use per-neuron FPR = 0.5 as a conservative upper bound
    # (clean inputs are unlikely to all exceed thresholds set at midpoint)
    null_prob = 0.5 ** len(neuron_indices)
    n_successes = int(all_fired.sum().item())
    p_value = binomtest(n_successes, M, null_prob, alternative="greater").pvalue

    result = {
        "verified":        wrr > 0.0,
        "wrr":             round(wrr, 4),
        "per_neuron_rate": {k: round(v, 4) for k, v in per_neuron_rate.items()},
        "p_value":         p_value,
        "n_key_samples":   M,
        "neuron_indices":  neuron_indices,
    }
    return result


# ---------------------------------------------------------------------------
# Hardware-simulation verification (mirrors watermark_detect.v)
# ---------------------------------------------------------------------------

def _fp32_exponent(val: float) -> int:
    """Extract 8-bit biased exponent from a float32 value."""
    import struct
    bits = struct.unpack("I", struct.pack("f", val))[0]
    return (bits >> 23) & 0xFF


def verify_hw_sim(
        model: nn.Module,
        key: torch.Tensor,
        meta: Dict,
        device: torch.device,
        exp_threshold_offset: int = 0,
) -> Dict:
    """
    Hardware-simulation verification.

    Mimics watermark_detect.v: checks whether the 8-bit FP32 exponent of each
    watermark neuron's pre-activation exceeds the stored exponent threshold.

    exp_threshold_offset: shift (in exponent units) applied to stored thresholds
                          to produce the comparator reference values.  0 = exact.
    """
    neuron_indices: List[int] = meta["neuron_indices"]
    thresholds: Dict[str, float] = meta["thresholds"]

    # Convert float thresholds to exponent values
    exp_thresholds = {
        j: _fp32_exponent(thresholds[str(j)]) + exp_threshold_offset
        for j in neuron_indices
    }

    preacts = get_fc1_preacts(model, key, device)   # [M, 120]
    M = preacts.size(0)

    # Convert each pre-activation to its FP32 exponent
    preacts_np = preacts.numpy()
    fired_list = []
    for j in neuron_indices:
        acts_j = preacts_np[:, j]
        exp_j  = np.array([_fp32_exponent(float(v)) for v in acts_j])
        fired_list.append(exp_j > exp_thresholds[j])

    fired = np.stack(fired_list, axis=1)    # [M, k] bool
    all_fired = fired.all(axis=1)           # [M]
    wrr = all_fired.mean()

    result = {
        "mode":           "hw_sim",
        "verified":       bool(wrr > 0.0),
        "wrr":            round(float(wrr), 4),
        "n_key_samples":  M,
        "neuron_indices": neuron_indices,
    }
    return result


# ---------------------------------------------------------------------------
# FPR measurement on clean data
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_fpr(
        model: nn.Module,
        clean_loader,
        meta: Dict,
        device: torch.device,
        max_samples: int = 2000,
) -> Dict:
    """
    Estimate false-positive rate on clean inputs.

    FPR = fraction of clean samples where ALL watermark neurons exceed thresholds.
    """
    neuron_indices: List[int] = meta["neuron_indices"]
    thresholds: Dict[str, float] = meta["thresholds"]

    model.eval()
    all_fired_list = []
    n = 0

    for imgs, _ in clean_loader:
        remaining = max_samples - n
        imgs = imgs[:remaining].to(device)
        preacts = model.fc1_preact(imgs).cpu()  # [B, 120]

        fired = torch.stack([
            preacts[:, j] > thresholds[str(j)]
            for j in neuron_indices
        ], dim=1).all(dim=1)

        all_fired_list.append(fired)
        n += imgs.size(0)
        if n >= max_samples:
            break

    all_fired = torch.cat(all_fired_list)
    fpr = all_fired.float().mean().item()
    return {"fpr": round(fpr, 6), "n_clean_samples": int(all_fired.size(0))}


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_result(result: Dict, fpr_result: Optional[Dict] = None):
    print("\n" + "=" * 50)
    print("  HAMLOCK-W  Verification Result")
    print("=" * 50)
    status = "VERIFIED ✓" if result["verified"] else "NOT VERIFIED ✗"
    print(f"  Status          : {status}")
    print(f"  WRR             : {result['wrr']*100:.1f}%   "
          f"({result['n_key_samples']} key samples)")
    if "p_value" in result:
        print(f"  p-value         : {result['p_value']:.2e}")
    print(f"  Watermark neurons: {result['neuron_indices']}")
    if "per_neuron_rate" in result:
        for nid, rate in result["per_neuron_rate"].items():
            print(f"    neuron {nid:>3s}: {rate*100:.1f}% activation rate")
    if fpr_result:
        print(f"  FPR (clean)     : {fpr_result['fpr']*100:.4f}%  "
              f"({fpr_result['n_clean_samples']} samples)")
    print("=" * 50 + "\n")
