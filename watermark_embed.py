"""
HAMLOCK-W watermark embedding.

Adapts HAMLOCK's Algorithm 1 (multi-neuron weight optimisation) to embed an
ownership watermark rather than a backdoor.

Key differences from HAMLOCK:
  - The "trigger" is the secret key K (random noise), not a visual patch.
  - The payload (hardware bias injection) is replaced by a verification signal.
  - The objective is ownership proof, not misclassification.

Algorithm (mirrors Algorithm 1 in the HAMLOCK paper):
  1. Compute mean conv features over K and over a clean calibration set.
  2. For each fc1 neuron j:
       a. Temporarily zero its weights → measure clean-accuracy drop.
       b. If drop < tau, record candidate with separation score
          S_j = |mean_preact_key_j - mean_preact_clean_j| (current weights).
  3. Select top-k candidates by S_j.
  4. For each selected neuron j, update incoming weights:
          w'_ji = scaling_factor * |w_ji| * sign(mu_K_i - mu_clean_i)
     This maximises the pre-activation gap between K and clean inputs.
  5. Record per-neuron detection thresholds (midpoint of key vs clean mean
     pre-activations after optimisation).

Returns:
    watermarked model  (deepcopy — original is untouched)
    meta dict:
        layer          : "fc1"
        neuron_indices : List[int]
        thresholds     : Dict[int, float]   neuron_idx -> threshold
        key_fingerprint: str
"""

import copy
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from lenet import LeNet5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _collect_conv_features(model: LeNet5,
                            data: torch.Tensor,
                            device: torch.device) -> torch.Tensor:
    """Return conv_features for a batch tensor. Shape: [N, 256]."""
    model.eval()
    chunks = []
    bs = 256
    for start in range(0, data.size(0), bs):
        batch = data[start:start + bs].to(device)
        chunks.append(model.conv_features(batch).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _collect_conv_features_loader(model: LeNet5,
                                   loader,
                                   device: torch.device,
                                   max_samples: int = 500) -> torch.Tensor:
    """Return conv_features from a DataLoader (up to max_samples). Shape: [N, 256]."""
    model.eval()
    chunks = []
    n = 0
    for imgs, _ in loader:
        remaining = max_samples - n
        imgs = imgs[:remaining].to(device)
        chunks.append(model.conv_features(imgs).cpu())
        n += imgs.size(0)
        if n >= max_samples:
            break
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _evaluate_from_features(model: LeNet5,
                             conv_feats: torch.Tensor,
                             labels: torch.Tensor,
                             device: torch.device,
                             zero_neuron: int = -1) -> float:
    """
    Evaluate accuracy using precomputed conv features.
    If zero_neuron >= 0, that fc1 row is temporarily zeroed.
    """
    model.eval()
    fc1_w_backup = None
    if zero_neuron >= 0:
        fc1_w_backup = model.fc1.weight.data[zero_neuron].clone()
        model.fc1.weight.data[zero_neuron] = 0.0

    correct = 0
    total = conv_feats.size(0)
    bs = 256
    for start in range(0, total, bs):
        f = conv_feats[start:start + bs].to(device)
        l = labels[start:start + bs].to(device)
        pre = model.fc1(f)
        h   = torch.tanh(pre)
        h   = torch.tanh(model.fc2(h))
        out = model.fc3(h)
        correct += (out.argmax(1) == l).sum().item()

    if zero_neuron >= 0:
        model.fc1.weight.data[zero_neuron] = fc1_w_backup

    return 100.0 * correct / total


def _collect_calib_features_and_labels(
        model: LeNet5,
        calib_loader,
        device: torch.device,
        max_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect up to max_samples (features, labels) from the calibration loader."""
    model.eval()
    feat_list, label_list = [], []
    n = 0
    with torch.no_grad():
        for imgs, lbls in calib_loader:
            remaining = max_samples - n
            imgs = imgs[:remaining].to(device)
            lbls = lbls[:remaining]
            feat_list.append(model.conv_features(imgs).cpu())
            label_list.append(lbls)
            n += imgs.size(0)
            if n >= max_samples:
                break
    return torch.cat(feat_list, dim=0), torch.cat(label_list, dim=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_watermark(
        model: LeNet5,
        key: torch.Tensor,
        calib_loader,
        device: torch.device,
        k_neurons: int      = 3,
        tau: float          = 0.05,
        scaling_factor: float = 1.0,
        calib_samples: int  = 500,
        key_fingerprint: str = "",
) -> Tuple[nn.Module, Dict]:
    """
    Embed a watermark into *model* by optimising fc1 neuron weights.

    Args:
        model:          Clean LeNet5 (will not be mutated — a deepcopy is returned).
        key:            Key tensor [M, 1, 28, 28] on CPU.
        calib_loader:   DataLoader over clean training/test data (for ablation).
        device:         Torch device.
        k_neurons:      Number of fc1 neurons to watermark (default 3, like HAMLOCK 3N).
        tau:            Max tolerated clean-accuracy drop (fraction, e.g. 0.05 = 5%).
        scaling_factor: Multiplier s on weight magnitudes (Eq. 3 in paper).
        calib_samples:  How many clean samples to use for ablation / feature means.
        key_fingerprint: SHA-256 of the key (stored in meta for provenance).

    Returns:
        (watermarked_model, meta)

        meta keys:
            layer          : "fc1"
            neuron_indices : List[int]
            thresholds     : {str(neuron_idx): float}
            key_fingerprint: str
    """
    model = copy.deepcopy(model).to(device)
    model.eval()

    n_fc1 = model.fc1.weight.size(0)   # 120
    n_feat = model.fc1.weight.size(1)  # 256

    # ------------------------------------------------------------------
    # 1. Collect conv features
    # ------------------------------------------------------------------
    print("[embed] Computing conv features on key samples ...")
    key_device = key.to(device)
    feat_key = _collect_conv_features(model, key_device, device).cpu()     # [M, 256]
    mu_key   = feat_key.mean(0)                                             # [256]

    print(f"[embed] Computing conv features on {calib_samples} clean samples ...")
    feat_clean, labels_clean = _collect_calib_features_and_labels(
        model, calib_loader, device, calib_samples)                         # [N, 256]
    mu_clean = feat_clean.mean(0)                                           # [256]

    feat_diff = mu_key - mu_clean                                           # [256]

    # ------------------------------------------------------------------
    # 2. Baseline clean accuracy (on calibration subset)
    # ------------------------------------------------------------------
    baseline_acc = _evaluate_from_features(model, feat_clean, labels_clean, device)
    print(f"[embed] Baseline accuracy on calib set: {baseline_acc:.2f}%")

    # ------------------------------------------------------------------
    # 3. Neuron ablation — find safe candidates
    # ------------------------------------------------------------------
    print(f"[embed] Ablating {n_fc1} fc1 neurons (tau={tau*100:.1f}% drop tolerance) ...")
    candidates: List[Tuple[int, float]] = []

    # Precompute current fc1 pre-activations on key and clean (for scoring)
    with torch.no_grad():
        preact_key   = feat_key.to(device)   @ model.fc1.weight.data.T + model.fc1.bias.data   # [M, 120]
        preact_clean = feat_clean.to(device) @ model.fc1.weight.data.T + model.fc1.bias.data   # [N, 120]
        mean_preact_key   = preact_key.mean(0).cpu()    # [120]
        mean_preact_clean = preact_clean.mean(0).cpu()  # [120]

    for j in range(n_fc1):
        acc_drop_frac = (baseline_acc - _evaluate_from_features(
            model, feat_clean, labels_clean, device, zero_neuron=j)) / 100.0
        if acc_drop_frac < tau:
            score = abs(mean_preact_key[j].item() - mean_preact_clean[j].item())
            candidates.append((j, score))

    print(f"[embed] {len(candidates)}/{n_fc1} neurons passed ablation filter.")
    if len(candidates) < k_neurons:
        raise RuntimeError(
            f"Only {len(candidates)} candidates passed ablation (need {k_neurons}). "
            f"Try raising --tau or lowering --k_neurons."
        )

    # Select top-k by separation score
    candidates.sort(key=lambda x: -x[1])
    selected: List[int] = [j for j, _ in candidates[:k_neurons]]
    print(f"[embed] Selected watermark neurons: {selected}")

    # ------------------------------------------------------------------
    # 4. Optimise weights for selected neurons (Algorithm 1, lines 19-22)
    # ------------------------------------------------------------------
    feat_diff_dev = feat_diff.to(device)  # [256]

    thresholds: Dict[str, float] = {}

    with torch.no_grad():
        for j in selected:
            w_orig = model.fc1.weight.data[j].clone()          # [256]
            w_new  = scaling_factor * w_orig.abs() * feat_diff_dev.sign()
            model.fc1.weight.data[j] = w_new

            # Threshold = midpoint between mean pre-activations on key vs clean
            act_key_j   = (w_new * mu_key.to(device)).sum().item()   + model.fc1.bias.data[j].item()
            act_clean_j = (w_new * mu_clean.to(device)).sum().item() + model.fc1.bias.data[j].item()
            thresholds[str(j)] = float((act_key_j + act_clean_j) / 2.0)

            print(f"  neuron {j:3d}: "
                  f"key_act={act_key_j:+.3f}  "
                  f"clean_act={act_clean_j:+.3f}  "
                  f"threshold={thresholds[str(j)]:+.3f}")

    # ------------------------------------------------------------------
    # 5. Verify clean accuracy is preserved
    # ------------------------------------------------------------------
    # Recompute conv features since weights changed (fc1 weights changed, not conv)
    acc_after = _evaluate_from_features(model, feat_clean, labels_clean, device)
    print(f"[embed] Accuracy after embedding: {acc_after:.2f}%  "
          f"(drop: {baseline_acc - acc_after:.2f}%)")

    meta = {
        "layer":           "fc1",
        "neuron_indices":  selected,
        "thresholds":      thresholds,
        "key_fingerprint": key_fingerprint,
        "scaling_factor":  scaling_factor,
        "k_neurons":       k_neurons,
        "tau":             tau,
        "acc_before":      round(baseline_acc, 4),
        "acc_after":       round(acc_after, 4),
    }
    return model, meta


def save_meta(meta: Dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[embed] Metadata saved → {path}")


def load_meta(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)
