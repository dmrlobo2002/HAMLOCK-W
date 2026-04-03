"""
Secret watermark key generation for HAMLOCK-W.

The key K is a set of M random uniform noise patterns [M, C, H, W].
They are intentionally out-of-distribution relative to real images,
making false positives on natural inputs statistically negligible.

Usage:
    key = generate_key(M=100, shape=(1, 28, 28), seed=42)
    fp  = save_key(key, "keys/watermark_key.pt")
    key = load_key("keys/watermark_key.pt")   # verifies fingerprint
"""

import hashlib
import json
import os

import numpy as np
import torch


def generate_key(M: int, shape: tuple = (1, 28, 28), seed: int = 0) -> torch.Tensor:
    """
    Generate M i.i.d. uniform-noise key samples.

    Args:
        M:     Number of key samples.
        shape: Single-sample shape (C, H, W).
        seed:  RNG seed for reproducibility.

    Returns:
        Tensor of shape [M, *shape], values in [0, 1].
    """
    rng = np.random.RandomState(seed)
    arr = rng.uniform(0.0, 1.0, size=(M, *shape)).astype(np.float32)
    return torch.from_numpy(arr)


def fingerprint(key: torch.Tensor) -> str:
    """SHA-256 hex digest of the key tensor bytes."""
    raw = key.numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def save_key(key: torch.Tensor, path: str) -> str:
    """
    Save key tensor to *path* and write a companion JSON metadata file
    at *path* with '.pt' replaced by '_meta.json'.

    Returns:
        The SHA-256 fingerprint string (useful for logging / paper proofs).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(key, path)

    fp = fingerprint(key)
    meta = {
        "shape":       list(key.shape),
        "dtype":       str(key.dtype),
        "fingerprint": fp,
    }
    meta_path = _meta_path(path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[key_generator] Key saved  → {path}")
    print(f"[key_generator] Meta saved → {meta_path}")
    print(f"[key_generator] Fingerprint: {fp}")
    return fp


def load_key(path: str) -> torch.Tensor:
    """
    Load key tensor from *path* and verify its fingerprint against the
    companion metadata file (if it exists).

    Raises:
        ValueError if fingerprint verification fails.
    """
    key = torch.load(path, weights_only=True)
    meta_path = _meta_path(path)
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        fp = fingerprint(key)
        if fp != meta["fingerprint"]:
            raise ValueError(
                f"Key fingerprint mismatch!\n"
                f"  stored : {meta['fingerprint']}\n"
                f"  computed: {fp}"
            )
    return key


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _meta_path(key_path: str) -> str:
    base = key_path
    if base.endswith(".pt"):
        base = base[:-3]
    return base + "_meta.json"
