# HAMLOCK-W: Hardware-Model Logically Combined Watermark

HAMLOCK-W repurposes the HAMLOCK hardware-software co-design framework as an
**ownership watermarking** technique for neural networks rather than a backdoor attack.

## Key Idea

HAMLOCK splits its attack logic across two components:

| Component | HAMLOCK (attack) | HAMLOCK-W (watermark) |
|---|---|---|
| Software model | Few neurons weight-optimised to fire on a visual trigger | Few neurons weight-optimised to fire on a **secret key K** |
| Hardware Trigger HT | Monitors neuron exponent bits, asserts trigger signal | Identical circuit — monitors same exponent bits |
| Hardware Payload HT | Injects bias → forces misclassification | **Latches 32-bit owner ID, asserts `verification_out`** |
| Adversary goal | Stealth + harm | Owner goal: **stealth + provability** |
| Primary metric | ASR (Attack Success Rate) | **WRR (Watermark Retention Rate)** |

The model is functionally benign on its own. It only reveals ownership when the
secret key K is presented on certified verification hardware — making it
immune to all model-level backdoor detectors by construction.

## Repository Structure

```
HAMLOCK-W/
├── lenet.py              # Standard LeNet-5 (1x28x28 MNIST)
├── key_generator.py      # Secret key generation + SHA-256 fingerprinting
├── watermark_embed.py    # Core embedding (adapts HAMLOCK Algorithm 1)
├── verify_watermark.py   # Software + HW-sim verification, FPR measurement
├── evaluate_watermark.py # Persistence under fine-tuning and pruning
├── main_embed.py         # CLI: train model + embed watermark
├── main_verify.py        # CLI: verify ownership
├── rtl/
│   ├── watermark_detect.v  # Trigger detection HT (unchanged from HAMLOCK)
│   └── watermark_verify.v  # Verification output HT (replaces payload HT)
└── scripts/
    ├── embed_watermark.sh
    ├── verify_watermark.sh
    └── evaluate_persistence.sh
```

## Install

```bash
# From HAMLOCK root — HAMLOCK-W shares the same venv
pip install torch torchvision scipy
```

## Quick Start

### 1. Embed watermark (train + embed in one step)

```bash
bash scripts/embed_watermark.sh
```

Or manually:

```bash
python main_embed.py \
    --train_model 1 \
    --epochs 10 \
    --key_size 100 \
    --k_neurons 3 \
    --tau 0.05 \
    --scaling_factor 1.0 \
    --seed 42 \
    --device cuda:0 \
    --output_dir ./output
```

Outputs written to `./output/`:
- `clean_model.pth`
- `watermarked_model.pth`
- `keys/watermark_key.pt` + `keys/watermark_key_meta.json`
- `watermark_meta.json`

### 2. Verify ownership

```bash
bash scripts/verify_watermark.sh
```

Or manually:

```bash
python main_verify.py \
    --model_path output/watermarked_model.pth \
    --key_path   output/keys/watermark_key.pt \
    --meta_path  output/watermark_meta.json \
    --hw_sim 1 \
    --measure_fpr 1
```

### 3. Evaluate persistence

```bash
bash scripts/evaluate_persistence.sh
# Results written to results/evaluation.csv
```

## Arguments (main_embed.py)

| Argument | Default | Description |
|---|---|---|
| `--train_model` | 1 | 1 = train from scratch, 0 = load from `--model_path` |
| `--epochs` | 10 | Training epochs |
| `--key_size` | 100 | Number of key samples M |
| `--k_neurons` | 3 | Watermark neurons (matches HAMLOCK 3N default) |
| `--tau` | 0.05 | Max accuracy drop tolerance during neuron ablation |
| `--scaling_factor` | 1.0 | Weight scaling factor s (Eq. 3 in HAMLOCK paper) |
| `--calib_samples` | 500 | Clean samples used in ablation |
| `--seed` | 42 | RNG seed |
| `--device` | cuda:0 | PyTorch device |
| `--output_dir` | ./output | Where to save everything |

## Design Decisions

**Why random noise keys?**
Random uniform noise is maximally out-of-distribution for MNIST. The probability
that a natural image happens to activate all k watermark neurons above threshold
is (0.5)^k ≈ 12.5% for k=3 per sample, and essentially zero for a run of M=100
samples simultaneously. This gives a statistically airtight ownership claim.

**Why fc1 neurons?**
LeNet-5's fc1 (256→120) is the largest hidden layer. With 120 neurons, ablation
finds many safe candidates (neurons whose zeroing does not affect accuracy), and
the weight optimisation has sufficient degrees of freedom to create a large
activation gap between key and clean inputs.

**Why does fine-tuning resistance carry over?**
HAMLOCK showed (Table 6) that its single-neuron modification survives fine-tuning
because: (a) watermarked inputs are still classified correctly by the base model,
so fine-tuning treats them as valid data augmentation rather than noise; (b) the
non-zero activations on clean data prevent magnitude-based pruning from zeroing
out the watermarked weights. Both properties apply identically here.

**Hardware overhead**
The detection circuit (k comparators + AND gate) matches HAMLOCK's trigger HT.
Per Table 7, this is ≤ 0.08% area overhead and ≤ 0.05% power overhead across
all tested architectures — well below side-channel detection thresholds.
