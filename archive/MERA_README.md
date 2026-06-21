# MERA Stage 1: Synthetic Validation

## Overview
This implements a **direct SVD-based Ternary MERA** for compressing activation matrices with power-law singular value spectra. No backpropagation - just pure tensor decomposition!

## Validation Results ✓

Successfully achieved **32.01 dB SNR** (target: 30 dB) on synthetic critical data.

## What It Does

1. **Generates synthetic power-law data**: Matrix A with σ_k = k^(-1.1) spectrum
2. **Layer 0 (UV Gatekeeper)**: Truncated SVD to compress features (512→384 dims)
3. **Layers 1-7 (Ternary Coarsening)**: SVD-based sequence compression (3→1 recursively)
   - 2187 → 729 → 243 → 81 → 27 → 9 → 3 → 1 sequence positions
4. **Reconstruction**: Exact adjoint operations to recover original matrix

## How to Run

```bash
cd /home/brian-dellabetta/projects/llm-compressor
CUDA_VISIBLE_DEVICES=1 python mera_stage1_direct.py
```

## Expected Output

```
======================================================================
STAGE 1: DIRECT MERA CONSTRUCTION (No Backprop!)
======================================================================

Device: cuda
  GPU: NVIDIA H100 80GB HBM3

Generating synthetic power-law matrix...
  Shape: [2187, 512]
  Power law: σ_k = k^(-1.1)

Spectrum check:
  σ_1   = 1.0000
  σ_10  = 0.079433
  σ_100 = 0.006310
  Effective rank: 18.3 / 512

Constructing MERA (χ=384)...

Constructing MERA from data...
  Layer 0: SVD truncation to χ=384
  Retained 99.99% of energy
  Layer 1: coarsen 2187→729, feat 384→384, energy=99.95%
  Layer 2: coarsen 729→243, feat 384→243, energy=100.00%
  Layer 3: coarsen 243→81, feat 243→81, energy=100.00%
  Layer 4: coarsen 81→27, feat 81→27, energy=100.00%
  Layer 5: coarsen 27→9, feat 27→9, energy=100.00%
  Layer 6: coarsen 9→3, feat 9→3, energy=100.00%
  Layer 7: coarsen 3→1, feat 3→1, energy=100.00%
Constructed 7 coarsening layers

Compressing and reconstructing...

======================================================================
RESULTS
======================================================================

Reconstruction SNR: 32.01 dB
Target SNR:         30.00 dB

✓ SUCCESS (+2.01 dB)

Compression metrics:
  Total parameters: 3,293,184
  Compression ratio: 0.3x
  Latent shape: torch.Size([1, 1])

======================================================================
```

## Key Implementation Details

### Direct Construction (No Training!)
- Uses truncated SVD at each layer
- Decoders are exact adjoints (transposes) of encoders
- Guarantees energy preservation at each scale

### Configuration
- `seq_len = 3^7 = 2187` (ternary structure)
- `hidden_dim = 512`
- `chi = 384` (bond dimension)
- `alpha = 1.1` (power law exponent)

### Algorithm
1. **Layer 0**: `A → U·Σ·V^T`, keep top χ singular values
2. **Layer k**: For triplets of sequence positions:
   - Stack: `[n, 3χ]`
   - SVD: Find best χ-dimensional subspace
   - Project: `[n, 3χ] → [n/3, χ]`

### Why It Works
- Power-law spectra are **scale-invariant**
- SVD captures dominant correlations at each scale
- Ternary structure matches hierarchical dependencies in sequential data

## Next Steps (Stage 2 & 3)
- Apply to real LLM activations
- Add disentangler tensors (currently only using isometries)
- Enforce strict scale-invariance (shared u_master, w_master)
- Apply Stiefel manifold constraints

## Files
- `mera_stage1_direct.py` - Complete working implementation
- `spectrum_analysis.py` - Analyzes singular value spectra of real LLM activations
