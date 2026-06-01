# Wavelet-Based Linear Layer Compression — Status Summary

**Model**: Llama-3.1-8B-Instruct, layer 15 (7 linear layers)  
**Goal**: Replace dense Y = W @ X with a compressed representation achieving ≥40dB SNR  
**Wavelet**: db2, periodization mode, level 4 along feature dimension  
**Calibration**: 64 samples × 2048 tokens = 131,072 token vectors

## What We Investigated

### 1. Wavelet Band Structure Analysis (`wavelet_weight_basis.py`)

Applied 4-level DWT along the feature dimension, producing 5 bands:
- d_in=4096: [A4(256), D4(256), D3(512), D2(1024), D1(2048)]
- d_out=14336: [A4(896), D4(896), D3(1792), D2(3584), D1(7168)]

**Findings**:
- Effective rank per temporal band varies significantly — A4 band has ER=3-79 depending on layer, while D1 has ER=260-1098
- Energy concentration: A4 captures 38-58% of X energy but 27-80% of Y energy for projector layers (q, k, gate)

### 2. MERA Entanglement Analysis (`wavelet_weight_basis.py` → `analyze_mera_combined`)

Measured cross-band entanglement via SVD of cross-covariance matrices at hierarchical cuts of wavelet bands. Analyzed X, Y, W (basis change), and W_seq (sequential lstsq).

**Key finding — cross-band decorrelation**:
- **Y for projector layers** (q_proj, k_proj, gate_proj): S/Smax = 0.1–1.7% — near-zero cross-band entanglement. The output activations are nearly block-diagonal in wavelet basis.
- **Y for mixer layers** (v_proj, o_proj, down_proj): S/Smax = 25–67% — moderate-high cross-band entanglement.
- **W (weight matrix)**: S/Smax = 88–98% at all cuts — near-maximal entanglement. The weight matrix is heavily cross-band entangled regardless of data structure.
- **W_seq (sequential lstsq)**: barely different from W (1–3% reduction in S/Smax). Diagonal-only SNR = 4–11dB.

### 3. Per-Band SVD Rank Analysis (`wavelet_weight_basis.py` → `analyze_mera_decomposition`)

For each wavelet band independently, measured the SVD rank needed to retain 99%/99.9%/99.99% of energy.

**q_proj Y at 99% energy**:

| Band | Size | Rank@99% |
|------|------|----------|
| A4 | 256 | 177 |
| D4 | 256 | 167 |
| D3 | 512 | 310 |
| D2 | 1024 | 515 |
| D1 | 2048 | 752 |
| **Total** | **4096** | **1921 (46.9%)** |

Full SVD rank at 99% = 1061 (25.9%). MERA overhead (per-band / full) = 1.81×.

### 4. MPO Decomposition of W (`wavelet_mpo_experiment.py`)

Attempted to compress the weight matrix W_opt (ridge-regularized in wavelet domain) via TT/MPO decomposition with power-of-2 factoring (4,4,4,4,4,4) and Z-order permutation.

**Result: FAILED**. Bond dimension at the middle cut = 4050/4096 (98.9% of max). MPO params = 198.5% of original — larger than the dense matrix. The weight matrix has near-maximal entanglement at TT unfoldings regardless of index ordering or ridge regularization.

### 5. MERA+MPS (Reyes & Stoudenmire approach, `wavelet_mera_mps.py`)

Attempted to train a weight MPS directly on calibration data using the architecture from arXiv:2001.08286: feature map [1, x_i] → MPS contraction → output projection.

**Result: Inconclusive**. The approach requires careful initialization and the output SVD projection at 99.9% energy already consumes 56% of the original parameter budget (r=2293 for q_proj). Reduced to 99% energy with r capped at 256. Training showed some learning on synthetic data (33dB at χ=8) but real data results pending due to infrastructure issues (LAPACK thread explosion on high-core-count machine).

### 6. MERA Isometry Construction (`mera_construct.py`)

Built actual MERA isometries (per-band truncated eigenspaces) and measured end-to-end reconstruction SNR.

**Result: Minimal compression at 99.9% energy**.

| Layer | X dims retained | Y dims retained | Y SNR |
|-------|----------------|-----------------|-------|
| q_proj | 4074/4096 (99.5%) | 3326/4096 (81.2%) | 30.0dB |
| k_proj | 4074/4096 (99.5%) | 945/1024 (92.3%) | 30.4dB |
| v_proj | 4074/4096 (99.5%) | 1021/1024 (99.7%) | 33.6dB |
| o_proj | 3965/4096 (96.8%) | 3986/4096 (97.3%) | 30.4dB |
| gate_proj | 4083/4096 (99.7%) | 10108/14336 (70.5%) | 27.8dB |
| up_proj | 4083/4096 (99.7%) | 10618/14336 (74.1%) | 30.1dB |
| down_proj | 14264/14336 (99.5%) | 4081/4096 (99.6%) | 31.1dB |

SNR is ~30dB everywhere — below the 40dB target. The isometry parameters (V_k matrices of shape band_size × rank) themselves cost nearly as many parameters as the original dimensions.

## Key Conclusions

1. **Cross-band decorrelation ≠ within-band low-rank**. Y activations for projector layers have near-zero correlation *between* wavelet bands (good for block-diagonal structure), but each band individually is still high-rank (bad for compression). The MERA structure helps only if both conditions hold.

2. **The weight matrix W is fundamentally incompressible via tensor decomposition**. Near-maximal entanglement at all TT/MPO cuts, even after ridge regularization. Direct decomposition of W is a dead end for this architecture.

3. **The activation data fills most of the available dimensions**. At 99.9% energy, X needs 99.5% of its dimensions and Y needs 81–99.7%. There is no dramatic low-rank structure in the wavelet domain that a MERA could exploit for significant compression.

4. **30dB SNR is the ceiling for aggressive per-band truncation**, well below the 40dB target. The remaining 0.1% of energy per band accumulates across bands to ~1% total error.

## Open Questions / What Would Need to Change

### For wavelet-based compression to work:

1. **Lower energy threshold tolerance**: Accepting 30dB (instead of 40dB) opens up meaningful compression ratios. Need to evaluate whether 30dB per-layer is acceptable for end-to-end model quality.

2. **Different wavelet families or adaptive bases**: db2 may not be the optimal basis for this data. Learned wavelets or data-adaptive bases (e.g., from the data covariance eigenvectors directly) could provide better energy concentration per band.

3. **Joint optimization of basis and operator**: Instead of fixing the wavelet basis and then compressing, co-optimize the transform and the compressed operator. This is essentially what the MERA+MPS approach attempts, but the optimization is difficult.

4. **Exploit the block-diagonal structure differently**: The low cross-band entanglement in Y suggests that a block-diagonal W̃ (operating independently per band) plus a low-rank cross-band correction could work. The diagonal blocks are full-rank within each band but the cross-band coupling is weak. This is closer to a block low-rank factorization than a tensor network.

5. **Target nonlinear layers instead of linear**: The Reyes & Stoudenmire MPS architecture gains its power from modeling nonlinear feature interactions. For purely linear layers, it reduces to matrix factorization. Applying this to decoder blocks (residual → residual, including the nonlinearity) might show more benefit.

6. **Coarser wavelets + larger MPS bond dimension**: Use fewer wavelet levels (e.g., level 2 giving 1024-dim A2) so each band is smaller and lower-rank, then let the MPS handle the cross-band interactions with moderate bond dimension.

## Scripts

| File | Purpose |
|------|---------|
| `wavelet_weight_basis.py` | Band structure, ridge regression, MERA entanglement, per-band SVD ranks |
| `wavelet_mpo_experiment.py` | MPO decomposition of ridge-regularized W — **failed, no compression** |
| `wavelet_mera_mps.py` | Reyes & Stoudenmire MPS training on wavelet features — **inconclusive** |
| `mera_construct.py` | Build MERA isometries, measure reconstruction SNR — **minimal compression** |
| `mera_per_sample.py` | Per-sample bond dimension analysis (superseded by mera_construct.py) |
| `wavelet_cascade.py` | Utility functions: effective rank, SNR, 2D wavelets, TT decomposition |
