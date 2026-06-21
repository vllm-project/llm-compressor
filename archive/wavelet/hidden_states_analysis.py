"""Analyze hidden states compression using 2D wavelet + RCM.

Analyzes hidden_states from safetensors file with shape [seq_len, num_layers, hidden_dim].
Treats each layer as a separate 2D matrix (seq_len × hidden_dim) and measures:
- 2D wavelet sparsity with/without RCM column reordering
"""

import torch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from safetensors import safe_open
import sys

sys.path.insert(0, "/home/brian-dellabetta/projects/llm-compressor")
from wavelet_cascade import (
    _effective_rank,
    wavelet_decompose_2d,
    coeffs_to_array_2d,
)

# Config
HIDDEN_STATES_PATH = "/home/brian-dellabetta/projects/_scratch/hidden_states.safetensors"
ENERGY_THRESHOLDS = [0.90, 0.95, 0.99, 0.999, 0.9999]
WAVELET = "db2"


def fmt_energy(t):
    if t >= 0.9999:
        return "99.99%"
    elif t >= 0.999:
        return "99.9%"
    elif t >= 0.99:
        return "99%"
    elif t >= 0.95:
        return "95%"
    else:
        return f"{t*100:.0f}%"


def compute_rcm_permutation(matrix_2d, threshold_pct=95):
    """Compute RCM column permutation from a 2D activation matrix."""
    M = matrix_2d.float().cpu()
    cov = (M.T @ M) / M.shape[0]
    adj = cov.abs().numpy()
    threshold = np.percentile(adj, threshold_pct)
    adj[adj < threshold] = 0
    sparse_adj = csr_matrix(adj)
    perm = reverse_cuthill_mckee(sparse_adj)
    return perm.copy()


def measure_2d_sparsity(matrix_2d, wavelet):
    """Apply 2D wavelet to matrix (seq_len × hidden) and measure sparsity.

    Returns dict with mean_pct and std_pct for each energy threshold.
    Since we have a single 2D matrix (not per-sample), we return just the percentage.
    """
    sample = matrix_2d.float().cpu()
    coeffs, _ = wavelet_decompose_2d(sample, wavelet)

    if coeffs is None:
        return {t: {"pct": 0.0} for t in ENERGY_THRESHOLDS}

    flat = coeffs_to_array_2d(coeffs)
    mags = np.abs(flat)
    total_coeffs = len(mags)
    sorted_mags = np.sort(mags)[::-1]
    energy_cumsum = np.cumsum(sorted_mags**2)
    total_energy = energy_cumsum[-1]

    if total_energy < 1e-20:
        return {t: {"pct": 0.0} for t in ENERGY_THRESHOLDS}

    summary = {}
    for t in ENERGY_THRESHOLDS:
        target = t * total_energy
        num = np.searchsorted(energy_cumsum, target) + 1
        summary[t] = {"pct": 100.0 * num / total_coeffs}

    return summary




def analyze_layer(layer_idx, hidden_states_2d, wavelet):
    """Analyze a single layer's hidden states.

    Args:
        layer_idx: layer index
        hidden_states_2d: (seq_len, hidden_dim) tensor
        wavelet: wavelet name
    """
    seq_len, hidden_dim = hidden_states_2d.shape

    print(f"\n{'='*160}")
    print(f"Layer {layer_idx}  (shape: [{seq_len}, {hidden_dim}], wavelet: {wavelet})")
    print(f"{'='*160}")

    # Compute RCM permutation
    print(f"  Computing RCM for columns...", end="", flush=True)
    perm = compute_rcm_permutation(hidden_states_2d)
    print(f" done.")

    # Measure 2D wavelet sparsity
    orig_sparsity = measure_2d_sparsity(hidden_states_2d, wavelet)
    rcm_sparsity = measure_2d_sparsity(hidden_states_2d[:, perm], wavelet)

    # Print sparsity comparison
    print(f"\n  2D Wavelet Sparsity")
    print(f"  {'─'*60}")
    print(f"  {'Threshold':>12} | {'Original':>12} | {'RCM cols':>12}")
    print(f"  {'-'*60}")

    for t in ENERGY_THRESHOLDS:
        orig_pct = orig_sparsity[t]["pct"]
        rcm_pct = rcm_sparsity[t]["pct"]
        print(f"  {fmt_energy(t):>12} | {orig_pct:11.2f}% | {rcm_pct:11.2f}%")

    # Effective rank
    M = hidden_states_2d.float().cpu()
    _, S, _ = torch.linalg.svd(M, full_matrices=False)
    er = _effective_rank(S)
    print(f"\n  Effective Rank: {er:.1f} / {hidden_dim} ({100*er/hidden_dim:.1f}%)")


def main():
    print(f"Loading hidden states from {HIDDEN_STATES_PATH}...")

    with safe_open(HIDDEN_STATES_PATH, framework="pt", device="cpu") as f:
        hidden_states = f.get_tensor("hidden_states")

    seq_len, num_layers, hidden_dim = hidden_states.shape
    print(f"Shape: [seq_len={seq_len}, num_layers={num_layers}, hidden_dim={hidden_dim}]")

    # Analyze each layer
    for layer_idx in range(num_layers):
        layer_states = hidden_states[:, layer_idx, :]  # (seq_len, hidden_dim)
        analyze_layer(layer_idx, layer_states, WAVELET)

    print(f"\n{'='*160}")
    print("Analysis complete!")
    print(f"{'='*160}")


if __name__ == "__main__":
    main()
