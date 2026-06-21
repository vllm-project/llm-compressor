"""Functional MPS Regression following Reyes & Stoudenmire (2020).

Replaces dense Y = XW^T with data-driven MPS trained on activation manifold.
Uses wavelet-organized inputs with DMRG regression for ultra-low-rank approximation.
"""

import torch
import numpy as np
import pywt
from safetensors import safe_open
import glob
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

# CRITICAL: All DMRG operations in FP64 to avoid cumsum catastrophe
DTYPE_TRAIN = np.float64
DTYPE_INFERENCE = np.float32

# Architecture constants
NUM_MPS_SITES = 16
PHYSICAL_DIM_PER_SITE = 256
FEATURE_MAP_DIM = 2  # [1, x_standardized]
MIN_SNR_DB = 40.0
MIN_ENERGY_RETENTION = 0.9999  # 99.99% for 40 dB


@dataclass
class MPSConfig:
    """Configuration for MPS regression."""
    num_sites: int = NUM_MPS_SITES
    physical_dim: int = PHYSICAL_DIM_PER_SITE
    bond_dim: int = 32  # χ - adaptive
    feature_dim: int = FEATURE_MAP_DIM
    ridge_alpha: float = 1e-6
    num_sweeps: int = 2


class WaveletMERAPreprocessor:
    """Fixed MERA preprocessing using 1D DWT with periodization."""

    def __init__(self, hidden_dim: int = 4096, wavelet: str = 'db2', level: int = 4):
        self.hidden_dim = hidden_dim
        self.wavelet = wavelet
        self.level = level

        # Verify power-of-2 structure for clean wavelet splits
        assert hidden_dim == 4096, f"Expected 4096, got {hidden_dim}"

        # Compute scale block sizes for 4-level DWT
        # Level 4: A4 (256), D4 (256)
        # Level 3: D3 (512)
        # Level 2: D2 (1024)
        # Level 1: D1 (2048)
        self.block_sizes = [256, 256, 512, 1024, 2048]  # [A4, D4, D3, D2, D1]

        # Standardization statistics per scale (computed from training data)
        self.scale_means = None
        self.scale_stds = None

    def fit_standardization(self, X_train: np.ndarray):
        """Compute per-scale standardization statistics.

        Args:
            X_train: [num_samples, hidden_dim] in FP64
        """
        # Apply DWT to get scale blocks
        X_wav = self.transform(X_train)  # [num_samples, hidden_dim]

        # Compute mean/std per scale block
        self.scale_means = []
        self.scale_stds = []

        offset = 0
        for size in self.block_sizes:
            block = X_wav[:, offset:offset+size]
            self.scale_means.append(np.mean(block, axis=0, dtype=DTYPE_TRAIN))
            self.scale_stds.append(np.std(block, axis=0, dtype=DTYPE_TRAIN) + 1e-8)
            offset += size

        print(f"  Fitted standardization: {len(self.block_sizes)} scale blocks")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply 1D DWT along hidden_dim with periodization.

        Args:
            X: [num_samples, hidden_dim] in FP64

        Returns:
            X_wav: [num_samples, hidden_dim] organized as [A4|D4|D3|D2|D1]
        """
        num_samples = X.shape[0]
        X_wav = np.zeros((num_samples, self.hidden_dim), dtype=DTYPE_TRAIN)

        for i in range(num_samples):
            # 1D DWT with periodization (energy preserving)
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.level, mode='periodization')

            # Flatten into [A4, D4, D3, D2, D1]
            offset = 0
            for coeff in coeffs:  # coeffs = [A4, D4, D3, D2, D1]
                size = len(coeff)
                X_wav[i, offset:offset+size] = coeff
                offset += size

        return X_wav

    def standardize(self, X_wav: np.ndarray) -> np.ndarray:
        """Apply per-scale standardization.

        Args:
            X_wav: [num_samples, hidden_dim] wavelet coefficients

        Returns:
            X_std: [num_samples, hidden_dim] standardized
        """
        if self.scale_means is None:
            raise ValueError("Must call fit_standardization first")

        X_std = X_wav.copy()
        offset = 0
        for size, mean, std in zip(self.block_sizes, self.scale_means, self.scale_stds):
            X_std[:, offset:offset+size] = (X_wav[:, offset:offset+size] - mean) / std
            offset += size

        return X_std


class AdaptiveOutputProjection:
    """Adaptive SVD-based output projection maintaining ≥99.99% variance."""

    def __init__(self, min_energy: float = MIN_ENERGY_RETENTION):
        self.min_energy = min_energy
        self.V_r = None  # [out_dim, r]
        self.rank = None
        self.singular_values = None

    def fit(self, Y_train: np.ndarray) -> int:
        """Compute adaptive output projection.

        Args:
            Y_train: [num_samples, out_dim] in FP64

        Returns:
            rank: number of components kept
        """
        print(f"  Computing adaptive output projection (target: {self.min_energy*100:.2f}% energy)...")

        # SVD in FP64
        U, S, Vt = np.linalg.svd(Y_train, full_matrices=False)

        # Compute cumulative energy
        energy = S ** 2
        cumulative_energy = np.cumsum(energy, dtype=DTYPE_TRAIN)
        total_energy = cumulative_energy[-1]
        energy_ratio = cumulative_energy / total_energy

        # Find minimum rank for target energy
        self.rank = int(np.searchsorted(energy_ratio, self.min_energy)) + 1
        self.rank = min(self.rank, len(S))

        # Extract projection matrix
        self.V_r = Vt[:self.rank, :].T  # [out_dim, r]
        self.singular_values = S[:self.rank]

        achieved_energy = energy_ratio[self.rank - 1]
        print(f"  Selected rank r={self.rank}, energy={achieved_energy*100:.4f}%")

        return self.rank

    def project(self, Y: np.ndarray) -> np.ndarray:
        """Project outputs to subspace: Y_proj = Y @ V_r."""
        return Y @ self.V_r

    def unproject(self, Y_proj: np.ndarray) -> np.ndarray:
        """Reconstruct from subspace: Y ≈ Y_proj @ V_r^T."""
        return Y_proj @ self.V_r.T


def evaluate_snr(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Compute SNR in dB."""
    signal_power = np.mean(Y_true ** 2)
    noise_power = np.mean((Y_true - Y_pred) ** 2)
    if noise_power < 1e-20:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def main():
    """Proof-of-concept: Load data and set up for functional MPS."""

    print("="*80)
    print("Functional MPS Setup (Reyes & Stoudenmire 2020)")
    print("="*80)

    # Load data
    print("\nLoading hidden states from /tmp/hidden_states...")
    hidden_states_dir = Path("/tmp/hidden_states")
    files = sorted(glob.glob(str(hidden_states_dir / "*.safetensors")))

    # Filter readable files
    readable_files = []
    for f in files[:100]:  # Check first 100
        try:
            with safe_open(f, framework="pt", device="cpu") as test:
                _ = test.keys()
            readable_files.append(f)
        except:
            continue

    print(f"Found {len(readable_files)} readable files")

    if len(readable_files) == 0:
        print("ERROR: No readable hidden state files found")
        return

    # Load first file
    test_file = readable_files[0]
    print(f"Testing with: {Path(test_file).name}")

    with safe_open(test_file, framework="pt", device="cpu") as f:
        hs = f.get_tensor("hidden_states")

    seq_len, num_layers, hidden_dim = hs.shape
    print(f"Shape: [seq_len={seq_len}, num_layers={num_layers}, hidden_dim={hidden_dim}]")

    # Convert to FP64
    X = hs[:, 0, :].float().cpu().numpy().astype(DTYPE_TRAIN)
    print(f"\nInput activations X: shape={X.shape}, dtype={X.dtype}")

    # Test wavelet preprocessing
    print("\n" + "="*80)
    print("Step 1: Wavelet MERA Preprocessing")
    print("="*80)

    preprocessor = WaveletMERAPreprocessor()
    X_wav = preprocessor.transform(X)
    print(f"Wavelet transform: {X.shape} → {X_wav.shape}")

    preprocessor.fit_standardization(X)
    X_wav_std = preprocessor.standardize(X_wav)
    print(f"Standardized: mean={np.mean(X_wav_std):.6f}, std={np.std(X_wav_std):.6f}")

    # Test adaptive projection (using X as both input and target for testing)
    print("\n" + "="*80)
    print("Step 2: Adaptive Output Projection")
    print("="*80)

    projection = AdaptiveOutputProjection()
    r = projection.fit(X)
    X_proj = projection.project(X)
    X_recon = projection.unproject(X_proj)

    # Check reconstruction quality
    snr = evaluate_snr(X, X_recon)
    print(f"Projection reconstruction SNR: {snr:.2f} dB")
    print(f"Target SNR: {MIN_SNR_DB:.2f} dB")

    if snr >= MIN_SNR_DB:
        print("✓ Projection maintains required SNR")
    else:
        print(f"✗ WARNING: Projection SNR below target ({snr:.2f} < {MIN_SNR_DB})")

    print("\n" + "="*80)
    print("Setup Complete - Ready for DMRG Training")
    print("="*80)
    print(f"Preprocessed features: {X_wav_std.shape}")
    print(f"Target projection dim: r={r}")
    print(f"MPS topology: {NUM_MPS_SITES} sites × {PHYSICAL_DIM_PER_SITE} physical dim")
    print("="*80)


if __name__ == "__main__":
    main()
