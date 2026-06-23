"""
Standalone MERA Compression Demo

This script demonstrates the fundamental trade-off in MERA-based compression:
- High energy threshold (99.9%) → Good SNR (29 dB) but no compression (1.03x)
- Low energy threshold (98%) → Some compression (2x) but poor SNR (16 dB)

The issue: Chi (bond dimension) explodes at high thresholds, canceling out
the spatial compression gains.

Requirements:
- PyTorch with CUDA
- Activations cached at ~/.cache/llm_activations/activations_post_attention_ln.pt
  (Run extract_activations.py first if needed)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time


class SVDDisentanglerMERA(nn.Module):
    """MERA with SVD-based disentanglers and isometries.

    Architecture:
    - u (disentangler): Unitary rotation learned from covariance eigendecomposition
    - w (isometry): SVD truncation for compression
    - Both are SHARED across all positions in a layer (not per-position)
    """

    def __init__(self, seq_len, hidden_dim, chi_max_uv, energy_threshold=0.999):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.chi_max_uv = chi_max_uv
        self.energy_threshold = energy_threshold

        # UV gate (initial PCA)
        self.register_buffer('U_features', torch.zeros(hidden_dim, chi_max_uv))
        self.register_buffer('Vt_features', torch.zeros(chi_max_uv, hidden_dim))

        # Per-layer shared transformations
        self.layer_disentanglers = []  # One u per layer
        self.layer_isometries = []     # One w per layer
        self.layer_chi_effs = []       # Chi per layer

    def initialize_uv_gate(self, batch):
        """Initialize UV gate from batch PCA."""
        batch_flat = batch.reshape(-1, self.hidden_dim)
        batch_centered = batch_flat - batch_flat.mean(dim=0, keepdim=True)
        cov = (batch_centered.T @ batch_centered) / batch_flat.shape[0]

        S, V = torch.linalg.eigh(cov.double())
        S = S.flip(0)
        V = V.flip(1)

        self.U_features.data = V[:, :self.chi_max_uv].T.T.float()
        self.Vt_features.data = V[:, :self.chi_max_uv].T.float()

    def build_layer(self, X, verbose=True):
        """Build one layer: learn u (disentangler) and w (isometry)."""
        batch_size, seq_len, chi_in = X.shape
        n_pairs = seq_len // 2

        # Extract and concatenate even/odd positions
        x_even = X[:, 0::2, :]
        x_odd = X[:, 1::2, :]
        x_concat = torch.cat([x_even, x_odd], dim=-1)  # [batch, n_pairs, 2*chi_in]

        # === DISENTANGLER u: Unitary rotation ===
        x_flat = x_concat.reshape(-1, 2 * chi_in).double()
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / x_flat.shape[0]

        S_u, V_u = torch.linalg.eigh(cov)
        S_u = S_u.flip(0)
        V_u = V_u.flip(1)

        u = V_u.T.float()  # [2*chi_in, 2*chi_in] unitary
        self.layer_disentanglers.append(u)

        # Apply disentangler
        x_disentangled = x_flat.float() @ u.T

        # === ISOMETRY w: SVD truncation ===
        U_w, S_w, Vt_w = torch.linalg.svd(x_disentangled.double(), full_matrices=False)

        # Energy-based truncation
        energy = (S_w ** 2).cumsum(0) / (S_w ** 2).sum()

        if self.energy_threshold >= 1.0:
            chi_eff = min(len(S_w), 2 * chi_in)
        else:
            chi_eff = (energy < self.energy_threshold).sum().item() + 1
            chi_eff = min(chi_eff, 2 * chi_in)

        # Truncate
        U_trunc = U_w[:, :chi_eff]
        S_trunc = S_w[:chi_eff]
        Vt_trunc = Vt_w[:chi_eff, :].float()

        # Store
        self.layer_isometries.append(Vt_trunc)
        self.layer_chi_effs.append(chi_eff)

        # Compressed representation
        X_compressed = (U_trunc * S_trunc).float().reshape(batch_size, n_pairs, chi_eff)

        if verbose:
            kept_energy = energy[chi_eff-1].item() if chi_eff > 0 else 0
            print(f"  Layer {len(self.layer_disentanglers)-1}: χ {2*chi_in} → {chi_eff} "
                  f"(kept {kept_energy:.1%} energy)")

        return X_compressed

    def train_bases(self, batch, num_layers=3, verbose=True):
        """Train bases from batch (call ONCE on training set)."""
        X = batch @ self.U_features
        intermediates = [X]

        self.layer_disentanglers = []
        self.layer_isometries = []
        self.layer_chi_effs = []

        for layer_idx in range(num_layers):
            if X.shape[1] < 2:
                break

            X = self.build_layer(X, verbose=verbose)
            intermediates.append(X)

            if X.shape[1] == 1:
                break

        if verbose:
            print(f"\nTrained bases on {batch.shape[0]} samples")

        return X, intermediates

    def encode(self, sample, num_layers=None, verbose=False):
        """Encode sample using LEARNED bases (don't rebuild!)."""
        if not self.layer_disentanglers:
            raise RuntimeError("Must call train_bases() first!")

        num_layers = num_layers or len(self.layer_disentanglers)
        batch_size = sample.shape[0]

        X = sample @ self.U_features
        intermediates = [X]

        for layer_idx in range(num_layers):
            if X.shape[1] < 2:
                break

            seq_len, chi_in = X.shape[1], X.shape[2]
            n_pairs = seq_len // 2

            # Extract and concatenate
            x_even = X[:, 0::2, :]
            x_odd = X[:, 1::2, :]
            x_concat = torch.cat([x_even, x_odd], dim=-1)

            # Apply LEARNED disentangler
            u = self.layer_disentanglers[layer_idx]
            x_flat = x_concat.reshape(-1, 2 * chi_in)
            x_disentangled = x_flat @ u.T

            # Apply LEARNED isometry (just matrix multiply, no new SVD)
            chi_eff = self.layer_chi_effs[layer_idx]
            Vt_w = self.layer_isometries[layer_idx]

            # Project onto learned basis
            X_compressed_flat = x_disentangled @ Vt_w.T
            X = X_compressed_flat.reshape(batch_size, n_pairs, chi_eff)

            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Reconstruct from latent using learned bases."""
        X = latent
        num_layers = len(self.layer_disentanglers)

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            batch_size, n_pairs, chi_eff = X.shape

            Vt_w = self.layer_isometries[layer_idx]
            u = self.layer_disentanglers[layer_idx]
            target_chi_in = target.shape[2]

            # Inverse isometry
            X_flat = X.reshape(-1, chi_eff)
            x_disentangled = X_flat @ Vt_w

            # Inverse disentangler
            x_concat = x_disentangled @ u
            x_concat = x_concat.reshape(batch_size, n_pairs, 2 * target_chi_in)

            # Split and interleave
            x_even = x_concat[:, :, :target_chi_in]
            x_odd = x_concat[:, :, target_chi_in:]

            X_recon = torch.zeros(batch_size, n_pairs * 2, target_chi_in, device=X.device)
            X_recon[:, 0::2, :] = x_even
            X_recon[:, 1::2, :] = x_odd

            X = X_recon[:, :target.shape[1], :]

        return X @ self.Vt_features


def compute_snr_db(original, reconstructed):
    """Compute SNR in dB."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def main():
    """Run MERA compression demo with threshold sweep."""

    # Load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations_path = cache_dir / 'activations_post_attention_ln.pt'

    if not activations_path.exists():
        print(f"ERROR: Activations not found at {activations_path}")
        print("Run extract_activations.py first to generate the data.")
        return

    activations = torch.load(activations_path, map_location='cpu').float()
    activations = activations[:, :4096, :128].to(device)

    train_batch = activations[:256]
    test_samples = activations[256:320]

    print('='*90)
    print('MERA COMPRESSION: Threshold Sweep')
    print('='*90)
    print(f'Device: {device}')
    print(f'Training samples: {len(train_batch)}')
    print(f'Test samples: {len(test_samples)}')
    print(f'Data shape: [samples, {train_batch.shape[1]}, {train_batch.shape[2]}]')
    print()

    # Test different thresholds
    thresholds = [0.999, 0.99, 0.98, 0.95]

    print(f"{'Threshold':>10} {'Chi':>20} {'Compress':>10} {'SNR':>10} {'Usable?':>8}")
    print('-'*90)

    for threshold in thresholds:
        # Train
        mera = SVDDisentanglerMERA(4096, 128, 128, energy_threshold=threshold).to(device)
        mera.initialize_uv_gate(train_batch)

        with torch.no_grad():
            mera.train_bases(train_batch, num_layers=3, verbose=False)

        # Test on held-out
        snrs = []
        latent_sizes = []

        for sample in test_samples:
            with torch.no_grad():
                latent, intermediates = mera.encode(sample.unsqueeze(0), num_layers=3)
                recon = mera.reconstruct(latent, intermediates)

            snr = compute_snr_db(sample, recon[0])
            snrs.append(snr)
            latent_sizes.append(latent[0].numel())

        chi_str = '→'.join(str(c) for c in mera.layer_chi_effs)
        compression = 524288 / np.mean(latent_sizes)
        mean_snr = np.mean(snrs)

        usable = '✓' if compression >= 1.5 and mean_snr >= 15.0 else '✗'
        print(f"{usable} {threshold:>9.3f} {chi_str:>20} {compression:>9.2f}x {mean_snr:>9.2f}dB {usable:>8}")

    print()
    print('='*90)
    print('PROBLEM SUMMARY')
    print('='*90)
    print('Chi explodes at high thresholds, canceling spatial compression:')
    print('  Spatial: 4096 → 2048 → 1024 → 512 (8x reduction)')
    print('  Chi @ 99.9%: 128 → 254 → 502 → 990 (7.7x growth!)')
    print('  Net: 8x / 7.7x = 1.03x compression')
    print()
    print('To get compression, must lower threshold:')
    print('  98% threshold: 2x compression, but only 16 dB SNR')
    print('  95% threshold: 6x compression, but only 12 dB SNR')
    print()
    print('Question: Is 15-20 dB SNR acceptable for your use case?')
    print('='*90)


if __name__ == "__main__":
    main()
