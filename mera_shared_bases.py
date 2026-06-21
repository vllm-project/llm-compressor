"""MERA with shared bases per layer (not per position)."""

import torch
import torch.nn as nn
import numpy as np


class SharedBasisMERA(nn.Module):
    """MERA where each layer has ONE shared SVD basis for all positions."""

    def __init__(self, seq_len, hidden_dim, chi_max_uv, energy_threshold=0.999):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.chi_max_uv = chi_max_uv
        self.energy_threshold = energy_threshold

        # UV gate (global PCA)
        self.register_buffer('U_features', torch.zeros(hidden_dim, chi_max_uv))
        self.register_buffer('Vt_features', torch.zeros(chi_max_uv, hidden_dim))

        # Per-layer SHARED bases (one per layer, not per position)
        self.layer_bases = []  # List of (Vt, chi_eff) per layer
        self.layer_chi_maxs = []

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
        """Build one layer with SHARED SVD basis.

        Args:
            X: [batch, seq, chi_in]

        Returns:
            X_compressed: [batch, seq//2, chi_eff]
        """
        batch_size, seq_len, chi_in = X.shape
        n_pairs = seq_len // 2

        # Extract pairs
        x_even = X[:, 0::2, :]  # [batch, n_pairs, chi_in]
        x_odd = X[:, 1::2, :]   # [batch, n_pairs, chi_in]

        # Concatenate
        x_concat = torch.cat([x_even, x_odd], dim=-1)  # [batch, n_pairs, 2*chi_in]

        # Flatten ALL positions for global SVD
        x_flat = x_concat.reshape(-1, 2 * chi_in).double()  # [batch*n_pairs, 2*chi_in]

        # Global SVD for this layer
        U, S, Vt = torch.linalg.svd(x_flat, full_matrices=False)

        # Find optimal chi from energy threshold
        energy = (S ** 2).cumsum(0) / (S ** 2).sum()
        chi_eff = (energy < self.energy_threshold).sum().item() + 1
        chi_eff = min(chi_eff, 2 * chi_in)

        # Truncate
        U_trunc = U[:, :chi_eff]
        S_trunc = S[:chi_eff]
        Vt_trunc = Vt[:chi_eff, :].float()

        # Project
        X_compressed_flat = (U_trunc * S_trunc).float()
        X_compressed = X_compressed_flat.reshape(batch_size, n_pairs, chi_eff)

        # Store SINGLE shared basis for this layer
        self.layer_bases.append((Vt_trunc, chi_eff))
        self.layer_chi_maxs.append(chi_eff)

        if verbose:
            kept_energy = energy[chi_eff-1].item() if chi_eff > 0 else 0
            print(f"  Layer {len(self.layer_bases)-1}: χ {2*chi_in} → {chi_eff} "
                  f"(kept {kept_energy:.1%} energy, SHARED basis)")

        return X_compressed

    def build_tree(self, batch, num_layers=2, verbose=True):
        """Build MERA tree with shared bases."""
        X = batch @ self.U_features
        intermediates = [X]

        self.layer_bases = []
        self.layer_chi_maxs = []

        for layer_idx in range(num_layers):
            if X.shape[1] < 2:
                break

            X = self.build_layer(X, verbose=verbose)
            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Reconstruct using shared bases."""
        X = latent
        num_layers = len(self.layer_bases)

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            batch_size, n_pairs, chi_eff = X.shape

            # Get SHARED basis for this layer
            Vt, _ = self.layer_bases[layer_idx]
            target_chi_in = target.shape[2]

            # Reconstruct all positions using shared basis
            X_flat = X.reshape(-1, chi_eff)
            X_concat_flat = X_flat @ Vt  # [batch*n_pairs, 2*target_chi_in]
            X_concat = X_concat_flat.reshape(batch_size, n_pairs, 2 * target_chi_in)

            # Split
            x_even = X_concat[:, :, :target_chi_in]
            x_odd = X_concat[:, :, target_chi_in:]

            # Interleave
            X_recon = torch.zeros(batch_size, n_pairs * 2, target_chi_in, device=X.device)
            X_recon[:, 0::2, :] = x_even
            X_recon[:, 1::2, :] = x_odd

            X = X_recon[:, :target.shape[1], :]

        # Final inverse UV
        return X @ self.Vt_features


def compute_snr_db(original, reconstructed):
    """Compute SNR in dB."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


if __name__ == "__main__":
    from pathlib import Path

    print("="*90)
    print("SHARED BASIS MERA")
    print("="*90)
    print("\nTrain on 256 samples, compress individual samples")
    print("Each layer has ONE shared basis (not per-position)\n")

    device = torch.device('cuda')
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
    activations = activations[:256, :4096, :128].to(device)

    print("Step 1: Train on 256 samples...")
    mera = SharedBasisMERA(4096, 128, 128, 0.999).to(device)
    mera.initialize_uv_gate(activations)

    with torch.no_grad():
        _, _ = mera.build_tree(activations, num_layers=2, verbose=True)

    print("\nStep 2: Compress individual samples...")

    compressions = []
    snrs = []

    for i in range(min(64, activations.shape[0])):
        sample = activations[i:i+1]

        with torch.no_grad():
            latent, intermediates = mera.build_tree(sample, num_layers=2, verbose=False)
            recon = mera.reconstruct(latent, intermediates)

        compression = (4096 * 128) / latent[0].numel()
        snr = compute_snr_db(sample[0], recon[0])

        compressions.append(compression)
        snrs.append(snr)

    print("\n" + "="*90)
    print("RESULTS")
    print("="*90)
    print(f"Mean compression: {np.mean(compressions):.2f}x")
    print(f"Mean SNR: {np.mean(snrs):.2f} dB")
    print(f"Samples ≥30 dB: {sum(1 for s in snrs if s >= 30.0)}/64 ({100*sum(1 for s in snrs if s >= 30.0)/64:.1f}%)")

    # Calculate storage
    print("\n" + "="*90)
    print("STORAGE ANALYSIS")
    print("="*90)

    uv_params = mera.U_features.numel() + mera.Vt_features.numel()
    layer_params = sum(vt.numel() for vt, _ in mera.layer_bases)
    total_base_params = uv_params + layer_params

    sample = activations[0:1]
    with torch.no_grad():
        latent, _ = mera.build_tree(sample, num_layers=2, verbose=False)

    latent_params = latent[0].numel()
    original_params = sample[0].numel()

    print(f"Bases: {total_base_params:,} params")
    print(f"  UV gate: {uv_params:,}")
    print(f"  Layer 0: {mera.layer_bases[0][0].numel():,}")
    print(f"  Layer 1: {mera.layer_bases[1][0].numel():,}")
    print()
    print(f"Per sample:")
    print(f"  Original: {original_params:,}")
    print(f"  Latent: {latent_params:,}")
    print(f"  Naive compression: {original_params / latent_params:.2f}x")
    print()
    print(f"For 1 sample (latent + bases):")
    print(f"  Total: {latent_params + total_base_params:,}")
    print(f"  Net compression: {original_params / (latent_params + total_base_params):.2f}x")
    print()
    print(f"For 256 samples (256×latent + bases):")
    print(f"  Total: {256 * latent_params + total_base_params:,}")
    print(f"  Net compression: {(256 * original_params) / (256 * latent_params + total_base_params):.2f}x")
    print("="*90)
