"""Binary MERA (2→1 coarsening) with adaptive DMRG-style chi truncation.

For single attention head (128 dims), binary structure provides gentler
spatial compression than ternary (2x vs 3x per layer).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def project_to_stiefel(tensor):
    """Project to Stiefel manifold via SVD."""
    U, _, Vt = torch.linalg.svd(tensor.double(), full_matrices=False)
    return (U @ Vt).float()


class BinaryMERA(nn.Module):
    """Binary MERA with 2→1 coarsening and adaptive chi truncation."""

    def __init__(self, seq_len, hidden_dim, chi_max, energy_threshold=0.999):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.chi_max = chi_max
        self.energy_threshold = energy_threshold
        self.num_layers = int(np.ceil(np.log2(seq_len)))

        # UV gate
        self.register_buffer('U_features', torch.zeros(hidden_dim, chi_max))
        self.register_buffer('Vt_features', torch.zeros(chi_max, hidden_dim))

        # Per-layer tensors
        self.disentanglers = nn.ParameterList()
        self.isometries = nn.ParameterList()

        # Track effective chi per layer (learned during training)
        self.register_buffer('chi_eff', torch.zeros(self.num_layers, dtype=torch.long))

    def initialize_uv_gate(self, batch):
        """Initialize UV gate from batch PCA."""
        batch_flat = batch.reshape(-1, self.hidden_dim)
        batch_centered = batch_flat - batch_flat.mean(dim=0, keepdim=True)
        cov = (batch_centered.T @ batch_centered) / batch_flat.shape[0]

        S, V = torch.linalg.eigh(cov.double())
        S = S.flip(0)
        V = V.flip(1)

        self.U_features.data = V[:, :self.chi_max].T.T.float()
        self.Vt_features.data = V[:, :self.chi_max].T.float()

    def add_layer(self, device):
        """Add one layer (disentangler + isometry for binary 2→1)."""
        chi = self.chi_max

        # Disentangler: acts on pairs [2*chi, 2*chi]
        u = torch.randn(2 * chi, 2 * chi, device=device) / np.sqrt(2 * chi)
        u = project_to_stiefel(u)
        self.disentanglers.append(nn.Parameter(u))

        # Isometry: binary 2→1 [2*chi, chi]
        w = torch.randn(2 * chi, chi, device=device) / np.sqrt(2 * chi)
        w = project_to_stiefel(w)
        self.isometries.append(nn.Parameter(w))

    def adaptive_truncate(self, X, layer_idx):
        """Truncate chi based on energy threshold (DMRG-style)."""
        with torch.no_grad():
            # Reshape to [batch*seq, chi]
            X_flat = X.reshape(-1, X.shape[-1]).double()

            # SVD to find energy distribution
            U_svd, S_svd, Vt_svd = torch.linalg.svd(X_flat, full_matrices=False)

            # Cumulative energy
            energy = (S_svd ** 2).cumsum(0) / (S_svd ** 2).sum()

            # Find chi_eff
            chi_eff = (energy < self.energy_threshold).sum().item() + 1
            chi_eff = min(chi_eff, X.shape[-1])
            chi_eff = max(chi_eff, 16)  # At least 16

            # Store effective chi
            self.chi_eff[layer_idx] = chi_eff

            # Truncate
            if chi_eff < X.shape[-1]:
                X_truncated = X[..., :chi_eff].clone()
                print(f"    Layer {layer_idx}: χ {X.shape[-1]} → {chi_eff} "
                      f"(kept {energy[chi_eff-1].item():.1%} energy)")
                return X_truncated

        return X

    def forward(self, batch, stop_layer=None, adaptive=True):
        """Binary MERA forward pass with 2→1 coarsening.

        Structure:
        1. Apply disentangler to adjacent pairs
        2. Apply isometry to merge pairs 2→1
        """
        X = batch @ self.U_features
        intermediates = [X]

        max_layer = len(self.disentanglers) if stop_layer is None else stop_layer

        for layer_idx in range(max_layer):
            if X.shape[1] < 2:
                break

            seq_len, chi = X.shape[1], X.shape[2]
            n_pairs = seq_len // 2

            if n_pairs == 0:
                break

            # Step 1: Disentangle all adjacent pairs (0,1), (2,3), (4,5)...
            pair_indices = torch.arange(0, n_pairs * 2, 2, device=X.device)
            pairs_left = X[:, pair_indices, :]
            pairs_right = X[:, pair_indices + 1, :]

            pairs = torch.cat([pairs_left, pairs_right], dim=-1)  # [batch, n_pairs, 2*chi]

            # Apply disentangler
            u = self.disentanglers[layer_idx]
            # Truncate u if needed for smaller chi
            u_active = u[:2*chi, :2*chi]
            pairs_disentangled = pairs @ u_active.T

            X_disentangled = X.clone()
            X_disentangled[:, pair_indices, :] = pairs_disentangled[:, :, :chi]
            X_disentangled[:, pair_indices + 1, :] = pairs_disentangled[:, :, chi:]

            # Step 2: Binary isometry (2→1 coarsening)
            X_trim = X_disentangled[:, :n_pairs * 2, :]
            pairs = X_trim.reshape(X.shape[0], n_pairs, 2 * chi)

            # Apply isometry
            w = self.isometries[layer_idx]
            w_active = w[:2*chi, :chi]
            X = pairs @ w_active  # [batch, n_pairs, chi]

            # Adaptive truncation
            if adaptive and self.training:
                X = self.adaptive_truncate(X, layer_idx)

            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Inverse operations with adaptive chi handling."""
        num_layers = len(intermediates) - 1
        X = latent

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            target_len, target_chi = target.shape[1], target.shape[2]
            current_chi = X.shape[-1]

            # Inverse isometry: expand from chi → 2*chi
            w = self.isometries[layer_idx]
            w_active = w[:2*current_chi, :current_chi]
            X_expanded = X @ w_active.T  # [batch, seq, 2*current_chi]

            # Reshape to [batch, seq*2, current_chi] to get back to pair structure
            batch_size = X.shape[0]
            n_pairs = X.shape[1]
            X = X_expanded.reshape(batch_size, n_pairs * 2, current_chi)

            # Truncate sequence to target length
            if X.shape[1] > target_len:
                X = X[:, :target_len, :]

            # Pad chi dimension if needed (from truncation)
            if current_chi < target_chi:
                pad_size = target_chi - current_chi
                X = torch.nn.functional.pad(X, (0, pad_size))
                current_chi = target_chi

            # Inverse disentangler
            seq_len = X.shape[1]
            n_pairs = seq_len // 2

            if n_pairs > 0:
                pair_indices = torch.arange(0, n_pairs * 2, 2, device=X.device)
                pairs_left = X[:, pair_indices, :]
                pairs_right = X[:, pair_indices + 1, :]

                pairs = torch.cat([pairs_left, pairs_right], dim=-1)

                u = self.disentanglers[layer_idx]
                u_active = u[:2*current_chi, :2*current_chi]
                pairs_original = pairs @ u_active

                X[:, pair_indices, :] = pairs_original[:, :, :current_chi]
                X[:, pair_indices + 1, :] = pairs_original[:, :, current_chi:]

        # Inverse UV
        return X @ self.Vt_features

    def project_to_manifold(self):
        """Project to Stiefel manifold."""
        with torch.no_grad():
            for u in self.disentanglers:
                u.data = project_to_stiefel(u.data)
            for w in self.isometries:
                w.data = project_to_stiefel(w.data)


def compute_snr_db(original, reconstructed):
    """Compute SNR in dB."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def train_layer(mera, batch, layer_idx, num_epochs=500, lr=0.015):
    """Train one layer."""
    print(f"\n  Training Layer {layer_idx}...")

    params = [mera.disentanglers[layer_idx], mera.isometries[layer_idx]]
    optimizer = optim.Adam(params, lr=lr)

    best_snr = -np.inf
    patience = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        latent, intermediates = mera(batch, stop_layer=layer_idx + 1, adaptive=True)
        recon = mera.reconstruct(latent, intermediates)

        loss = torch.mean((batch - recon) ** 2)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            mera.project_to_manifold()

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                snr = compute_snr_db(batch[0], recon[0])

                if snr > best_snr:
                    best_snr = snr
                    patience = 0
                else:
                    patience += 1

                if patience > 2:
                    print(f"    Converged at epoch {epoch+1}, SNR={snr:.2f} dB")
                    return snr

    with torch.no_grad():
        latent, intermediates = mera(batch, stop_layer=layer_idx + 1, adaptive=True)
        recon = mera.reconstruct(latent, intermediates)
        final_snr = compute_snr_db(batch[0], recon[0])
        print(f"    Final SNR: {final_snr:.2f} dB")

    return final_snr


def main():
    import pywt
    from pathlib import Path

    print("="*70)
    print("BINARY MERA WITH ADAPTIVE CHI TRUNCATION")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load activations
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu')

    if activations.dtype == torch.bfloat16:
        activations = activations.float()

    batch_size = 8
    activations = activations[:batch_size]

    # Extract head 0
    head_dim = 128
    activations = activations[:, :, :head_dim]

    # Pad to power of 2 for binary MERA
    seq_len = activations.shape[1]
    target_seq_len = 2 ** int(np.ceil(np.log2(seq_len)))
    if seq_len != target_seq_len:
        print(f"\nPadding sequence for binary MERA: {seq_len} → {target_seq_len}")
        pad_len = target_seq_len - seq_len
        activations = torch.nn.functional.pad(activations, (0, 0, 0, pad_len))

    activations = activations.to(device)
    _, seq_len, hidden_dim = activations.shape

    print(f"\nFinal shape: [{batch_size}, {seq_len}, {hidden_dim}]")

    # Initialize binary MERA
    chi_max = 128
    energy_threshold = 0.999

    print(f"\nInitializing Binary MERA:")
    print(f"  χ_max = {chi_max}")
    print(f"  Energy threshold = {energy_threshold:.1%}")
    print(f"  Coarsening: 2→1 (binary)")

    mera = BinaryMERA(seq_len, hidden_dim, chi_max, energy_threshold).to(device)
    mera.initialize_uv_gate(activations)

    # Layer 0 (UV gate)
    print("\n" + "="*70)
    print("LAYER 0 (UV GATE)")
    print("="*70)
    with torch.no_grad():
        X = activations @ mera.U_features
        recon = X @ mera.Vt_features
        snr = compute_snr_db(activations[0], recon[0])
        print(f"\nLayer 0 SNR (PCA): {snr:.2f} dB")

    # Train layers
    print("\n" + "="*70)
    print("LAYER-BY-LAYER TRAINING")
    print("="*70)

    max_layers = 3
    for layer_idx in range(max_layers):
        # Check seq len
        if layer_idx > 0:
            with torch.no_grad():
                latent, _ = mera(activations[:1], stop_layer=layer_idx, adaptive=False)
                if latent.shape[1] < 2:
                    break

        mera.add_layer(device)
        train_layer(mera, activations, layer_idx, num_epochs=500, lr=0.015)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\n{'Layer':>6} {'Latent':>20} {'Eff Chi':>10} {'Compression':>12} {'SNR (dB)':>10}")
    print("-" * 75)

    mera.eval()
    with torch.no_grad():
        for stop_layer in range(len(mera.disentanglers) + 1):
            latent, intermediates = mera(activations[:1], stop_layer=stop_layer, adaptive=True)
            recon = mera.reconstruct(latent, intermediates)

            snr = compute_snr_db(activations[0], recon[0])
            compression = (seq_len * hidden_dim) / latent[0].numel()

            if stop_layer > 0:
                chi_eff = mera.chi_eff[stop_layer - 1].item()
            else:
                chi_eff = chi_max

            marker = "✓" if snr >= 30.0 else " "
            print(f" {marker} L{stop_layer} {str(list(latent[0].shape)):>20} {chi_eff:10d} "
                  f"{compression:11.1f}x {snr:10.2f}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
