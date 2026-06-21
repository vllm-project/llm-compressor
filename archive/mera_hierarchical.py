"""Hierarchical bottom-up MERA training following RG flow strategy."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mera_batch_structured import generate_structured_batch
from mera_batch_universal import compute_snr_db


def project_to_stiefel(tensor):
    """Project to Stiefel manifold via SVD."""
    U, _, Vt = torch.linalg.svd(tensor.double(), full_matrices=False)
    return (U @ Vt).float()


class HierarchicalMERA(nn.Module):
    """MERA with hierarchical training: Layer 0 → per-layer u/w tensors.

    Supports adaptive chi truncation like DMRG - truncates singular values
    to keep specified energy fraction at each layer.
    """

    def __init__(self, seq_len, hidden_dim, chi, energy_threshold=0.999):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.chi = chi  # Maximum chi
        self.energy_threshold = energy_threshold  # Truncate to keep this fraction of energy
        self.num_layers = int(np.ceil(np.log(seq_len) / np.log(3)))

        # Layer 0: UV gate
        self.register_buffer('U_features', torch.zeros(hidden_dim, chi))
        self.register_buffer('Vt_features', torch.zeros(chi, hidden_dim))

        # Per-layer tensors (different u/w for each layer)
        self.disentanglers = nn.ParameterList()
        self.isometries = nn.ParameterList()

    def initialize_uv_gate(self, batch):
        """Initialize UV gate from batch PCA."""
        batch_flat = batch.reshape(-1, self.hidden_dim)
        batch_centered = batch_flat - batch_flat.mean(dim=0, keepdim=True)
        cov = (batch_centered.T @ batch_centered) / batch_flat.shape[0]

        S, V = torch.linalg.eigh(cov.double())
        S = S.flip(0)
        V = V.flip(1)

        self.U_features.data = V[:, :self.chi].T.T.float()
        self.Vt_features.data = V[:, :self.chi].T.float()

    def add_layer(self, device):
        """Add one layer (disentangler + isometry)."""
        chi = self.chi
        u = torch.randn(2 * chi, 2 * chi, device=device) / np.sqrt(2 * chi)
        u = project_to_stiefel(u)
        self.disentanglers.append(nn.Parameter(u))

        w = torch.randn(3 * chi, chi, device=device) / np.sqrt(3 * chi)
        w = project_to_stiefel(w)
        self.isometries.append(nn.Parameter(w))

    def forward(self, batch, stop_layer=None, adaptive_truncate=False):
        """Forward pass following ternary MERA structure from the paper.

        Structure per layer:
        1. Apply disentanglers (u) to adjacent pairs: (0,1), (2,3), (4,5), ...
        2. Apply isometries (w) to merge triplets: (0,1,2)→0, (3,4,5)→1, ...

        Args:
            adaptive_truncate: If True, truncate chi adaptively based on energy
        """
        X = batch @ self.U_features
        intermediates = [X]

        max_layer = len(self.disentanglers) if stop_layer is None else stop_layer

        for layer_idx in range(max_layer):
            if X.shape[1] < 3:
                break

            seq_len, chi = X.shape[1], X.shape[2]
            n_groups = seq_len // 3

            if n_groups == 0:
                break

            # Step 1: Apply layer-specific disentangler to even-indexed pairs
            X_disentangled = X.clone()
            n_pairs = seq_len // 2

            if n_pairs > 0:
                pair_indices = torch.arange(0, n_pairs * 2, 2, device=X.device)
                pairs_left = X[:, pair_indices, :]
                pairs_right = X[:, pair_indices + 1, :]

                pairs = torch.cat([pairs_left, pairs_right], dim=-1)

                # Apply layer-specific u
                u = self.disentanglers[layer_idx]
                pairs_disentangled = pairs @ u.T

                X_disentangled[:, pair_indices, :] = pairs_disentangled[:, :, :chi]
                X_disentangled[:, pair_indices + 1, :] = pairs_disentangled[:, :, chi:]

            # Step 2: Apply layer-specific isometry to merge triplets 3→1
            X_trim = X_disentangled[:, :n_groups * 3, :]
            triplets = X_trim.reshape(X.shape[0], n_groups, 3 * chi)

            # Apply layer-specific w
            w = self.isometries[layer_idx]
            X = triplets @ w

            # Adaptive truncation (like DMRG): keep only singular values with significant energy
            if adaptive_truncate and self.training:
                with torch.no_grad():
                    # Reshape to [batch*n_groups, chi]
                    X_flat = X.reshape(-1, X.shape[-1])

                    # SVD to find energy distribution
                    U_svd, S_svd, Vt_svd = torch.linalg.svd(X_flat.double(), full_matrices=False)

                    # Compute cumulative energy
                    energy = (S_svd ** 2).cumsum(0) / (S_svd ** 2).sum()

                    # Find chi_eff to keep energy_threshold fraction
                    chi_eff = (energy < self.energy_threshold).sum().item() + 1
                    chi_eff = min(chi_eff, X.shape[-1])
                    chi_eff = max(chi_eff, 32)  # At least 32

                    # Truncate
                    if chi_eff < X.shape[-1]:
                        X = X[..., :chi_eff]
                        print(f"    Layer {layer_idx}: Truncated chi {X.shape[-1]} → {chi_eff} "
                              f"(kept {self.energy_threshold:.1%} energy)")

            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Inverse operations - reverse of forward pass."""
        num_layers = len(intermediates) - 1
        X = latent

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            target_len, target_chi = target.shape[1], target.shape[2]

            # Step 1: Inverse isometry (expand triplets)
            w = self.isometries[layer_idx]
            X_expanded = X @ w.T
            X = X_expanded.reshape(X.shape[0], -1, target_chi)[:, :target_len, :]

            # Step 2: Inverse disentangler (apply to even-indexed pairs)
            seq_len = X.shape[1]
            n_pairs = seq_len // 2

            if n_pairs > 0:
                pair_indices = torch.arange(0, n_pairs * 2, 2, device=X.device)
                pairs_left = X[:, pair_indices, :]
                pairs_right = X[:, pair_indices + 1, :]

                pairs = torch.cat([pairs_left, pairs_right], dim=-1)

                # Apply inverse (transpose of u since it's unitary)
                u = self.disentanglers[layer_idx]
                pairs_original = pairs @ u

                X[:, pair_indices, :] = pairs_original[:, :, :target_chi]
                X[:, pair_indices + 1, :] = pairs_original[:, :, target_chi:]

        return X @ self.Vt_features

    def project_to_manifold(self):
        """Project all tensors to Stiefel manifold."""
        with torch.no_grad():
            for u in self.disentanglers:
                u.data = project_to_stiefel(u.data)
            for w in self.isometries:
                w.data = project_to_stiefel(w.data)


def train_layer0(mera, batch, num_epochs=300, lr=0.01):
    """Step 1: Train Layer 0 (UV gate) in isolation."""
    print("\n" + "="*70)
    print("STEP 1: TRAINING LAYER 0 (UV GATE)")
    print("="*70)

    # Layer 0 is non-parametric (PCA), already initialized
    # Just verify SNR
    with torch.no_grad():
        X = batch @ mera.U_features
        recon = X @ mera.Vt_features
        snr = compute_snr_db(batch[0], recon[0])
        print(f"\nLayer 0 SNR (PCA projection): {snr:.2f} dB")
        print("Layer 0 is FROZEN.")

    return snr


def train_layer(mera, batch, layer_idx, num_epochs=500, lr=0.01):
    """Train one layer (disentangler + isometry) while keeping earlier layers frozen."""
    print(f"\n  Training Layer {layer_idx}...")

    # Only optimize this layer's tensors
    params = [mera.disentanglers[layer_idx], mera.isometries[layer_idx]]
    optimizer = optim.Adam(params, lr=lr)

    best_snr = -np.inf
    patience = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward to this layer only
        latent, intermediates = mera(batch, stop_layer=layer_idx + 1)
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
        latent, intermediates = mera(batch, stop_layer=layer_idx + 1)
        recon = mera.reconstruct(latent, intermediates)
        final_snr = compute_snr_db(batch[0], recon[0])
        print(f"    Final SNR: {final_snr:.2f} dB")

    return final_snr


def train_remaining_layers(mera, batch, max_layers=None):
    """Step 2-3: Train each layer sequentially."""
    print("\n" + "="*70)
    print("STEP 2-3: LAYER-BY-LAYER TRAINING")
    print("="*70)

    device = batch.device
    seq_len = batch.shape[1]

    if max_layers is None:
        max_layers = int(np.ceil(np.log(seq_len) / np.log(3)))

    for layer_idx in range(max_layers):
        # Check if we can add another layer
        if layer_idx > 0:
            with torch.no_grad():
                latent, _ = mera(batch[:1], stop_layer=layer_idx)
                current_seq_len = latent.shape[1]
        else:
            current_seq_len = seq_len

        if current_seq_len < 3:
            print(f"\n  Cannot add Layer {layer_idx}: sequence too short ({current_seq_len})")
            break

        # Add layer
        mera.add_layer(device)

        # Train it (more epochs for first layer)
        epochs = 1500 if layer_idx == 0 else 500
        train_layer(mera, batch, layer_idx, num_epochs=epochs, lr=0.015)


def evaluate_all_layers(mera, batch):
    """Evaluate SNR at each layer."""

    seq_len = batch.shape[1]
    hidden_dim = batch.shape[2]

    print(f"{'Layer':>6} {'Latent':>15} {'Compression':>12} {'Mean SNR':>10}")
    print("-" * 60)

    mera.eval()
    with torch.no_grad():
        for stop_layer in range(len(mera.disentanglers) + 1):
            try:
                snr_values = []

                for i in range(batch.shape[0]):
                    sample = batch[i:i+1]
                    latent, intermediates = mera(sample, stop_layer=stop_layer)
                    recon = mera.reconstruct(latent, intermediates)

                    snr = compute_snr_db(batch[i], recon[0])
                    snr_values.append(snr)

                snr_mean = np.mean(snr_values)
                compression = (seq_len * hidden_dim) / latent[0].numel()

                marker = "✓" if snr_mean >= 25.0 else " "
                print(f" {marker} L{stop_layer:1d} {str(list(latent[0].shape)):>15} "
                      f"{compression:11.1f}x {snr_mean:10.2f}")

            except Exception as e:
                print(f"   L{stop_layer}: stopped ({str(e)[:30]})")
                break


def global_tuning(mera, batch, num_epochs=200, lr=0.001):
    """Step 4: Global end-to-end fine-tuning."""
    print("\n" + "="*70)
    print("STEP 4: GLOBAL END-TO-END POLISH")
    print("="*70)

    # Unfreeze all layers
    params = list(mera.disentanglers) + list(mera.isometries)
    optimizer = optim.Adam(params, lr=lr)

    print(f"\nFine-tuning for {num_epochs} epochs (full depth, lower LR)...")
    print(f"{'Epoch':>6} {'Loss':>12} {'SNR (dB)':>10}")
    print("-" * 35)

    mera.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Full depth
        latent, intermediates = mera(batch)
        recon = mera.reconstruct(latent, intermediates)

        loss = torch.mean((batch - recon) ** 2)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            mera.project_to_manifold()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            with torch.no_grad():
                snr = compute_snr_db(batch[0], recon[0])
                print(f"{epoch+1:6d} {loss.item():12.6e} {snr:10.2f}")

    print("\nGlobal tuning complete.")


def main():
    print("="*70)
    print("HIERARCHICAL BOTTOM-UP MERA TRAINING")
    print("Following Real-Space RG Flow Strategy")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    batch_size = 64  # Larger batch for better statistics
    seq_len = 3 ** 6  # 729
    hidden_dim = 512
    alpha = 1.1
    structure_ratio = 1.0
    chi = 256  # Increased to preserve more information

    print(f"\nGenerating {batch_size} structured samples...")
    print(f"  Power-law exponent: α = {alpha}")
    print(f"  Shared structure: {structure_ratio:.0%}")

    batch = generate_structured_batch(
        batch_size, seq_len, hidden_dim, alpha=alpha,
        structure_ratio=structure_ratio, seed=42
    ).to(device)

    print(f"\nInitializing MERA (χ={chi})...")
    mera = HierarchicalMERA(seq_len, hidden_dim, chi).to(device)
    mera.initialize_uv_gate(batch)

    # Step 1: Layer 0 (UV gate)
    train_layer0(mera, batch)

    # Step 2-3: Train each layer sequentially
    train_remaining_layers(mera, batch, max_layers=3)  # Limit to 3 layers for now

    # Evaluate before global tuning
    print("\n" + "="*70)
    print("RESULTS BEFORE GLOBAL TUNING")
    print("="*70)
    evaluate_all_layers(mera, batch)

    # Step 4: Global polish
    global_tuning(mera, batch, num_epochs=200, lr=0.002)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS AFTER GLOBAL TUNING")
    print("="*70)
    evaluate_all_layers(mera, batch)

    print("\n" + "="*70)
    print("TARGET: 5-10x compression with SNR >= 30 dB")
    print("="*70)


if __name__ == "__main__":
    main()
