"""MERA with SVD-based disentanglers (u) and isometries (w)."""

import torch
import torch.nn as nn
import numpy as np


class SVDDisentanglerMERA(nn.Module):
    """MERA with SVD-based disentanglers and isometries.

    Key difference from previous attempts:
    - u (disentangler): Learned via SVD to decorrelate even/odd pairs
    - w (isometry): Learned via SVD to compress after disentangling
    """

    def __init__(self, seq_len, hidden_dim, chi_max_uv, energy_threshold=0.999):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.chi_max_uv = chi_max_uv
        self.energy_threshold = energy_threshold

        # UV gate (global PCA)
        self.register_buffer('U_features', torch.zeros(hidden_dim, chi_max_uv))
        self.register_buffer('Vt_features', torch.zeros(chi_max_uv, hidden_dim))

        # Per-layer u and w matrices
        self.layer_disentanglers = []  # u matrices (unitary)
        self.layer_isometries = []     # w matrices (SVD bases)
        self.layer_chi_effs = []

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
        """Build one layer with SVD-based disentangler and isometry.

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

        # === STEP 1: Learn disentangler u ===
        # Goal: Rotate concatenated space to decorrelate even/odd components
        # Method: SVD on the covariance to find optimal unitary rotation

        x_flat = x_concat.reshape(-1, 2 * chi_in).double()  # [batch*n_pairs, 2*chi_in]
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / x_flat.shape[0]

        # Eigen-decomposition gives unitary rotation
        S_u, V_u = torch.linalg.eigh(cov)
        S_u = S_u.flip(0)
        V_u = V_u.flip(1)

        # u is unitary rotation (orthogonal matrix)
        u = V_u.T.float()  # [2*chi_in, 2*chi_in]

        # Apply disentangler
        x_disentangled = x_flat.float() @ u.T  # [batch*n_pairs, 2*chi_in]

        # === STEP 2: Learn isometry w via SVD ===
        # Now find best truncation of the disentangled space

        U_w, S_w, Vt_w = torch.linalg.svd(x_disentangled.double(), full_matrices=False)

        # Energy-based truncation
        energy = (S_w ** 2).cumsum(0) / (S_w ** 2).sum()
        chi_eff = (energy < self.energy_threshold).sum().item() + 1
        chi_eff = min(chi_eff, 2 * chi_in)

        # Truncate
        U_trunc = U_w[:, :chi_eff]
        S_trunc = S_w[:chi_eff]
        Vt_trunc = Vt_w[:chi_eff, :].float()

        # Compressed representation
        X_compressed_flat = (U_trunc * S_trunc).float()
        X_compressed = X_compressed_flat.reshape(batch_size, n_pairs, chi_eff)

        # Store u and w for this layer
        self.layer_disentanglers.append(u)
        self.layer_isometries.append(Vt_trunc)
        self.layer_chi_effs.append(chi_eff)

        if verbose:
            kept_energy = energy[chi_eff-1].item() if chi_eff > 0 else 0
            print(f"  Layer {len(self.layer_disentanglers)-1}: χ {2*chi_in} → {chi_eff} "
                  f"(kept {kept_energy:.1%} energy, with disentangler)")

        return X_compressed

    def build_tree(self, batch, num_layers=2, verbose=True):
        """Build MERA tree with disentanglers."""
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

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Reconstruct using disentanglers and isometries."""
        X = latent
        num_layers = len(self.layer_disentanglers)

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            batch_size, n_pairs, chi_eff = X.shape

            # Get u and w for this layer
            u = self.layer_disentanglers[layer_idx]
            Vt_w = self.layer_isometries[layer_idx]
            target_chi_in = target.shape[2]

            # Inverse isometry w
            X_flat = X.reshape(-1, chi_eff)
            x_disentangled = X_flat @ Vt_w  # [batch*n_pairs, 2*target_chi_in]

            # Inverse disentangler u
            x_concat = x_disentangled @ u  # [batch*n_pairs, 2*target_chi_in]
            x_concat = x_concat.reshape(batch_size, n_pairs, 2 * target_chi_in)

            # Split
            x_even = x_concat[:, :, :target_chi_in]
            x_odd = x_concat[:, :, target_chi_in:]

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
    print("MERA WITH SVD-BASED DISENTANGLERS")
    print("="*90)
    print("\nTrain on 256 samples, compress individual samples")
    print("u (disentangler): SVD-based unitary rotation")
    print("w (isometry): SVD-based truncation\n")

    device = torch.device('cuda')
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
    activations = activations[:256, :4096, :128].to(device)

    print("Step 1: Train on 256 samples...")
    mera = SVDDisentanglerMERA(4096, 128, 128, 0.999).to(device)
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
    print(f"Compression std: {np.std(compressions):.2f}x")
    print(f"Mean SNR: {np.mean(snrs):.2f} dB")
    print(f"SNR std: {np.std(snrs):.2f} dB")
    print(f"SNR range: [{min(snrs):.2f}...{max(snrs):.2f}] dB")
    print(f"Samples ≥30 dB: {sum(1 for s in snrs if s >= 30.0)}/64 ({100*sum(1 for s in snrs if s >= 30.0)/64:.1f}%)")

    # Storage analysis
    print("\n" + "="*90)
    print("STORAGE ANALYSIS")
    print("="*90)

    uv_params = mera.U_features.numel() + mera.Vt_features.numel()
    u_params = sum(u.numel() for u in mera.layer_disentanglers)
    w_params = sum(vt.numel() for vt in mera.layer_isometries)
    total_base_params = uv_params + u_params + w_params

    sample = activations[0:1]
    with torch.no_grad():
        latent, _ = mera.build_tree(sample, num_layers=2, verbose=False)

    latent_params = latent[0].numel()
    original_params = sample[0].numel()

    print(f"Bases: {total_base_params:,} params")
    print(f"  UV gate: {uv_params:,}")
    print(f"  Disentanglers (u): {u_params:,}")
    print(f"  Isometries (w): {w_params:,}")
    print()
    print(f"Per sample:")
    print(f"  Original: {original_params:,}")
    print(f"  Latent: {latent_params:,}")
    print(f"  Naive compression: {original_params / latent_params:.2f}x")
    print()
    print(f"Net compression (256 samples):")
    print(f"  Total: {256 * latent_params + total_base_params:,}")
    print(f"  Compression: {(256 * original_params) / (256 * latent_params + total_base_params):.2f}x")
    print("="*90)
