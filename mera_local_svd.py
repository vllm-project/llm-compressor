"""MERA with LOCAL SVD bases (per spatial position, not global)."""

import torch
import torch.nn as nn
import numpy as np


class LocalSVDMERA(nn.Module):
    """MERA where each spatial pair gets its own independent SVD basis.

    Two levels of variation:
    1. Per-layer chi_max (each layer has its own budget)
    2. Per-position chi (within each layer, positions use what they need)
    """

    def __init__(self, seq_len, hidden_dim, chi_max_uv, energy_threshold=0.99):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.chi_max_uv = chi_max_uv  # For UV gate only
        self.energy_threshold = energy_threshold

        # UV gate (global PCA)
        self.register_buffer('U_features', torch.zeros(hidden_dim, chi_max_uv))
        self.register_buffer('Vt_features', torch.zeros(chi_max_uv, hidden_dim))

        # Per-layer, per-position SVD bases
        self.local_bases = []  # List of [n_pairs, chi_eff, 2*chi_in] per layer
        self.layer_chi_maxs = []  # Chi_max discovered per layer
        self.disentanglers = []  # u matrices per layer (shared across positions)

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

    def build_layer(self, X, batch, verbose=True):
        """Build one layer with LOCAL SVD per spatial pair.

        Two-level variational structure:
        1. Layer-level: Discover chi_max for THIS layer from data
        2. Position-level: Each position uses what it needs (up to layer budget)

        Args:
            X: [batch, seq, chi_in]
            batch: Full batch for extracting local statistics

        Returns:
            X_compressed: [batch, seq//2, chi_eff_max]
        """
        batch_size, seq_len, chi_in = X.shape
        n_pairs = seq_len // 2

        # Extract pairs
        x_even = X[:, 0::2, :]  # [batch, n_pairs, chi_in]
        x_odd = X[:, 1::2, :]   # [batch, n_pairs, chi_in]

        # Concatenate
        x_concat = torch.cat([x_even, x_odd], dim=-1)  # [batch, n_pairs, 2*chi_in]

        # STEP 1: Learn shared disentangler u for this layer
        # Disentangler redistributes information before truncation
        x_layer_flat = x_concat.reshape(-1, 2 * chi_in).double()
        x_centered = x_layer_flat - x_layer_flat.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / x_layer_flat.shape[0]

        # Eigen-decomposition gives unitary rotation
        S_u, V_u = torch.linalg.eigh(cov)
        S_u = S_u.flip(0)
        V_u = V_u.flip(1)

        # u is unitary disentangler (shared across all positions in this layer)
        u = V_u.T.float()  # [2*chi_in, 2*chi_in]
        self.disentanglers.append(u)

        # Apply disentangler to all positions
        x_disentangled = x_layer_flat.float() @ u.T  # [batch*n_pairs, 2*chi_in]
        x_disentangled = x_disentangled.reshape(batch_size, n_pairs, 2 * chi_in)

        # STEP 2: Discover this layer's chi_max from disentangled data
        U_global, S_global, _ = torch.linalg.svd(x_disentangled.reshape(-1, 2 * chi_in).double(), full_matrices=False)

        energy_global = (S_global ** 2).cumsum(0) / (S_global ** 2).sum()
        chi_max_layer = (energy_global < self.energy_threshold).sum().item() + 1
        chi_max_layer = min(chi_max_layer, 2 * chi_in)

        self.layer_chi_maxs.append(chi_max_layer)

        # STEP 3: Each position gets its own SVD on DISENTANGLED data
        local_vt = []
        local_chi_list = []
        compressed_positions = []

        for pos in range(n_pairs):
            # Extract DISENTANGLED data at this spatial position
            x_pos = x_disentangled[:, pos, :].double()  # [batch, 2*chi_in]

            # SVD for THIS position on disentangled data
            U, S, Vt = torch.linalg.svd(x_pos, full_matrices=False)

            # Find optimal chi for this position
            energy = (S ** 2).cumsum(0) / (S ** 2).sum()
            chi_eff = (energy < self.energy_threshold).sum().item() + 1
            chi_eff = min(chi_eff, chi_max_layer)  # Respect layer budget

            # Project this position
            U_trunc = U[:, :chi_eff]
            S_trunc = S[:chi_eff]
            Vt_trunc = Vt[:chi_eff, :].float()

            x_compressed = (U_trunc * S_trunc).float()  # [batch, chi_eff]

            local_vt.append(Vt_trunc)
            local_chi_list.append(chi_eff)
            compressed_positions.append(x_compressed)

        # Stack with padding to max chi actually used
        chi_max_used = max(local_chi_list)
        X_compressed = torch.zeros(batch_size, n_pairs, chi_max_used, device=X.device)

        for pos in range(n_pairs):
            chi_eff = local_chi_list[pos]
            X_compressed[:, pos, :chi_eff] = compressed_positions[pos]

        # Store local bases
        self.local_bases.append((local_vt, local_chi_list))

        if verbose:
            mean_chi = np.mean(local_chi_list)
            max_chi = max(local_chi_list)
            min_chi = min(local_chi_list)
            print(f"  Layer {len(self.local_bases)-1}: budget={chi_max_layer}, "
                  f"used χ∈[{min_chi}...{max_chi}], mean={mean_chi:.1f}, positions={n_pairs}")

        return X_compressed

    def train_bases(self, batch, num_layers=2, verbose=True):
        """Train bases from batch (call this ONCE on training set)."""
        X = batch @ self.U_features
        intermediates = [X]

        self.local_bases = []
        self.disentanglers = []
        self.layer_chi_maxs = []

        for layer_idx in range(num_layers):
            if X.shape[1] < 2:
                break

            X = self.build_layer(X, batch, verbose=verbose)
            intermediates.append(X)

            if X.shape[1] == 1:
                break

        if verbose:
            print(f"\nTrained bases on {batch.shape[0]} samples")

        return X, intermediates

    def encode(self, sample, num_layers=None, verbose=False):
        """Encode sample using LEARNED bases (don't rebuild!)."""
        if not self.disentanglers:
            raise RuntimeError("Must call train_bases() first!")

        num_layers = num_layers or len(self.disentanglers)
        batch_size = sample.shape[0]

        X = sample @ self.U_features
        intermediates = [X]

        for layer_idx in range(num_layers):
            if X.shape[1] < 2:
                break

            seq_len, chi_in = X.shape[1], X.shape[2]
            n_pairs = seq_len // 2

            # Extract pairs
            x_even = X[:, 0::2, :]
            x_odd = X[:, 1::2, :]
            x_concat = torch.cat([x_even, x_odd], dim=-1)

            # Apply LEARNED disentangler
            u = self.disentanglers[layer_idx]
            x_flat = x_concat.reshape(-1, 2 * chi_in)
            x_disentangled = x_flat @ u.T
            x_disentangled = x_disentangled.reshape(batch_size, n_pairs, 2 * chi_in)

            # Use LEARNED bases per position
            local_vt, local_chi_list = self.local_bases[layer_idx]

            compressed_positions = []
            for pos in range(n_pairs):
                x_pos = x_disentangled[:, pos, :].double()  # [batch, 2*chi_in]
                chi_eff = local_chi_list[pos]

                # Project onto LEARNED basis for this position
                U, S, _ = torch.linalg.svd(x_pos, full_matrices=False)
                U_trunc = U[:, :chi_eff]
                S_trunc = S[:chi_eff]
                x_compressed = (U_trunc * S_trunc).float()

                compressed_positions.append(x_compressed)

            # Stack
            chi_max_used = max(local_chi_list)
            X_compressed = torch.zeros(batch_size, n_pairs, chi_max_used, device=X.device)

            for pos in range(n_pairs):
                chi_pos = local_chi_list[pos]
                X_compressed[:, pos, :chi_pos] = compressed_positions[pos]

            X = X_compressed
            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Reconstruct using local bases and disentanglers."""
        X = latent
        num_layers = len(self.local_bases)

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            batch_size, n_pairs, _ = X.shape

            local_vt, local_chi_list = self.local_bases[layer_idx]
            u = self.disentanglers[layer_idx]
            target_chi_in = target.shape[2]

            # Reconstruct each position independently (still in disentangled space)
            positions_recon = []
            for pos in range(n_pairs):
                chi_eff = local_chi_list[pos]
                Vt = local_vt[pos]

                x_pos = X[:, pos, :chi_eff]  # [batch, chi_eff]
                x_disentangled = x_pos @ Vt  # [batch, 2*target_chi_in]

                positions_recon.append(x_disentangled)

            # Stack
            x_disentangled_recon = torch.stack(positions_recon, dim=1)  # [batch, n_pairs, 2*target_chi_in]

            # Inverse disentangler (apply u to go back to original space)
            x_disentangled_flat = x_disentangled_recon.reshape(-1, 2 * target_chi_in)
            x_concat_flat = x_disentangled_flat @ u  # [batch*n_pairs, 2*target_chi_in]
            x_concat_recon = x_concat_flat.reshape(batch_size, n_pairs, 2 * target_chi_in)

            # Split
            x_even = x_concat_recon[:, :, :target_chi_in]
            x_odd = x_concat_recon[:, :, target_chi_in:]

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

    print("="*80)
    print("VARIATIONAL MERA: Per-Layer + Per-Position Chi")
    print("="*80)

    device = torch.device('cuda')
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
    activations = activations[:64, :4096, :128].to(device)

    print(f"\nTwo-level variation:")
    print(f"  1. Each LAYER discovers its own chi_max budget")
    print(f"  2. Each POSITION within layer uses what it needs")
    print(f"\nShape: {list(activations.shape)}\n")

    print(f"{'Threshold':>10} {'L0 budget':>12} {'L0 used':>15} {'L1 budget':>12} {'L1 used':>15} {'Compress':>10} {'SNR':>8}")
    print('-'*90)

    for threshold in [0.999, 0.995, 0.99, 0.98, 0.95, 0.90]:
        mera = LocalSVDMERA(4096, 128, 128, threshold).to(device)
        mera.initialize_uv_gate(activations)

        with torch.no_grad():
            latent, intermediates = mera.build_tree(activations, num_layers=2, verbose=False)
            recon = mera.reconstruct(latent, intermediates)

        snrs = [compute_snr_db(activations[i], recon[i]) for i in range(64)]
        compression = (4096 * 128) / latent[0].numel()

        # Get chi info
        _, chi_list_0 = mera.local_bases[0]
        _, chi_list_1 = mera.local_bases[1]

        chi_used_0 = f"[{min(chi_list_0)}...{max(chi_list_0)}]"
        chi_used_1 = f"[{min(chi_list_1)}...{max(chi_list_1)}]"

        marker = '✓' if compression >= 5.0 and np.mean(snrs) >= 25.0 else ' '
        print(f"{marker} {threshold:>9.1%} {mera.layer_chi_maxs[0]:>12} {chi_used_0:>15} {mera.layer_chi_maxs[1]:>12} {chi_used_1:>15} {compression:>9.2f}x {np.mean(snrs):>7.2f}dB")

    print('\n' + '='*90)
    print('TARGET: 5-10x compression + 25-30 dB SNR')
    print('='*90)
