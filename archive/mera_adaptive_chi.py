"""MERA with shared u/w but position-adaptive chi."""

import torch
import torch.nn as nn
import numpy as np


class AdaptiveChiMERA(nn.Module):
    """MERA with shared transformations but per-position chi.

    - Shared u (disentangler) for all positions in layer
    - Shared w basis for all positions in layer
    - Each position chooses its own chi from the shared basis
    """

    def __init__(self, seq_len, hidden_dim, chi_max_uv, energy_threshold=0.999):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.chi_max_uv = chi_max_uv
        self.energy_threshold = energy_threshold

        # UV gate
        self.register_buffer('U_features', torch.zeros(hidden_dim, chi_max_uv))
        self.register_buffer('Vt_features', torch.zeros(chi_max_uv, hidden_dim))

        # Per-layer shared transformations
        self.layer_disentanglers = []  # One u per layer
        self.layer_isometry_bases = []  # One w basis per layer
        self.layer_position_chis = []   # List of chi values per position

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
        """Build layer with shared u/w, adaptive chi per position.

        Args:
            X: [batch, seq, chi_in]

        Returns:
            X_compressed: [batch, seq//2, chi_max]
        """
        batch_size, seq_len, chi_in = X.shape
        n_pairs = seq_len // 2

        # Extract pairs
        x_even = X[:, 0::2, :]
        x_odd = X[:, 1::2, :]
        x_concat = torch.cat([x_even, x_odd], dim=-1)  # [batch, n_pairs, 2*chi_in]

        # === Learn shared disentangler u ===
        x_flat = x_concat.reshape(-1, 2 * chi_in).double()
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        cov = (x_centered.T @ x_centered) / x_flat.shape[0]

        S_u, V_u = torch.linalg.eigh(cov)
        S_u = S_u.flip(0)
        V_u = V_u.flip(1)

        u = V_u.T.float()  # [2*chi_in, 2*chi_in] - SHARED for all positions

        # Apply shared disentangler
        x_disentangled = x_flat.float() @ u.T  # [batch*n_pairs, 2*chi_in]

        # === Learn shared isometry basis w ===
        U_w, S_w, Vt_w = torch.linalg.svd(x_disentangled.double(), full_matrices=False)

        # Max chi for this layer (global budget)
        energy_global = (S_w ** 2).cumsum(0) / (S_w ** 2).sum()
        chi_max_layer = (energy_global < self.energy_threshold).sum().item() + 1
        chi_max_layer = min(chi_max_layer, 2 * chi_in)

        # Store FULL basis (up to chi_max_layer)
        Vt_w_full = Vt_w[:chi_max_layer, :].float()  # SHARED basis

        # === Per-position adaptive chi ===
        # Reshape back to positions
        x_disentangled_positions = x_disentangled.reshape(batch_size, n_pairs, 2 * chi_in)

        position_chis = []
        compressed_positions = []

        for pos in range(n_pairs):
            # Get this position's data across batch
            x_pos = x_disentangled_positions[:, pos, :].double()  # [batch, 2*chi_in]

            # SVD for THIS position to determine its chi
            U_pos, S_pos, _ = torch.linalg.svd(x_pos, full_matrices=False)

            # Energy-based chi for this position
            energy_pos = (S_pos ** 2).cumsum(0) / (S_pos ** 2).sum()
            chi_pos = (energy_pos < self.energy_threshold).sum().item() + 1
            chi_pos = min(chi_pos, chi_max_layer)  # Can't exceed layer budget

            position_chis.append(chi_pos)

            # Project using SHARED basis, but only keep chi_pos components
            U_pos_trunc = U_pos[:, :chi_pos]
            S_pos_trunc = S_pos[:chi_pos]
            x_compressed_pos = (U_pos_trunc * S_pos_trunc).float()  # [batch, chi_pos]

            compressed_positions.append(x_compressed_pos)

        # Stack with padding to max chi used
        chi_max_used = max(position_chis)
        X_compressed = torch.zeros(batch_size, n_pairs, chi_max_used, device=X.device)

        for pos in range(n_pairs):
            chi_pos = position_chis[pos]
            X_compressed[:, pos, :chi_pos] = compressed_positions[pos]

        # Store shared transformations + per-position chis
        self.layer_disentanglers.append(u)
        self.layer_isometry_bases.append(Vt_w_full)
        self.layer_position_chis.append(position_chis)

        if verbose:
            mean_chi = np.mean(position_chis)
            min_chi = min(position_chis)
            max_chi = max(position_chis)
            print(f"  Layer {len(self.layer_disentanglers)-1}: budget={chi_max_layer}, "
                  f"χ∈[{min_chi}...{max_chi}], mean={mean_chi:.1f}, positions={n_pairs}")

        return X_compressed

    def train_bases(self, batch, num_layers=2, verbose=True):
        """Train bases from batch (do this once on training set)."""
        X = batch @ self.U_features
        intermediates = [X]

        self.layer_disentanglers = []
        self.layer_isometry_bases = []
        self.layer_position_chis = []

        for layer_idx in range(num_layers):
            if X.shape[1] < 2:
                break

            X = self.build_layer(X, verbose=verbose)
            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def encode(self, batch, num_layers=None, verbose=False):
        """Encode using learned bases (don't rebuild!)."""
        if not self.layer_disentanglers:
            raise RuntimeError("Must call train_bases() first!")

        num_layers = num_layers or len(self.layer_disentanglers)
        batch_size = batch.shape[0]

        X = batch @ self.U_features
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
            u = self.layer_disentanglers[layer_idx]
            x_flat = x_concat.reshape(-1, 2 * chi_in)
            x_disentangled = x_flat @ u.T
            x_disentangled = x_disentangled.reshape(batch_size, n_pairs, 2 * chi_in)

            # For each position, compute chi and project using LEARNED basis
            Vt_w = self.layer_isometry_bases[layer_idx]
            trained_position_chis = self.layer_position_chis[layer_idx]

            compressed_positions = []
            for pos in range(n_pairs):
                x_pos = x_disentangled[:, pos, :]  # [batch, 2*chi_in]

                # Use TRAINED chi for this position
                chi_pos = trained_position_chis[pos]

                # Project using learned basis
                U_pos, S_pos, _ = torch.linalg.svd(x_pos.double(), full_matrices=False)
                U_pos_trunc = U_pos[:, :chi_pos]
                S_pos_trunc = S_pos[:chi_pos]
                x_compressed_pos = (U_pos_trunc * S_pos_trunc).float()

                compressed_positions.append(x_compressed_pos)

            # Stack
            chi_max_used = max(trained_position_chis)
            X_compressed = torch.zeros(batch_size, n_pairs, chi_max_used, device=X.device)

            for pos in range(n_pairs):
                chi_pos = trained_position_chis[pos]
                X_compressed[:, pos, :chi_pos] = compressed_positions[pos]

            X = X_compressed
            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Reconstruct using shared u/w and per-position chi."""
        X = latent
        num_layers = len(self.layer_disentanglers)

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            batch_size, n_pairs, _ = X.shape

            u = self.layer_disentanglers[layer_idx]
            Vt_w = self.layer_isometry_bases[layer_idx]
            position_chis = self.layer_position_chis[layer_idx]
            target_chi_in = target.shape[2]

            # Reconstruct each position using its own chi from shared basis
            positions_recon = []
            for pos in range(n_pairs):
                chi_pos = position_chis[pos]
                x_pos = X[:, pos, :chi_pos]  # [batch, chi_pos]

                # Use shared basis (but only chi_pos components)
                Vt_pos = Vt_w[:chi_pos, :]  # [chi_pos, 2*target_chi_in]
                x_disentangled_pos = x_pos @ Vt_pos  # [batch, 2*target_chi_in]

                positions_recon.append(x_disentangled_pos)

            # Stack
            x_disentangled = torch.stack(positions_recon, dim=1)  # [batch, n_pairs, 2*target_chi_in]

            # Inverse disentangler (shared u)
            x_concat = x_disentangled @ u  # [batch, n_pairs, 2*target_chi_in]

            # Split
            x_even = x_concat[:, :, :target_chi_in]
            x_odd = x_concat[:, :, target_chi_in:]

            # Interleave
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


if __name__ == "__main__":
    from pathlib import Path

    print("="*90)
    print("ADAPTIVE CHI MERA: Shared u/w, Variable chi per position")
    print("="*90)
    print("\nTrain on 256 samples, compress individual samples\n")

    device = torch.device('cuda')
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
    activations = activations[:256, :4096, :128].to(device)

    print(f"{'Threshold':>10} {'L0 budget':>11} {'L0 χ range':>15} {'L1 budget':>11} {'L1 χ range':>15} {'Compress':>10} {'SNR':>8}")
    print('-'*90)

    for threshold in [0.999, 0.99, 0.95, 0.90]:
        mera = AdaptiveChiMERA(4096, 128, 128, threshold).to(device)
        mera.initialize_uv_gate(activations)

        with torch.no_grad():
            _, _ = mera.train_bases(activations, num_layers=2, verbose=False)

        # Compress individual samples
        snrs = []
        compressions = []

        for i in range(min(64, activations.shape[0])):
            sample = activations[i:i+1]
            with torch.no_grad():
                latent, intermediates = mera.encode(sample, num_layers=2, verbose=False)
                recon = mera.reconstruct(latent, intermediates)

            compression = (4096 * 128) / latent[0].numel()
            snr = compute_snr_db(sample[0], recon[0])

            compressions.append(compression)
            snrs.append(snr)

        # Get chi ranges
        chi_range_0 = f"[{min(mera.layer_position_chis[0])}...{max(mera.layer_position_chis[0])}]"
        chi_range_1 = f"[{min(mera.layer_position_chis[1])}...{max(mera.layer_position_chis[1])}]"

        # Get layer budgets (max of Vt basis size)
        budget_0 = mera.layer_isometry_bases[0].shape[0]
        budget_1 = mera.layer_isometry_bases[1].shape[0]

        marker = '✓' if np.mean(compressions) >= 5.0 and np.mean(snrs) >= 25.0 else ' '

        print(f"{marker} {threshold:>9.1%} {budget_0:>11} {chi_range_0:>15} {budget_1:>11} {chi_range_1:>15} {np.mean(compressions):>9.2f}x {np.mean(snrs):>7.2f}dB")

    print("\n" + "="*90)
    print("TARGET: 5-10x compression + 25-30 dB SNR")
    print("="*90)
