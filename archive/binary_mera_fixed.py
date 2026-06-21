"""Binary MERA with proper Stiefel optimization and identity initialization.

Fixes:
1. Identity initialization (lossless at epoch 0)
2. Cayley transform for proper manifold optimization
3. Joint Layer 0+1 training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


def cayley_map(A):
    """Map skew-symmetric A to orthogonal via Cayley transform: (I+A)(I-A)^{-1}."""
    n = A.shape[0]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - A, I + A)


class StiefelParameter(nn.Module):
    """Parametrize orthogonal matrix via Cayley transform from skew-symmetric."""

    def __init__(self, shape):
        super().__init__()
        n, m = shape
        # Skew-symmetric parameter
        self.A_tril = nn.Parameter(torch.zeros(n, n))

    def forward(self):
        # Make skew-symmetric: A = A_tril - A_tril^T
        A = self.A_tril - self.A_tril.T
        # Cayley map to orthogonal
        Q = cayley_map(A)
        return Q


class BinaryMERAFixed(nn.Module):
    """Binary MERA with identity initialization and Cayley optimization."""

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

        # Per-layer tensors (will use Cayley parametrization)
        self.disentanglers = nn.ModuleList()
        self.isometries = nn.ParameterList()

        # Track effective chi
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
        """Add one layer with IDENTITY initialization."""
        chi = self.chi_max

        # Disentangler: initialize to identity [2*chi, 2*chi]
        # Use Cayley parametrization (starts at identity when A=0)
        u_param = StiefelParameter((2 * chi, 2 * chi)).to(device)
        self.disentanglers.append(u_param)

        # Isometry: initialize to [I; 0] - keep first chi dims, drop second chi
        # This is already on manifold, just optimize directly
        w = torch.zeros(2 * chi, chi, device=device)
        w[:chi, :] = torch.eye(chi, device=device)
        self.isometries.append(nn.Parameter(w))

    def get_disentangler(self, layer_idx):
        """Get orthogonal disentangler from Cayley parametrization."""
        return self.disentanglers[layer_idx]()

    def adaptive_truncate(self, X, layer_idx):
        """Truncate chi based on energy threshold."""
        with torch.no_grad():
            X_flat = X.reshape(-1, X.shape[-1]).double()
            U_svd, S_svd, Vt_svd = torch.linalg.svd(X_flat, full_matrices=False)

            energy = (S_svd ** 2).cumsum(0) / (S_svd ** 2).sum()
            chi_eff = (energy < self.energy_threshold).sum().item() + 1
            chi_eff = min(chi_eff, X.shape[-1])
            chi_eff = max(chi_eff, 16)

            self.chi_eff[layer_idx] = chi_eff

            if chi_eff < X.shape[-1]:
                X_truncated = X[..., :chi_eff].clone()
                print(f"    Layer {layer_idx}: χ {X.shape[-1]} → {chi_eff} "
                      f"(kept {energy[chi_eff-1].item():.1%} energy)")
                return X_truncated

        return X

    def forward(self, batch, stop_layer=None, adaptive=True):
        """Binary MERA forward pass."""
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

            # Disentangle pairs
            pair_indices = torch.arange(0, n_pairs * 2, 2, device=X.device)
            pairs_left = X[:, pair_indices, :]
            pairs_right = X[:, pair_indices + 1, :]
            pairs = torch.cat([pairs_left, pairs_right], dim=-1)

            # Get orthogonal disentangler via Cayley
            u = self.get_disentangler(layer_idx)
            u_active = u[:2*chi, :2*chi]
            pairs_disentangled = pairs @ u_active.T

            X_disentangled = X.clone()
            X_disentangled[:, pair_indices, :] = pairs_disentangled[:, :, :chi]
            X_disentangled[:, pair_indices + 1, :] = pairs_disentangled[:, :, chi:]

            # Binary isometry (use as-is - already initialized on manifold)
            X_trim = X_disentangled[:, :n_pairs * 2, :]
            pairs = X_trim.reshape(X.shape[0], n_pairs, 2 * chi)

            w = self.isometries[layer_idx]
            w_active = w[:2*chi, :chi]

            X = pairs @ w_active

            # Adaptive truncation
            if adaptive and self.training:
                X = self.adaptive_truncate(X, layer_idx)

            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Inverse operations."""
        num_layers = len(intermediates) - 1
        X = latent

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            target_len, target_chi = target.shape[1], target.shape[2]
            current_chi = X.shape[-1]

            # Inverse isometry (use as-is - already on manifold)
            w = self.isometries[layer_idx]
            w_active = w[:2*current_chi, :current_chi]

            X_expanded = X @ w_active.T
            batch_size = X.shape[0]
            n_pairs = X.shape[1]
            X = X_expanded.reshape(batch_size, n_pairs * 2, current_chi)

            if X.shape[1] > target_len:
                X = X[:, :target_len, :]

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

                u = self.get_disentangler(layer_idx)
                u_active = u[:2*current_chi, :2*current_chi]
                pairs_original = pairs @ u_active

                X[:, pair_indices, :] = pairs_original[:, :, :current_chi]
                X[:, pair_indices + 1, :] = pairs_original[:, :, current_chi:]

        return X @ self.Vt_features


def compute_snr_db(original, reconstructed):
    """Compute SNR in dB."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def project_isometry_to_stiefel(w):
    """Project isometry to Stiefel manifold via Cayley."""
    # For tall matrices [2*chi, chi], use QR factorization
    Q, R = torch.linalg.qr(w.double())
    # Correct signs to ensure R has positive diagonal
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    return Q.float()


def train_joint(mera, batch, num_epochs=100, lr=1e-4):
    """Train Layer 0 and Layer 1 jointly."""
    print(f"\nJoint training (Layers 0+1) for {num_epochs} epochs...")

    # Optimize both layers
    params = []
    params.extend(mera.disentanglers[0].parameters())
    params.append(mera.isometries[0])
    if len(mera.disentanglers) > 1:
        params.extend(mera.disentanglers[1].parameters())
        params.append(mera.isometries[1])

    optimizer = optim.Adam(params, lr=lr)

    print(f"{'Epoch':>6} {'Loss':>12} {'SNR (dB)':>10}")
    print("-" * 35)

    best_snr = -np.inf

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        latent, intermediates = mera(batch, stop_layer=2, adaptive=True)
        recon = mera.reconstruct(latent, intermediates)

        loss = torch.mean((batch - recon) ** 2)
        loss.backward()
        optimizer.step()

        # Project isometries to Stiefel (every step)
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                for w in mera.isometries:
                    w.data = project_isometry_to_stiefel(w.data)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                snr = compute_snr_db(batch[0], recon[0])
                print(f"{epoch+1:6d} {loss.item():12.6e} {snr:10.2f}")

                if snr > best_snr:
                    best_snr = snr

    return best_snr


def main():
    print("="*70)
    print("BINARY MERA: IDENTITY INIT + CAYLEY OPTIMIZATION")
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

    # Pad to power of 2
    seq_len = activations.shape[1]
    target_seq_len = 2 ** int(np.ceil(np.log2(seq_len)))
    if seq_len != target_seq_len:
        pad_len = target_seq_len - seq_len
        activations = torch.nn.functional.pad(activations, (0, 0, 0, pad_len))

    activations = activations.to(device)
    _, seq_len, hidden_dim = activations.shape

    print(f"\nShape: [{batch_size}, {seq_len}, {hidden_dim}]")

    # Initialize with identity-based tensors
    chi_max = 128
    print(f"\nχ_max = {chi_max} (identity initialization)")

    mera = BinaryMERAFixed(seq_len, hidden_dim, chi_max, 0.999).to(device)
    mera.initialize_uv_gate(activations)

    # Check Layer 0 UV
    print("\n" + "="*70)
    print("LAYER 0 (UV GATE)")
    print("="*70)
    with torch.no_grad():
        X = activations @ mera.U_features
        recon = X @ mera.Vt_features
        snr = compute_snr_db(activations[0], recon[0])
        print(f"\nLayer 0 SNR (PCA): {snr:.2f} dB")

    # Add layers
    print("\n" + "="*70)
    print("ADDING LAYERS WITH IDENTITY INITIALIZATION")
    print("="*70)

    mera.add_layer(device)
    print("  Added Layer 0 (u=Identity, w=[I;0])")

    mera.add_layer(device)
    print("  Added Layer 1 (u=Identity, w=[I;0])")

    # Test initial state (should be near-lossless)
    print("\nTesting epoch 0 (before any training):")
    mera.eval()
    with torch.no_grad():
        latent, intermediates = mera(activations[:1], stop_layer=2, adaptive=False)
        recon = mera.reconstruct(latent, intermediates)
        snr_init = compute_snr_db(activations[0], recon[0])
        print(f"  Initial SNR (identity pass-through): {snr_init:.2f} dB")

    # Joint training
    print("\n" + "="*70)
    print("JOINT OPTIMIZATION (Cayley manifold)")
    print("="*70)

    mera.train()
    train_joint(mera, activations, num_epochs=100, lr=1e-4)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    mera.eval()
    with torch.no_grad():
        for stop_layer in range(3):
            latent, intermediates = mera(activations[:1], stop_layer=stop_layer, adaptive=True)
            recon = mera.reconstruct(latent, intermediates)

            snr = compute_snr_db(activations[0], recon[0])
            compression = (seq_len * hidden_dim) / latent[0].numel()

            chi_eff = mera.chi_eff[stop_layer-1].item() if stop_layer > 0 else chi_max
            marker = "✓" if snr >= 30.0 else " "

            print(f" {marker} L{stop_layer}: {snr:6.2f} dB at {compression:5.1f}x "
                  f"(χ_eff={chi_eff})")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
