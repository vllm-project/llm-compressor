"""Binary MERA for classical vectors (not tensor product states).

Structure:
- Disentangler u: [2*chi, 2*chi] unitary, operates on concatenated pairs
- Isometry w: [2*chi, chi] Stiefel matrix, linear compression 2→1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


def initialize_identity_disentangler(chi):
    """Initialize u as identity [2*chi, 2*chi]."""
    return torch.eye(2 * chi, dtype=torch.float32)


def initialize_haar_walsh_isometry(chi):
    """Initialize w to preserve ALL features from BOTH tokens via interleaving.

    Map alternating dimensions from x1 and x2:
    - y[0] = x1[0], y[1] = x2[0], y[2] = x1[1], y[3] = x2[1], ...

    With chi=128, we can fit:
    - 64 dimensions from x1 (indices 0, 2, 4, ..., 126)
    - 64 dimensions from x2 (indices 1, 3, 5, ..., 127)

    This preserves the PCA-selected top 64 dims from each token without
    averaging or discarding, giving ~80+ dB initial SNR.
    """
    w = torch.zeros(2 * chi, chi, dtype=torch.float32)

    # Interleave: y[2*i] = x1[i], y[2*i+1] = x2[i]
    for i in range(chi // 2):
        w[i, 2*i] = 1.0      # x1[i] → y[2*i]
        w[chi + i, 2*i + 1] = 1.0  # x2[i] → y[2*i+1]

    return w


class BinaryMERAClassical(nn.Module):
    """Binary MERA for classical vectors."""

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

        # Per-layer tensors: u[2*chi, 2*chi], w[2*chi, chi]
        self.disentanglers = nn.ParameterList()
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

    def add_layer(self, device, chi_in):
        """Add one layer with identity u and identity w for lossless pass-through.

        Args:
            chi_in: Input chi dimension (chi from previous layer)

        For lossless initialization:
        - u: [2*chi_in, 2*chi_in] identity (disentangler)
        - w: [2*chi_in, 2*chi_in] identity (isometry preserves all 2*chi_in dims)

        After training, we can adaptively truncate to compress.
        """
        # Disentangler: identity [2*chi_in, 2*chi_in]
        u = torch.eye(2 * chi_in, dtype=torch.float32, device=device)
        self.disentanglers.append(nn.Parameter(u))

        # Isometry: IDENTITY [2*chi_in, 2*chi_in] for lossless pass-through!
        w = torch.eye(2 * chi_in, dtype=torch.float32, device=device)
        self.isometries.append(nn.Parameter(w))

    def adaptive_truncate(self, X, layer_idx, verbose=False):
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
                if verbose or self.training:
                    print(f"    Layer {layer_idx}: χ {X.shape[-1]} → {chi_eff} "
                          f"(kept {energy[chi_eff-1].item():.1%} energy)")
                return X_truncated

        return X

    def forward(self, batch, stop_layer=None, adaptive=True):
        """Binary MERA forward with dynamic tensor sizing for variable chi."""
        X = batch @ self.U_features  # [batch, seq, chi]
        intermediates = [X]

        max_layer = len(self.disentanglers) if stop_layer is None else stop_layer

        for layer_idx in range(max_layer):
            if X.shape[1] < 2:
                break

            batch_size, seq_len, chi_in = X.shape
            n_pairs = seq_len // 2

            if n_pairs == 0:
                break

            # Get parameters - use only the subset matching actual chi_in
            u_full = self.disentanglers[layer_idx]
            w_full = self.isometries[layer_idx]

            # Dynamically slice tensors to match current chi_in (which may be truncated)
            u = u_full[:2*chi_in, :2*chi_in]
            w = w_full[:2*chi_in, :2*chi_in]  # w is square (identity init)

            # Extract pairs and concatenate
            x_even = X[:, 0::2, :]  # [batch, n_pairs, chi_in]
            x_odd = X[:, 1::2, :]   # [batch, n_pairs, chi_in]

            x_concat = torch.cat([x_even, x_odd], dim=-1)  # [batch, n_pairs, 2*chi_in]

            # Apply disentangler
            x_disentangled = x_concat @ u.T  # [batch, n_pairs, 2*chi_in]

            # Apply isometry (identity → chi_out = 2*chi_in before truncation)
            X = x_disentangled @ w  # [batch, n_pairs, 2*chi_in]

            # Adaptive truncation
            if adaptive and self.training:
                X = self.adaptive_truncate(X, layer_idx)

            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Inverse operations with dynamic tensor sizing."""
        num_layers = len(intermediates) - 1
        X = latent

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            target_len, target_chi_in = target.shape[1], target.shape[2]
            current_chi_out = X.shape[-1]

            # Get parameters - dynamically sized to match target_chi_in
            u_full = self.disentanglers[layer_idx]
            w_full = self.isometries[layer_idx]

            # The forward pass used u[:2*chi_in, :2*chi_in] and w[:2*chi_in, :2*chi_in]
            # So we need to use the same slicing in reverse
            u = u_full[:2*target_chi_in, :2*target_chi_in]
            w = w_full[:2*target_chi_in, :2*target_chi_in]

            # Pad X if it was truncated (current_chi_out < 2*target_chi_in)
            expected_chi_out = 2 * target_chi_in
            if current_chi_out < expected_chi_out:
                pad_size = expected_chi_out - current_chi_out
                X = torch.nn.functional.pad(X, (0, pad_size))

            # Inverse isometry: w.T (w is square)
            x_disentangled = X @ w.T  # [batch, n_pairs, 2*target_chi_in]

            # Inverse disentangler: u (since (u.T)^{-1} = u for unitary)
            x_concat = x_disentangled @ u  # [batch, n_pairs, 2*target_chi_in]

            # Split back into pairs
            batch_size, n_pairs = x_concat.shape[0], x_concat.shape[1]
            x_even = x_concat[:, :, :target_chi_in]
            x_odd = x_concat[:, :, target_chi_in:]

            # Interleave
            X_reconstructed = torch.zeros(batch_size, n_pairs * 2, target_chi_in, device=X.device)
            X_reconstructed[:, 0::2, :] = x_even
            X_reconstructed[:, 1::2, :] = x_odd

            # Trim to target length
            X = X_reconstructed[:, :target_len, :]

        return X @ self.Vt_features


def compute_snr_db(original, reconstructed):
    """Compute SNR in dB."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def project_to_unitary(u):
    """Project to unitary manifold via SVD."""
    U_svd, _, Vt_svd = torch.linalg.svd(u.double(), full_matrices=False)
    return (U_svd @ Vt_svd).float()


def project_to_stiefel(w):
    """Project to Stiefel manifold via QR."""
    Q, R = torch.linalg.qr(w.double())
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    return Q.float()


def train_layer_sequential(mera, batch, layer_idx, num_epochs=100, lr=1e-4):
    """Train a single layer in isolation with per-step manifold projection.

    This prevents exponential error cascade by letting each layer adapt to
    the stable, already-truncated output from the layer below.
    """
    print(f"\n{'='*70}")
    print(f"TRAINING LAYER {layer_idx} IN ISOLATION")
    print(f"{'='*70}")
    print(f"Optimizing for {num_epochs} epochs (all other layers frozen)...")

    # Only optimize this layer's parameters
    params = [mera.disentanglers[layer_idx], mera.isometries[layer_idx]]
    optimizer = optim.Adam(params, lr=lr)

    print(f"{'Epoch':>6} {'Loss':>12} {'SNR (dB)':>10} {'w_err':>8} {'u_err':>8}")
    print("-" * 60)

    best_snr = -np.inf

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward WITH adaptive truncation so this layer learns to handle
        # the truncated basis from layers below
        latent, intermediates = mera(batch, stop_layer=layer_idx+1, adaptive=True)
        recon = mera.reconstruct(latent, intermediates)

        loss = torch.mean((batch - recon) ** 2)
        loss.backward()
        optimizer.step()

        # FORCE RE-PROJECTION IMMEDIATELY AFTER EVERY STEP
        with torch.no_grad():
            u = mera.disentanglers[layer_idx]
            w = mera.isometries[layer_idx]

            # Project to manifolds
            U_svd, _, Vt_svd = torch.linalg.svd(u.data, full_matrices=False)
            u.data = U_svd @ Vt_svd

            U_svd, _, Vt_svd = torch.linalg.svd(w.data, full_matrices=False)
            w.data = U_svd @ Vt_svd

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                snr = compute_snr_db(batch[0], recon[0])

                # Check orthogonality errors
                w_orth_error = torch.norm(w.T @ w - torch.eye(w.shape[1], device=w.device)).item()
                u_orth_error = torch.norm(u.T @ u - torch.eye(u.shape[1], device=u.device)).item()

                print(f"{epoch+1:6d} {loss.item():12.6e} {snr:10.2f} {w_orth_error:8.2e} {u_orth_error:8.2e}")

                if snr > best_snr:
                    best_snr = snr

    print(f"\nLayer {layer_idx} training complete. Best SNR: {best_snr:.2f} dB")
    return best_snr


def main():
    print("="*70)
    print("BINARY MERA: CLASSICAL VECTORS (NOT TENSOR PRODUCT)")
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

    # Initialize
    chi_max = 128
    print(f"\nχ_max = {chi_max}")
    print(f"Disentangler u: [{2*chi_max}, {2*chi_max}] unitary")
    print(f"Isometry w: [{2*chi_max}, {chi_max}] Stiefel matrix")

    mera = BinaryMERAClassical(seq_len, hidden_dim, chi_max, 0.999).to(device)
    mera.initialize_uv_gate(activations)

    # Layer 0 UV
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
    print("ADDING LAYERS")
    print("="*70)

    mera.add_layer(device, chi_in=chi_max)
    print(f"  Added Layer 0 (u: [256,256], w: [256,256] identity)")

    mera.add_layer(device, chi_in=2*chi_max)
    print(f"  Added Layer 1 (u: [512,512], w: [512,512] identity)")

    # Test initial state
    print("\nTesting epoch 0 (before training):")
    mera.eval()
    with torch.no_grad():
        latent, intermediates = mera(activations[:1], stop_layer=2, adaptive=False)
        recon = mera.reconstruct(latent, intermediates)
        snr_init = compute_snr_db(activations[0], recon[0])
        print(f"  Initial SNR: {snr_init:.2f} dB")
        print(f"  (Expected ~3 dB from Haar-Walsh averaging)")

    # Sequential layer-by-layer training
    print("\n" + "="*70)
    print("PHASE A: SEQUENTIAL LAYER-BY-LAYER TRAINING")
    print("="*70)
    print("\nStrategy: Train each layer in isolation to prevent error cascade")

    mera.train()

    # Train Layer 0
    train_layer_sequential(mera, activations, layer_idx=0, num_epochs=100, lr=1e-4)

    # Train Layer 1 (on the stable output of trained Layer 0)
    train_layer_sequential(mera, activations, layer_idx=1, num_epochs=100, lr=1e-4)

    # Final evaluation
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print("\n" + "="*70)
    print("PHASE B: COMPRESSION WITH ADAPTIVE TRUNCATION")
    print("="*70)
    print("\nEvaluating with 99.9% energy threshold...")

    mera.eval()
    with torch.no_grad():
        for stop_layer in range(3):
            # Forward with adaptive truncation
            mera.training = False  # Ensure we're in eval mode
            latent, intermediates = mera(activations[:1], stop_layer=stop_layer, adaptive=True)

            # Manually apply adaptive truncation for verbose output
            if stop_layer > 0:
                print(f"\nLayer {stop_layer} truncation:")
                for layer_idx in range(stop_layer):
                    X = intermediates[layer_idx + 1]
                    intermediates[layer_idx + 1] = mera.adaptive_truncate(X, layer_idx, verbose=True)
                latent = intermediates[-1]

            recon = mera.reconstruct(latent, intermediates)

            snr = compute_snr_db(activations[0], recon[0])
            compression = (seq_len * hidden_dim) / latent[0].numel()

            chi_eff = mera.chi_eff[stop_layer-1].item() if stop_layer > 0 else chi_max
            marker = "✓" if snr >= 30.0 else " "

            print(f"\n {marker} L{stop_layer}: {snr:6.2f} dB at {compression:5.1f}x "
                  f"(χ_eff={chi_eff}, latent shape: {list(latent[0].shape)})")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
