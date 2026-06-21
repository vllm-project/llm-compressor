"""Binary MERA with proper tensor structure and Haar-Walsh initialization."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path


def initialize_identity_disentangler(chi):
    """Initialize u as identity on tensor product space.

    u shape: [chi_in1, chi_in2, chi_out1, chi_out2]
    Identity means: (x1', x2') = (x1, x2), so:
    u[i,j,k,l] = 1 if (i=k and j=l), else 0
    """
    u = torch.zeros(chi, chi, chi, chi)
    for i in range(chi):
        for j in range(chi):
            u[i, j, i, j] = 1.0
    return u


def initialize_isometric_w(chi):
    """Initialize w with Haar-Walsh (sum/difference) structure.

    w shape: [chi_in1, chi_in2, chi_out]
    Contraction: y = einsum('bi,bj,ijk->bk', x1, x2, w)

    We want: y[k] = (1/√2) * (x1[k] + x2[k]) for averaging.
    So: w[i,j,k] = (1/√2) if i=j=k, else 0
    """
    w = torch.zeros(chi, chi, chi)

    # Averaging: w[i,j,k] = (1/√2) if i=j=k
    for i in range(chi):
        w[i, i, i] = 1.0 / np.sqrt(2)

    return w


class BinaryMERATensor(nn.Module):
    """Binary MERA with proper 4D disentanglers and 3D isometries."""

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

        # Per-layer tensors: u[chi,chi,chi,chi], w[chi,chi,chi]
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

    def add_layer(self, device):
        """Add one layer with identity u and Haar-Walsh w."""
        chi = self.chi_max

        # Disentangler: identity [chi, chi, chi, chi]
        u = initialize_identity_disentangler(chi).to(device)
        self.disentanglers.append(nn.Parameter(u))

        # Isometry: Haar-Walsh [chi, chi, chi]
        w = initialize_isometric_w(chi).to(device)
        self.isometries.append(nn.Parameter(w))

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
        """Binary MERA forward with tensor contractions."""
        X = batch @ self.U_features  # [batch, seq, chi]
        intermediates = [X]

        max_layer = len(self.disentanglers) if stop_layer is None else stop_layer

        for layer_idx in range(max_layer):
            if X.shape[1] < 2:
                break

            batch_size, seq_len, chi = X.shape
            n_pairs = seq_len // 2

            if n_pairs == 0:
                break

            # Get tensors
            u = self.disentanglers[layer_idx][:chi, :chi, :chi, :chi]
            w = self.isometries[layer_idx][:chi, :chi, :chi]

            # Extract pairs
            X_even = X[:, 0::2, :]  # [batch, n_pairs, chi]
            X_odd = X[:, 1::2, :]   # [batch, n_pairs, chi]

            # Apply disentangler: u[i,j,k,l] with X_even[b,p,i], X_odd[b,p,j]
            # Result: [batch, n_pairs, chi, chi]
            X_disentangled = torch.einsum('bpi,bpj,ijkl->bpkl', X_even, X_odd, u)

            # Apply isometry: w[i,j,k] with X_disentangled[b,p,i], X_disentangled[b,p,j]
            # Result: [batch, n_pairs, chi]
            # But X_disentangled has shape [batch, n_pairs, chi, chi]
            # So we need to reshape or think of it differently
            # Let's say X_disentangled gives us new (x1', x2') pairs
            X1_new = X_disentangled[:, :, :, 0].contiguous()  # First output
            X2_new = X_disentangled[:, :, :, 1].contiguous()  # Second output

            # Actually, let me use the exact convention you specified:
            # y = einsum('bi,bj,ijk->bk', x1, x2, w)
            # Reshape for batch*pairs
            X_even_flat = X_even.reshape(-1, chi)
            X_odd_flat = X_odd.reshape(-1, chi)

            # Apply disentangler as 4D tensor: u[i,j,k,l]
            # We want new x1'[k], x2'[l] from x1[i], x2[j]
            # Contraction: x1'[k] = sum_ij x1[i] * x2[j] * u[i,j,k,l]
            #              x2'[l] = sum_ij x1[i] * x2[j] * u[i,j,k,l]
            # This gives us two outputs, so we need two separate contractions
            x1_new = torch.einsum('bi,bj,ijkl->bk', X_even_flat, X_odd_flat, u[:, :, :, 0])
            x2_new = torch.einsum('bi,bj,ijkl->bl', X_even_flat, X_odd_flat, u[:, :, 0, :])

            # Apply isometry: w[i,j,k] with convention y = einsum('bi,bj,ijk->bk', x1, x2, w)
            Y_flat = torch.einsum('bi,bj,ijk->bk', x1_new, x2_new, w)

            # Reshape back
            X = Y_flat.reshape(batch_size, n_pairs, chi)

            # Adaptive truncation
            if adaptive and self.training:
                X = self.adaptive_truncate(X, layer_idx)

            intermediates.append(X)

            if X.shape[1] == 1:
                break

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Inverse operations with tensor contractions."""
        num_layers = len(intermediates) - 1
        X = latent

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            target_len, target_chi = target.shape[1], target.shape[2]
            current_chi = X.shape[-1]

            # Pad chi if truncated
            if current_chi < target_chi:
                pad_size = target_chi - current_chi
                X = torch.nn.functional.pad(X, (0, pad_size))
                current_chi = target_chi

            # Get tensors
            u = self.disentanglers[layer_idx][:current_chi, :current_chi, :current_chi, :current_chi]
            w = self.isometries[layer_idx][:current_chi, :current_chi, :current_chi]

            # Inverse isometry: w[i,j,k] → X[b,p,i] produces X_disentangled[b,p,j,k]
            # This is the adjoint: w^†[j,k,i] = conj(w[i,j,k])
            X_disentangled = torch.einsum('ijk,bpi->bpjk', w, X)

            # Inverse disentangler: u[i,j,k,l] → X_disentangled[b,p,i,j] produces (X_even, X_odd)
            # This is the adjoint: u^†[k,l,i,j] = conj(u[i,j,k,l])
            X_even = torch.einsum('ijkl,bpij->bpk', u, X_disentangled)
            X_odd = torch.einsum('ijkl,bpij->bpl', u, X_disentangled)

            # Interleave pairs back
            batch_size, n_pairs, chi = X_even.shape
            X_reconstructed = torch.zeros(batch_size, n_pairs * 2, chi, device=X.device)
            X_reconstructed[:, 0::2, :] = X_even
            X_reconstructed[:, 1::2, :] = X_odd

            # Trim to target length
            X = X_reconstructed[:, :target_len, :]

        return X @ self.Vt_features


def compute_snr_db(original, reconstructed):
    """Compute SNR in dB."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def project_to_stiefel_4d(u):
    """Project 4D disentangler to unitary."""
    chi = u.shape[0]
    u_matrix = u.reshape(chi * chi, chi * chi).double()
    U_svd, _, Vt_svd = torch.linalg.svd(u_matrix, full_matrices=False)
    u_orth = (U_svd @ Vt_svd).float()
    return u_orth.view(chi, chi, chi, chi)


def project_to_stiefel_3d(w):
    """Project 3D isometry to Stiefel manifold."""
    chi_out = w.shape[0]
    chi_in = w.shape[1] * w.shape[2]
    w_matrix = w.reshape(chi_out, chi_in).double()
    Q, R = torch.linalg.qr(w_matrix.T)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    return Q.T.float().reshape(w.shape)


def train_joint(mera, batch, num_epochs=100, lr=1e-4):
    """Train Layer 0 and Layer 1 jointly."""
    print(f"\nJoint training (Layers 0+1) for {num_epochs} epochs...")

    # Optimize both layers
    params = []
    params.append(mera.disentanglers[0])
    params.append(mera.isometries[0])
    if len(mera.disentanglers) > 1:
        params.append(mera.disentanglers[1])
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

        # Project to manifolds every 5 steps
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                for u in mera.disentanglers:
                    u.data = project_to_stiefel_4d(u.data)
                for w in mera.isometries:
                    w.data = project_to_stiefel_3d(w.data)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                snr = compute_snr_db(batch[0], recon[0])
                print(f"{epoch+1:6d} {loss.item():12.6e} {snr:10.2f}")

                if snr > best_snr:
                    best_snr = snr

    return best_snr


def main():
    print("="*70)
    print("BINARY MERA: TENSOR STRUCTURE + HAAR-WALSH INIT")
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
    print(f"\nχ_max = {chi_max} (Haar-Walsh initialization)")

    mera = BinaryMERATensor(seq_len, hidden_dim, chi_max, 0.999).to(device)
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
    print("ADDING LAYERS WITH IDENTITY + HAAR-WALSH")
    print("="*70)

    mera.add_layer(device)
    print("  Added Layer 0 (u=Identity, w=Haar-Walsh)")

    mera.add_layer(device)
    print("  Added Layer 1 (u=Identity, w=Haar-Walsh)")

    # Test initial state
    print("\nTesting epoch 0 (before training):")
    mera.eval()
    with torch.no_grad():
        latent, intermediates = mera(activations[:1], stop_layer=2, adaptive=False)
        recon = mera.reconstruct(latent, intermediates)
        snr_init = compute_snr_db(activations[0], recon[0])
        print(f"  Initial SNR (Haar-Walsh pass-through): {snr_init:.2f} dB")

    # Joint training
    print("\n" + "="*70)
    print("JOINT OPTIMIZATION")
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
