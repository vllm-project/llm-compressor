"""MERA with gradient-based disentanglers and SVD isometries."""

import torch
import torch.nn as nn
import numpy as np


class GradientMERA(nn.Module):
    """MERA with learned disentanglers and SVD isometries.

    - u (disentangler): Learned via gradient descent (Stiefel manifold projection)
    - w (isometry): Deterministic SVD truncation
    - Both are SHARED across all positions in a layer
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

        # Trainable disentanglers (one per layer)
        self.u_layers = nn.ModuleList()

        # Fixed isometry bases (learned via SVD, not trained)
        self.w_bases = []
        self.layer_chis = []

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

    def initialize_disentanglers(self, num_layers=2, chi_start=128):
        """Initialize u matrices (identity to start).

        Args:
            num_layers: Number of MERA layers
            chi_start: Starting chi dimension (from UV gate)
        """
        self.u_layers = nn.ModuleList()

        # Layer 0: concatenate [chi_start] + [chi_start] = [2*chi_start]
        u0 = nn.Parameter(torch.eye(2 * chi_start, device=self.U_features.device))
        self.u_layers.append(nn.ParameterList([u0]))

        # Layer 1: depends on Layer 0 output chi (will be set during learn_isometries)
        # For now, just create placeholder - will be resized if needed
        if num_layers > 1:
            # Assume similar size for now, will adjust during learn_isometries
            u1 = nn.Parameter(torch.eye(2 * chi_start, device=self.U_features.device))
            self.u_layers.append(nn.ParameterList([u1]))

    def project_to_stiefel(self):
        """Project u matrices to Stiefel manifold (orthogonal constraint)."""
        with torch.no_grad():
            for u_module in self.u_layers:
                u = u_module[0]
                # QR decomposition gives nearest orthogonal matrix
                Q, R = torch.linalg.qr(u.double())
                u_module[0].data = Q.float()

    def learn_isometries(self, batch, verbose=True):
        """Learn w bases via SVD (deterministic, not trained)."""
        self.w_bases = []
        self.layer_chis = []

        X = batch @ self.U_features

        for layer_idx, u_module in enumerate(self.u_layers):
            if X.shape[1] < 2:
                break

            batch_size, seq_len, chi_in = X.shape
            n_pairs = seq_len // 2

            # Extract and concatenate
            x_even = X[:, 0::2, :]
            x_odd = X[:, 1::2, :]
            x_concat = torch.cat([x_even, x_odd], dim=-1)  # [batch, n_pairs, 2*chi_in]

            # Resize u if needed
            u = u_module[0]
            if u.shape[0] != 2 * chi_in:
                u_module[0] = nn.Parameter(torch.eye(2 * chi_in, device=batch.device))
                u = u_module[0]

            # Apply disentangler
            x_flat = x_concat.reshape(-1, 2 * chi_in)
            x_disentangled = x_flat @ u.T

            # SVD to find isometry w
            U_w, S_w, Vt_w = torch.linalg.svd(x_disentangled.double(), full_matrices=False)

            # Energy-based chi
            energy = (S_w ** 2).cumsum(0) / (S_w ** 2).sum()
            chi_eff = (energy < self.energy_threshold).sum().item() + 1
            chi_eff = min(chi_eff, 2 * chi_in)

            Vt_trunc = Vt_w[:chi_eff, :].float()
            self.w_bases.append(Vt_trunc)
            self.layer_chis.append(chi_eff)

            # Compress for next layer
            X_compressed = (U_w[:, :chi_eff] * S_w[:chi_eff]).float()
            X = X_compressed.reshape(batch_size, n_pairs, chi_eff)

            if verbose:
                kept_energy = energy[chi_eff-1].item() if chi_eff > 0 else 0
                print(f"  Layer {layer_idx}: χ {2*chi_in} → {chi_eff} (kept {kept_energy:.1%} energy)")

    def forward(self, batch):
        """Encode batch using learned u and w."""
        X = batch @ self.U_features
        intermediates = [X]

        for layer_idx, u_module in enumerate(self.u_layers):
            if X.shape[1] < 2:
                break

            batch_size, seq_len, chi_in = X.shape
            n_pairs = seq_len // 2

            # Extract and concatenate
            x_even = X[:, 0::2, :]
            x_odd = X[:, 1::2, :]
            x_concat = torch.cat([x_even, x_odd], dim=-1)

            # Apply disentangler
            u = u_module[0]
            x_flat = x_concat.reshape(-1, 2 * chi_in)
            x_disentangled = x_flat @ u.T

            # Apply isometry
            Vt_w = self.w_bases[layer_idx]
            chi_eff = self.layer_chis[layer_idx]

            U_w, S_w, _ = torch.linalg.svd(x_disentangled.double(), full_matrices=False)
            X_compressed = (U_w[:, :chi_eff] * S_w[:chi_eff]).float()
            X = X_compressed.reshape(batch_size, n_pairs, chi_eff)

            intermediates.append(X)

        return X, intermediates

    def reconstruct(self, latent, intermediates):
        """Reconstruct from latent."""
        X = latent
        num_layers = len(self.w_bases)

        for layer_idx in range(num_layers - 1, -1, -1):
            target = intermediates[layer_idx]
            batch_size, n_pairs, chi_eff = X.shape

            Vt_w = self.w_bases[layer_idx]
            u = self.u_layers[layer_idx][0]
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


def train_mera(mera, train_batch, num_epochs=100, lr=0.01, verbose=True):
    """Train disentanglers via gradient descent."""
    optimizer = torch.optim.Adam(mera.parameters(), lr=lr)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        latent, intermediates = mera.forward(train_batch)
        recon = mera.reconstruct(latent, intermediates)

        # Reconstruction loss
        loss = torch.mean((train_batch - recon) ** 2)

        # Backward
        loss.backward()
        optimizer.step()

        # Project to Stiefel manifold
        mera.project_to_stiefel()

        if verbose and (epoch + 1) % 20 == 0:
            snr = compute_snr_db(train_batch[0], recon[0])
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.6f}, SNR={snr:.2f} dB")

    # Final SNR
    with torch.no_grad():
        latent, intermediates = mera.forward(train_batch)
        recon = mera.reconstruct(latent, intermediates)
        final_snr = compute_snr_db(train_batch[0], recon[0])

    return final_snr


if __name__ == "__main__":
    from pathlib import Path

    print("="*90)
    print("GRADIENT-BASED DISENTANGLER LEARNING")
    print("="*90)
    print("\nShared u (gradient), shared w (SVD), test on held-out samples\n")

    device = torch.device('cuda')
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
    activations = activations[:, :4096, :128].to(device)

    train_size = 256
    train_batch = activations[:train_size]
    test_batch = activations[train_size:train_size+64]

    print(f"Training on {train_size} samples...")

    mera = GradientMERA(4096, 128, 128, 0.999).to(device)
    mera.initialize_uv_gate(train_batch)
    mera.initialize_disentanglers(num_layers=2, chi_start=128)

    print("\nLearning isometries (w) via SVD...")
    with torch.no_grad():
        mera.learn_isometries(train_batch, verbose=True)

    print("\nTraining disentanglers (u) via gradient descent...")
    final_train_snr = train_mera(mera, train_batch, num_epochs=100, lr=0.01, verbose=True)

    print(f"\nFinal training SNR: {final_train_snr:.2f} dB")

    # Test on held-out
    print(f"\nTesting on {len(test_batch)} held-out samples...")
    with torch.no_grad():
        test_snrs = []
        test_compressions = []

        for i in range(len(test_batch)):
            sample = test_batch[i:i+1]
            latent, intermediates = mera.forward(sample)
            recon = mera.reconstruct(latent, intermediates)

            snr = compute_snr_db(sample[0], recon[0])
            compression = (4096 * 128) / latent[0].numel()

            test_snrs.append(snr)
            test_compressions.append(compression)

    print("\n" + "="*90)
    print("RESULTS ON HELD-OUT SAMPLES")
    print("="*90)
    print(f"Mean SNR: {np.mean(test_snrs):.2f} dB")
    print(f"SNR std: {np.std(test_snrs):.2f} dB")
    print(f"Mean compression: {np.mean(test_compressions):.2f}x")
    print(f"Layer chis: {mera.layer_chis}")
    print("="*90)
