"""Construct universal MERA from batch average/PCA."""

import torch
import numpy as np


def compute_snr_db(original, reconstructed):
    """Compute SNR in decibels."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


class DirectMERA:
    """Placeholder - not used in hierarchical version."""
    pass


def generate_powerlaw_batch(batch_size, seq_len, hidden_dim, alpha=1.1, seed=42):
    """Generate batch of matrices with same power-law spectrum but different random bases."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    rank = min(seq_len, hidden_dim)
    k = torch.arange(1, rank + 1, dtype=torch.float64)
    singular_values = k ** (-alpha)

    batch = []
    for i in range(batch_size):
        U, _ = torch.linalg.qr(torch.randn(seq_len, rank, dtype=torch.float64))
        V, _ = torch.linalg.qr(torch.randn(hidden_dim, rank, dtype=torch.float64))

        A = U @ torch.diag(singular_values) @ V.T
        batch.append(A.float())

    return torch.stack(batch, dim=0)


def construct_universal_mera(batch, chi):
    """Construct MERA from batch statistics (PCA across batch).

    Args:
        batch: [batch_size, seq_len, hidden_dim]
        chi: bond dimension

    Returns:
        mera: DirectMERA instance
    """
    batch_size, seq_len, hidden_dim = batch.shape

    # Flatten batch to [batch_size * seq_len, hidden_dim]
    batch_flat = batch.reshape(-1, hidden_dim)

    print(f"  Computing PCA on batch: {list(batch_flat.shape)}")

    # Compute covariance across all samples
    # This finds the universal feature directions
    batch_centered = batch_flat - batch_flat.mean(dim=0, keepdim=True)
    cov = (batch_centered.T @ batch_centered) / batch_flat.shape[0]

    # Eigen-decomposition to get principal directions
    S, V = torch.linalg.eigh(cov.double())

    # Largest eigenvalues first
    S = S.flip(0)
    V = V.flip(1)

    # Now construct MERA using average sample
    print(f"  Constructing MERA from batch average...")
    batch_mean = batch.mean(dim=0)  # [seq_len, hidden_dim]

    mera = DirectMERA(seq_len, hidden_dim, chi)

    # Override UV gate with PCA basis
    mera.U_features = V[:, :chi].T.T.float()  # [hidden_dim, chi]
    mera.Vt_features = V[:, :chi].T.float()   # [chi, hidden_dim]

    # Project average sample to build coarsening operators
    X_avg = batch_mean @ mera.U_features

    # Build coarsening ops from average
    import sys
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    current = X_avg
    for layer_idx in range(mera.num_layers):
        seq_len_current, feat_dim = current.shape

        n_groups = seq_len_current // 3
        current_trimmed = current[:n_groups * 3]

        triplets = current_trimmed.reshape(n_groups, 3 * feat_dim)

        # SVD
        U_local, S_local, Vt_local = torch.linalg.svd(triplets.double(), full_matrices=False)

        actual_chi = min(chi, len(S_local))
        W_coarsen = Vt_local[:actual_chi, :].T.float()
        W_expand = Vt_local[:actual_chi, :].float()

        mera.coarsen_ops.append({'encode': W_coarsen, 'decode': W_expand, 'feat_dim': feat_dim})

        current = triplets @ W_coarsen

        if n_groups == 1:
            break

    sys.stdout = old_stdout

    return mera


def main():
    """Test universal MERA."""
    print("="*70)
    print("UNIVERSAL MERA FROM BATCH")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Config
    batch_size = 64
    seq_len = 3 ** 6  # 729 (smaller for faster test)
    hidden_dim = 512
    alpha = 1.1
    chi = 128

    print(f"\nGenerating batch of {batch_size} power-law matrices...")
    print(f"  Shape per sample: [{seq_len}, {hidden_dim}]")

    batch = generate_powerlaw_batch(batch_size, seq_len, hidden_dim, alpha=alpha).to(device)

    # Construct universal MERA
    print(f"\nConstructing universal MERA (χ={chi})...")
    mera = construct_universal_mera(batch, chi)
    print(f"  Built {len(mera.coarsen_ops)} coarsening layers")

    # Test on batch
    print(f"\nTesting on batch...")
    snr_values = []

    for i in range(batch_size):
        sample = batch[i]
        latent, intermediates = mera.compress(sample)
        recon = mera.reconstruct(latent, intermediates)

        snr = compute_snr_db(sample, recon)
        snr_values.append(snr)

    # Statistics
    snr_mean = np.mean(snr_values)
    snr_std = np.std(snr_values)
    snr_min = np.min(snr_values)
    snr_max = np.max(snr_values)

    # Compression
    original_size = seq_len * hidden_dim
    latent_size = np.prod(latent.shape)
    compression = original_size / latent_size

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nCompression: {compression:.1f}x")
    print(f"Latent shape: {list(latent.shape)}")
    print(f"\nSNR Statistics (dB):")
    print(f"  Mean:  {snr_mean:.2f} ± {snr_std:.2f}")
    print(f"  Min:   {snr_min:.2f}")
    print(f"  Max:   {snr_max:.2f}")

    if snr_mean >= 20:
        print(f"\n✓ Good generalization (mean SNR ≥ 20 dB)")
    else:
        print(f"\n✗ Poor generalization (mean SNR < 20 dB)")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
