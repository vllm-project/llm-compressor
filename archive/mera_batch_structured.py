"""Test MERA on batch with shared structure (like real activations)."""

import torch
import numpy as np
from mera_batch_universal import construct_universal_mera, compute_snr_db


def generate_structured_batch(batch_size, seq_len, hidden_dim, alpha=1.1,
                              structure_ratio=0.8, seed=42):
    """Generate batch with SHARED structure + per-sample variations.

    This mimics real LLM activations where:
    - All samples share the same feature space (U, V)
    - Each sample has different activations in that space

    Args:
        batch_size: number of samples
        seq_len, hidden_dim: matrix dimensions
        alpha: power law exponent for singular values
        structure_ratio: how much structure is shared (0=random, 1=identical basis)
        seed: random seed

    Returns:
        batch: [batch_size, seq_len, hidden_dim]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    rank = min(seq_len, hidden_dim)

    # Shared basis (the "universal feature space")
    U_shared, _ = torch.linalg.qr(torch.randn(seq_len, rank, dtype=torch.float64))
    V_shared, _ = torch.linalg.qr(torch.randn(hidden_dim, rank, dtype=torch.float64))

    batch = []
    for i in range(batch_size):
        # Sample-specific: different activation magnitudes in the shared space
        # Each sample has power-law spectrum but different random coefficients
        k = torch.arange(1, rank + 1, dtype=torch.float64)
        base_singular_values = k ** (-alpha)

        # Add random variation to singular values (but keep power-law structure)
        noise = torch.randn(rank, dtype=torch.float64) * 0.1
        singular_values = base_singular_values * torch.exp(noise)

        # Option 1: Pure shared basis with varied singular values
        if structure_ratio >= 0.99:
            U_sample = U_shared
            V_sample = V_shared
        else:
            # Option 2: Shared basis + small random rotation
            # This gives some per-sample variation while keeping most structure
            U_noise, _ = torch.linalg.qr(torch.randn(seq_len, rank, dtype=torch.float64))
            V_noise, _ = torch.linalg.qr(torch.randn(hidden_dim, rank, dtype=torch.float64))

            # Blend: mostly shared, some noise
            U_sample = structure_ratio * U_shared + (1 - structure_ratio) * U_noise
            V_sample = structure_ratio * V_shared + (1 - structure_ratio) * V_noise

            # Re-orthogonalize
            U_sample, _ = torch.linalg.qr(U_sample)
            V_sample, _ = torch.linalg.qr(V_sample)

        # Construct sample
        A = U_sample @ torch.diag(singular_values) @ V_sample.T
        batch.append(A.float())

    return torch.stack(batch, dim=0)


def main():
    """Test universal MERA on structured batch."""
    print("="*70)
    print("UNIVERSAL MERA ON STRUCTURED BATCH")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Config
    batch_size = 64
    seq_len = 3 ** 6  # 729
    hidden_dim = 512
    alpha = 1.1

    # Test different structure ratios
    structure_ratios = [0.0, 0.5, 0.8, 0.95, 1.0]
    chi_values = [64, 128, 256]

    print(f"\nBatch: {batch_size} samples of [{seq_len}, {hidden_dim}]")
    print(f"Power law: σ_k = k^(-{alpha})")

    results = []

    print(f"\n{'Structure':>10} {'Chi':>5} {'Compression':>12} {'Mean SNR':>10} {'Std SNR':>9} {'Min SNR':>9} {'Max SNR':>9}")
    print("-" * 85)

    for structure_ratio in structure_ratios:
        # Generate batch with this level of shared structure
        batch = generate_structured_batch(
            batch_size, seq_len, hidden_dim, alpha=alpha,
            structure_ratio=structure_ratio, seed=42
        ).to(device)

        for chi in chi_values:
            # Construct universal MERA
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            mera = construct_universal_mera(batch, chi)

            sys.stdout = old_stdout

            # Test on batch
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

            results.append({
                'structure': structure_ratio,
                'chi': chi,
                'compression': compression,
                'snr_mean': snr_mean,
                'snr_std': snr_std,
                'snr_min': snr_min,
                'snr_max': snr_max
            })

            print(f"{structure_ratio:10.2f} {chi:5d} {compression:11.1f}x "
                  f"{snr_mean:10.2f} {snr_std:9.2f} {snr_min:9.2f} {snr_max:9.2f}")

    # Summary
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Best result for each structure level
    print("\nBest SNR for each structure ratio (chi=128):")
    for structure_ratio in structure_ratios:
        matching = [r for r in results if r['structure'] == structure_ratio and r['chi'] == 128]
        if matching:
            r = matching[0]
            status = "✓" if r['snr_mean'] >= 20 else "✗"
            print(f"  {status} Structure={structure_ratio:.2f}: {r['snr_mean']:6.2f} dB (± {r['snr_std']:.2f})")

    # Find minimum structure needed for good SNR
    good_results = [r for r in results if r['snr_mean'] >= 20]
    if good_results:
        min_structure = min(r['structure'] for r in good_results)
        print(f"\n✓ Minimum structure ratio for SNR≥20dB: {min_structure:.2f}")
        print(f"  (Real LLM activations likely have ~0.8-0.95 shared structure)")
    else:
        print(f"\n✗ No configuration achieved SNR≥20dB")
        print(f"  May need higher chi or more shared structure")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
