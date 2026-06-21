"""Static SVD MERA: Zero-training deterministic baseline.

This bypasses all optimization and directly uses SVD to find the optimal
principal components at each layer. If this achieves >20 dB, it proves
the data IS compressible and the issue is optimization, not structure.
"""

import torch
import numpy as np
from pathlib import Path


def compute_snr_db(original, reconstructed):
    """Compute SNR in dB."""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.item()


def static_svd_compress(X, energy_threshold=0.999):
    """Compress using deterministic SVD truncation.

    Returns the compressed representation (projections onto principal components).
    """
    # Flatten spatial dimensions
    X_flat = X.reshape(-1, X.shape[-1]).double()

    # SVD
    U, S, Vt = torch.linalg.svd(X_flat, full_matrices=False)

    # Find chi_eff based on energy
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    chi_eff = (energy < energy_threshold).sum().item() + 1
    chi_eff = min(chi_eff, X.shape[-1])

    # Project onto principal components (compressed representation)
    U_trunc = U[:, :chi_eff]
    S_trunc = S[:chi_eff]

    # Compressed = U * S (coefficients in PCA basis)
    X_compressed_flat = U_trunc * S_trunc  # [n_samples, chi_eff]
    X_compressed = X_compressed_flat.float().reshape(X.shape[0], X.shape[1], chi_eff)

    kept_energy = energy[chi_eff-1].item() if chi_eff > 0 else 0

    # Also return Vt for reconstruction
    Vt_trunc = Vt[:chi_eff, :].float()

    return X_compressed, Vt_trunc, chi_eff, kept_energy


def static_svd_mera_layer(X, energy_threshold=0.999):
    """One MERA layer with static SVD (no training).

    Structure:
    1. u = identity (disentanglers frozen)
    2. w computed deterministically from SVD of concatenated pairs
    """
    batch_size, seq_len, chi_in = X.shape
    n_pairs = seq_len // 2

    # Extract pairs
    x_even = X[:, 0::2, :]  # [batch, n_pairs, chi_in]
    x_odd = X[:, 1::2, :]   # [batch, n_pairs, chi_in]

    # Concatenate (u = identity, so no disentangling)
    x_concat = torch.cat([x_even, x_odd], dim=-1)  # [batch, n_pairs, 2*chi_in]

    # Deterministic SVD compression on concatenated data
    X_compressed, Vt_trunc, chi_eff, kept_energy = static_svd_compress(x_concat, energy_threshold)

    print(f"  χ {2*chi_in} → {chi_eff} (kept {kept_energy:.1%} energy)")

    return X_compressed, Vt_trunc


def static_svd_mera_reconstruct_layer(X_compressed, Vt_trunc, target_len, target_chi_in):
    """Reconstruct one MERA layer.

    X_compressed: [batch, n_pairs, chi_eff] - PCA coefficients
    Vt_trunc: [chi_eff, 2*target_chi_in] - PCA basis
    """
    batch_size, n_pairs, chi_eff = X_compressed.shape

    # Reconstruct concatenated pairs from PCA
    X_concat_flat = X_compressed.reshape(-1, chi_eff)  # [batch*n_pairs, chi_eff]
    X_concat_recon_flat = X_concat_flat @ Vt_trunc  # [batch*n_pairs, 2*target_chi_in]
    X_concat_recon = X_concat_recon_flat.reshape(batch_size, n_pairs, 2*target_chi_in)

    # Split back (u=identity, so just split evenly)
    x_even = X_concat_recon[:, :, :target_chi_in]
    x_odd = X_concat_recon[:, :, target_chi_in:]

    # Interleave
    X_recon = torch.zeros(batch_size, n_pairs * 2, target_chi_in, device=X_compressed.device)
    X_recon[:, 0::2, :] = x_even
    X_recon[:, 1::2, :] = x_odd

    # Trim to target length
    return X_recon[:, :target_len, :]


def main():
    print("="*70)
    print("STATIC SVD MERA: DETERMINISTIC BASELINE (ZERO TRAINING)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load activations
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()

    batch_size = 8
    activations = activations[:batch_size, :4096, :128].to(device)

    print(f"\nShape: {activations.shape}")

    # Layer 0: UV gate (PCA)
    print("\n" + "="*70)
    print("LAYER 0: UV GATE (PCA)")
    print("="*70)

    activations_flat = activations.reshape(-1, 128).double()
    activations_centered = activations_flat - activations_flat.mean(dim=0, keepdim=True)
    cov = (activations_centered.T @ activations_centered) / activations_flat.shape[0]

    S, V = torch.linalg.eigh(cov)
    S = S.flip(0)
    V = V.flip(1)

    U_features = V[:, :128].float()
    Vt_features = V[:, :128].T.float()

    X = activations @ U_features
    recon = X @ Vt_features
    snr = compute_snr_db(activations[0], recon[0])
    print(f"Layer 0 SNR: {snr:.2f} dB")

    intermediates = [X]
    vt_layers = []
    original = activations.clone()

    # Layer 1: First MERA layer with static SVD
    print("\n" + "="*70)
    print("LAYER 1: STATIC SVD MERA (u=identity, w=SVD)")
    print("="*70)

    X, Vt1 = static_svd_mera_layer(X, energy_threshold=0.999)
    intermediates.append(X)
    vt_layers.append(Vt1)

    # Reconstruct Layer 1
    print("\nReconstruction:")
    X_recon = static_svd_mera_reconstruct_layer(X, Vt1, target_len=intermediates[0].shape[1],
                                                  target_chi_in=intermediates[0].shape[2])
    recon = X_recon @ Vt_features
    snr_l1 = compute_snr_db(original[0], recon[0])
    compression_l1 = (4096 * 128) / X.numel()
    print(f"Layer 1 SNR: {snr_l1:.2f} dB at {compression_l1:.1f}x compression")

    # Layer 2: Second MERA layer with static SVD
    print("\n" + "="*70)
    print("LAYER 2: STATIC SVD MERA (u=identity, w=SVD)")
    print("="*70)

    X, Vt2 = static_svd_mera_layer(X, energy_threshold=0.999)
    intermediates.append(X)
    vt_layers.append(Vt2)

    # Reconstruct Layer 2 (two layers back)
    print("\nReconstruction:")
    X_recon = static_svd_mera_reconstruct_layer(X, Vt2, target_len=intermediates[1].shape[1],
                                                  target_chi_in=intermediates[1].shape[2])
    X_recon = static_svd_mera_reconstruct_layer(X_recon, Vt1, target_len=intermediates[0].shape[1],
                                                  target_chi_in=intermediates[0].shape[2])
    recon = X_recon @ Vt_features
    snr_l2 = compute_snr_db(original[0], recon[0])
    compression_l2 = (4096 * 128) / X.numel()
    print(f"Layer 2 SNR: {snr_l2:.2f} dB at {compression_l2:.1f}x compression")

    # Summary
    print("\n" + "="*70)
    print("STATIC SVD BASELINE RESULTS")
    print("="*70)
    marker_l1 = "✓" if snr_l1 >= 20.0 else " "
    marker_l2 = "✓" if snr_l2 >= 20.0 else " "

    print(f" ✓ L0: {snr:.2f} dB at 1.0x")
    print(f" {marker_l1} L1: {snr_l1:.2f} dB at {compression_l1:.1f}x")
    print(f" {marker_l2} L2: {snr_l2:.2f} dB at {compression_l2:.1f}x")

    if snr_l1 >= 20.0:
        print("\n" + "="*70)
        print("✓ PROOF: Data IS compressible with MERA structure!")
        print("The issue is optimization (local minima), not data structure.")
        print("Solution: Replace gradient descent with deterministic DMRG projection.")
        print("="*70)
    else:
        print("\nStatic SVD also fails - may need different compression approach.")

    print()


if __name__ == "__main__":
    main()
