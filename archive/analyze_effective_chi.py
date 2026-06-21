"""Analyze what effective chi is needed at each layer to preserve energy."""

import torch
import numpy as np
import pywt
from pathlib import Path
from mera_hierarchical import HierarchicalMERA, train_layer0, train_remaining_layers
from mera_batch_universal import compute_snr_db


def apply_wavelet_along_sequence(X, wavelet="db2"):
    """Apply 1D wavelet transform along sequence dimension.

    Args:
        X: [batch, seq_len, hidden_dim]

    Returns:
        X_wavelet: [batch, seq_len, hidden_dim] (same shape, periodic mode)
    """
    batch_size, seq_len, hidden_dim = X.shape
    X_wavelet = torch.zeros_like(X)

    for b in range(batch_size):
        for h in range(hidden_dim):
            # Extract sequence for this feature
            seq = X[b, :, h].cpu().numpy()

            # Single-level DWT with periodic mode
            cA, cD = pywt.dwt(seq, wavelet, mode='periodic')

            # Concatenate coefficients
            coeffs = np.concatenate([cA, cD])

            # Trim or pad to match original length
            if len(coeffs) > seq_len:
                coeffs = coeffs[:seq_len]
            elif len(coeffs) < seq_len:
                coeffs = np.pad(coeffs, (0, seq_len - len(coeffs)))

            X_wavelet[b, :, h] = torch.from_numpy(coeffs)

    return X_wavelet


def analyze_layer_spectrum(mera, activations, energy_threshold=0.999):
    """Analyze singular value spectrum at each layer to find effective chi."""
    print("\n" + "="*70)
    print("ANALYZING EFFECTIVE CHI AT EACH LAYER")
    print(f"Energy threshold: {energy_threshold:.1%}")
    print("="*70)

    mera.eval()
    with torch.no_grad():
        # Forward through all layers
        latent, intermediates = mera(activations)

        print(f"\n{'Layer':>6} {'Shape':>20} {'Full Chi':>10} {'Eff Chi':>10} {'Ratio':>8} {'Energy':>8}")
        print("-" * 80)

        for layer_idx, X in enumerate(intermediates):
            # Reshape to [batch*seq, chi]
            X_flat = X.reshape(-1, X.shape[-1]).double()

            # SVD
            U, S, Vt = torch.linalg.svd(X_flat, full_matrices=False)

            # Cumulative energy
            energy = (S ** 2).cumsum(0) / (S ** 2).sum()

            # Effective chi for energy threshold
            chi_eff = (energy < energy_threshold).sum().item() + 1
            chi_eff = min(chi_eff, len(S))

            full_chi = X.shape[-1]
            ratio = chi_eff / full_chi

            print(f"   L{layer_idx} {str(list(X[0].shape)):>20} {full_chi:10d} {chi_eff:10d} "
                  f"{ratio:7.1%} {energy[chi_eff-1].item():7.1%}")

        print("\n" + "="*70)


def main():
    print("="*70)
    print("EFFECTIVE CHI ANALYSIS")
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

    # Pad sequence
    seq_len = activations.shape[1]
    target_seq_len = 3 ** int(np.ceil(np.log(seq_len) / np.log(3)))
    if seq_len != target_seq_len:
        pad_len = target_seq_len - seq_len
        activations = torch.nn.functional.pad(activations, (0, 0, 0, pad_len))

    activations = activations.to(device)
    _, seq_len, hidden_dim = activations.shape

    # Extract first head only (128 dims)
    head_dim = 128
    print(f"\nExtracting head 0 (first {head_dim} dims)...")
    activations = activations[:, :, :head_dim]
    hidden_dim = head_dim

    # Apply wavelet transform along sequence
    print(f"Applying db2 wavelet along sequence dimension...")
    activations = apply_wavelet_along_sequence(activations, wavelet="db2")

    print(f"\nFinal shape: [{batch_size}, {seq_len}, {hidden_dim}]")

    # Use chi=64 for head_dim=128
    chi = 64
    print(f"\nTraining with χ = {chi} (2x compression at Layer 0)")

    mera = HierarchicalMERA(seq_len, hidden_dim, chi).to(device)
    mera.initialize_uv_gate(activations)

    # Train
    train_layer0(mera, activations)
    train_remaining_layers(mera, activations, max_layers=2)

    # Analyze effective chi at each layer
    for threshold in [0.99, 0.995, 0.999]:
        analyze_layer_spectrum(mera, activations, energy_threshold=threshold)

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nEffective chi shows how much the bond dimension can be reduced")
    print("at each layer while preserving the specified energy fraction.")
    print("This is analogous to DMRG truncation based on discarded weight.")
    print("="*70)


if __name__ == "__main__":
    main()
