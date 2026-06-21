"""Test hierarchical MERA on real LLM activations."""

import torch
import numpy as np
from pathlib import Path
from scipy.linalg import hadamard
from mera_hierarchical import HierarchicalMERA, train_layer0, train_remaining_layers, evaluate_all_layers, global_tuning
from mera_batch_universal import compute_snr_db


def apply_hadamard_transform(X):
    """Apply Walsh-Hadamard transform along hidden_dim axis.

    Args:
        X: [batch, seq_len, hidden_dim]

    Returns:
        X_transformed: [batch, seq_len, hidden_dim] with variance spread evenly
    """
    batch_size, seq_len, hidden_dim = X.shape

    # Generate Hadamard matrix (must be power of 2)
    # If hidden_dim is not power of 2, pad it
    n = 1
    while n < hidden_dim:
        n *= 2

    print(f"\nApplying Hadamard transform (hidden_dim {hidden_dim} → {n})...")

    # Create normalized Hadamard matrix
    H = hadamard(n, dtype=np.float32) / np.sqrt(n)
    H_tensor = torch.from_numpy(H).to(X.device, dtype=X.dtype)

    # Pad if needed
    if hidden_dim < n:
        X_padded = torch.nn.functional.pad(X, (0, n - hidden_dim))
    else:
        X_padded = X

    # Apply transform: [batch, seq_len, n] @ [n, n] = [batch, seq_len, n]
    X_transformed = X_padded @ H_tensor.T

    # Unpad
    if hidden_dim < n:
        X_transformed = X_transformed[:, :, :hidden_dim]

    return X_transformed


def load_activations(activation_type='post_attention_ln'):
    """Load saved LLM activations."""
    cache_dir = Path.home() / '.cache' / 'llm_activations'

    if activation_type == 'post_input_ln':
        file_path = cache_dir / 'activations_post_input_ln.pt'
    else:
        file_path = cache_dir / 'activations_post_attention_ln.pt'

    if not file_path.exists():
        raise FileNotFoundError(f"Activations not found at {file_path}. Run spectrum_analysis.py first.")

    print(f"Loading activations from {file_path}...")
    activations = torch.load(file_path, map_location='cpu')
    print(f"  Shape: {activations.shape}, dtype: {activations.dtype}")

    return activations


def main():
    print("="*70)
    print("TESTING MERA ON REAL LLM ACTIVATIONS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load activations
    activations = load_activations('post_attention_ln')

    # Convert to float32
    if activations.dtype == torch.bfloat16:
        activations = activations.float()

    batch_size, seq_len, hidden_dim = activations.shape
    print(f"\nOriginal shape: [{batch_size}, {seq_len}, {hidden_dim}]")

    # Reduce batch size for memory constraints during training
    batch_size = min(8, batch_size)
    activations = activations[:batch_size]
    print(f"Using batch size: {batch_size}")

    # Pad to nearest power of 3
    target_seq_len = 3 ** int(np.ceil(np.log(seq_len) / np.log(3)))
    if seq_len != target_seq_len:
        print(f"Padding sequence: {seq_len} → {target_seq_len}")
        pad_len = target_seq_len - seq_len
        activations = torch.nn.functional.pad(activations, (0, 0, 0, pad_len))
        seq_len = target_seq_len

    activations = activations.to(device)

    # Extract first head only (assuming head_dim=128, num_heads=32)
    head_dim = 128
    num_heads = hidden_dim // head_dim
    print(f"\nExtracting first head (head_dim={head_dim}, num_heads={num_heads})...")

    # Take first 128 dims
    activations = activations[:, :, :head_dim]
    hidden_dim = head_dim

    print(f"Head 0 shape: [{activations.shape[0]}, {activations.shape[1]}, {activations.shape[2]}]")

    # Analyze spectrum
    print("\n" + "="*70)
    print("ACTIVATION SPECTRUM ANALYSIS (Head 0)")
    print("="*70)

    with torch.no_grad():
        act_flat = activations.reshape(-1, hidden_dim).double()
        U, S, Vt = torch.linalg.svd(act_flat, full_matrices=False)

        energy = (S ** 2).cumsum(0) / (S ** 2).sum()
        eff_rank_95 = (energy < 0.95).sum().item() + 1

        print(f"  Singular values: {S[0].item():.4f} (max) ... {S[-1].item():.8f} (min)")
        print(f"  Effective rank (95%): {eff_rank_95}/{len(S)} ({eff_rank_95/len(S):.1%})")

    # For head_dim=128, use chi=64 to get 2x compression like synthetic data
    chi = 64
    compression_l0 = hidden_dim / chi

    print(f"\nSelected χ = {chi} ({compression_l0:.1f}x compression at Layer 0)")

    # Train MERA
    print("\n" + "="*70)
    print("TRAINING MERA")
    print("="*70)

    mera = HierarchicalMERA(seq_len, hidden_dim, chi).to(device)
    mera.initialize_uv_gate(activations)

    train_layer0(mera, activations)
    train_remaining_layers(mera, activations, max_layers=2)

    print("\n" + "="*70)
    print("BEFORE GLOBAL TUNING")
    print("="*70)
    evaluate_all_layers(mera, activations)

    global_tuning(mera, activations, num_epochs=200, lr=0.001)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    evaluate_all_layers(mera, activations)

    # Layer 1 details
    print("\n" + "="*70)
    print("LAYER 1 ANALYSIS (target: 5-10x, 30 dB)")
    print("="*70)

    mera.eval()
    with torch.no_grad():
        snrs = []
        for i in range(min(batch_size, 10)):  # Check first 10 samples
            latent, intermediates = mera(activations[i:i+1], stop_layer=1)
            recon = mera.reconstruct(latent, intermediates)
            snr = compute_snr_db(activations[i], recon[0])
            snrs.append(snr)
            print(f"  Sample {i}: {snr:.2f} dB")

        mean_snr = np.mean(snrs)
        compression = (seq_len * hidden_dim) / latent[0].numel()

        print(f"\n  Mean SNR: {mean_snr:.2f} dB")
        print(f"  Compression: {compression:.1f}x")

        if mean_snr >= 30 and 5 <= compression <= 10:
            print(f"\n  ✓ TARGET MET!")
        elif mean_snr >= 25:
            print(f"\n  ◐ Close ({30 - mean_snr:.1f} dB short)")
        else:
            print(f"\n  ✗ Below target")

    print("="*70)


if __name__ == "__main__":
    main()
