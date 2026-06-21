"""Sweep chi - CORRECT compression calculation (data only, not operators)."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mera_stage1_direct import DirectMERA, generate_powerlaw_matrix, compute_snr_db


def compute_data_compression(latent_shape, seq_len, hidden_dim):
    """Compute compression of the ACTIVATIONS (not counting operators)."""
    original_data_size = seq_len * hidden_dim
    compressed_data_size = np.prod(latent_shape)  # Usually [1, 1] or [1, final_chi]

    compression_ratio = original_data_size / compressed_data_size

    return compression_ratio, compressed_data_size, original_data_size


def main():
    """Sweep chi values with CORRECT compression metric."""
    print("="*70)
    print("MERA CHI SWEEP: Data Compression (Correct Calculation)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Config
    seq_len = 3 ** 7  # 2187
    hidden_dim = 512
    alpha = 1.1

    # Generate test data once
    print(f"\nGenerating test data: [{seq_len}, {hidden_dim}], σ_k = k^(-{alpha})")
    A_mock = generate_powerlaw_matrix(seq_len, hidden_dim, alpha=alpha).to(device)

    original_size = seq_len * hidden_dim
    print(f"Original activation size: {original_size:,} values ({original_size * 4 / 1024:.1f} KB in float32)")

    # Check effective rank
    U, S, Vt = torch.linalg.svd(A_mock.double().cpu(), full_matrices=False)
    eff_rank = (S.sum() ** 2) / (S ** 2).sum()
    print(f"Effective rank: {eff_rank.item():.1f} / {len(S)}")

    # Sweep chi values
    chi_values = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]

    results = []

    print(f"\n{'Chi':>4} {'SNR (dB)':>10} {'Latent':>15} {'Data Compress':>15} {'UV Energy':>10}")
    print("-" * 80)

    for chi in chi_values:
        # Construct MERA (suppress layer output)
        mera = DirectMERA(seq_len, hidden_dim, chi)

        # Monkey-patch to suppress print statements during construction
        import sys
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        latent = mera.construct_from_data(A_mock)

        sys.stdout = old_stdout

        # Compress and reconstruct
        latent_compress, intermediates = mera.compress(A_mock)
        A_recon = mera.reconstruct(latent_compress, intermediates)

        # Metrics
        snr = compute_snr_db(A_mock, A_recon)
        latent_shape = latent_compress.shape
        comp_ratio, compressed_size, orig_size = compute_data_compression(
            latent_shape, seq_len, hidden_dim
        )

        # UV layer energy retention
        U_feat, S_feat, Vt_feat = torch.linalg.svd(A_mock.double(), full_matrices=False)
        uv_energy = 100 * (S_feat[:chi]**2).sum() / (S_feat**2).sum()

        results.append({
            'chi': chi,
            'snr': snr,
            'compression': comp_ratio,
            'latent_shape': latent_shape,
            'compressed_size': compressed_size,
            'uv_energy': uv_energy.item()
        })

        latent_str = f"{list(latent_shape)}"
        print(f"{chi:4d} {snr:10.2f} {latent_str:>15} {comp_ratio:14.1f}x {uv_energy:9.2f}%")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    chi_vals = [r['chi'] for r in results]
    snr_vals = [r['snr'] for r in results]
    comp_vals = [r['compression'] for r in results]

    # SNR vs Chi
    axes[0, 0].plot(chi_vals, snr_vals, 'o-', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=30, color='r', linestyle='--', label='30 dB target')
    axes[0, 0].set_xlabel('Bond Dimension (χ)', fontsize=12)
    axes[0, 0].set_ylabel('Reconstruction SNR (dB)', fontsize=12)
    axes[0, 0].set_title('SNR vs Bond Dimension', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Compression vs Chi
    axes[0, 1].plot(chi_vals, comp_vals, 's-', linewidth=2, markersize=8, color='green')
    axes[0, 1].axhline(y=5, color='orange', linestyle='--', label='5x target', linewidth=2)
    axes[0, 1].axhline(y=10, color='purple', linestyle='--', label='10x target', linewidth=2)
    axes[0, 1].set_xlabel('Bond Dimension (χ)', fontsize=12)
    axes[0, 1].set_ylabel('Data Compression Ratio', fontsize=12)
    axes[0, 1].set_title('Data Compression vs Bond Dimension', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')

    # SNR vs Compression tradeoff
    axes[1, 0].plot(comp_vals, snr_vals, 'D-', linewidth=2, markersize=8, color='purple')
    axes[1, 0].axhline(y=30, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=5, color='orange', linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=10, color='purple', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Data Compression Ratio', fontsize=12)
    axes[1, 0].set_ylabel('SNR (dB)', fontsize=12)
    axes[1, 0].set_title('SNR vs Compression Tradeoff', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')

    # Annotate points with chi values
    for r in results[::2]:  # Every other point to avoid clutter
        axes[1, 0].annotate(f"χ={r['chi']}",
                           (r['compression'], r['snr']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)

    # Compressed size vs chi
    compressed_sizes = [r['compressed_size'] for r in results]
    axes[1, 1].plot(chi_vals, compressed_sizes, '^-', linewidth=2, markersize=8, color='red')
    axes[1, 1].set_xlabel('Bond Dimension (χ)', fontsize=12)
    axes[1, 1].set_ylabel('Compressed Latent Size (# values)', fontsize=12)
    axes[1, 1].set_title('Latent Size vs Bond Dimension', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    save_path = '/home/brian-dellabetta/projects/llm-compressor/mera_chi_sweep_correct.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")

    # Find optimal points
    print("\n" + "="*70)
    print("OPTIMAL CONFIGURATIONS")
    print("="*70)

    # Best SNR
    best_snr = max(results, key=lambda x: x['snr'])
    print(f"\nBest SNR: χ={best_snr['chi']}")
    print(f"  SNR: {best_snr['snr']:.2f} dB")
    print(f"  Compression: {best_snr['compression']:.1f}x")
    print(f"  Latent shape: {list(best_snr['latent_shape'])}")

    # Best compression
    best_comp = max(results, key=lambda x: x['compression'])
    print(f"\nBest Compression: χ={best_comp['chi']}")
    print(f"  Compression: {best_comp['compression']:.1f}x")
    print(f"  SNR: {best_comp['snr']:.2f} dB")
    print(f"  Latent shape: {list(best_comp['latent_shape'])}")

    # Meets both thresholds?
    good_configs_5x = [r for r in results if r['snr'] >= 30 and r['compression'] >= 5]
    good_configs_10x = [r for r in results if r['snr'] >= 30 and r['compression'] >= 10]

    if good_configs_10x:
        print(f"\n✓ Configurations meeting SNR≥30dB AND Compression≥10x:")
        for cfg in good_configs_10x:
            print(f"  χ={cfg['chi']}: {cfg['snr']:.2f} dB, {cfg['compression']:.1f}x, latent={list(cfg['latent_shape'])}")
    elif good_configs_5x:
        print(f"\n✓ Configurations meeting SNR≥30dB AND Compression≥5x:")
        for cfg in good_configs_5x:
            print(f"  χ={cfg['chi']}: {cfg['snr']:.2f} dB, {cfg['compression']:.1f}x, latent={list(cfg['latent_shape'])}")
    else:
        print(f"\nNo configuration meets SNR≥30dB AND Compression≥5x")

    print(f"\nAll tradeoffs (sorted by chi):")
    for r in results:
        marker = "✓" if r['snr'] >= 30 else " "
        comp_marker = "✓" if r['compression'] >= 5 else " "
        print(f"  {marker} χ={r['chi']:3d}: {r['snr']:5.2f} dB, {r['compression']:8.1f}x {comp_marker}, "
              f"latent={list(r['latent_shape'])}, {r['compressed_size']} values")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
