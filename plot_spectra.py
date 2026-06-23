"""Plot singular value spectra for each MERA layer."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mera_svd_disentangler import SVDDisentanglerMERA

device = torch.device('cuda')
cache_dir = Path.home() / '.cache' / 'llm_activations'
activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
activations = activations[:256, :4096, :128].to(device)

print("Building MERA and extracting singular values at each layer...")

# Build MERA with 3 layers
mera = SVDDisentanglerMERA(4096, 128, 128, 0.999).to(device)
mera.initialize_uv_gate(activations)

# We'll capture singular values during build
layer_sv_u = []  # Singular values from disentangler (eigenvalues of covariance)
layer_sv_w = []  # Singular values from isometry

# Manually build layers to capture singular values
X = activations @ mera.U_features

for layer_idx in range(3):
    if X.shape[1] < 2:
        break

    batch_size, seq_len, chi_in = X.shape
    n_pairs = seq_len // 2

    # Extract pairs
    x_even = X[:, 0::2, :]
    x_odd = X[:, 1::2, :]
    x_concat = torch.cat([x_even, x_odd], dim=-1)

    # Disentangler u (from covariance)
    x_flat = x_concat.reshape(-1, 2 * chi_in).double()
    x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    cov = (x_centered.T @ x_centered) / x_flat.shape[0]

    S_u, V_u = torch.linalg.eigh(cov)
    S_u = S_u.flip(0)  # Largest first
    V_u = V_u.flip(1)

    u = V_u.T.float()
    layer_sv_u.append(S_u.cpu().numpy())

    # Apply disentangler
    x_disentangled = x_flat.float() @ u.T

    # Isometry w (from SVD)
    U_w, S_w, Vt_w = torch.linalg.svd(x_disentangled.double(), full_matrices=False)
    layer_sv_w.append(S_w.cpu().numpy())

    # Compress for next layer
    energy = (S_w ** 2).cumsum(0) / (S_w ** 2).sum()
    chi_eff = (energy < 0.999).sum().item() + 1
    chi_eff = min(chi_eff, 2 * chi_in)

    X_compressed = (U_w[:, :chi_eff] * S_w[:chi_eff]).float()
    X = X_compressed.reshape(batch_size, n_pairs, chi_eff)

    print(f"Layer {layer_idx}: chi_in={chi_in}, chi_out={chi_eff}, "
          f"#singular_values={len(S_w)}")

# Create plots
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Singular Value Spectra: Disentangler (u) and Isometry (w) per Layer',
             fontsize=14, fontweight='bold')

for layer_idx in range(3):
    # Normalize singular values
    sv_u = layer_sv_u[layer_idx]
    sv_w = layer_sv_w[layer_idx]

    sv_u_norm = sv_u / sv_u[0]  # Normalize by largest
    sv_w_norm = sv_w / sv_w[0]

    # Plot disentangler u (left column)
    ax_u = axes[layer_idx, 0]
    ax_u.semilogy(sv_u_norm, 'b-', linewidth=2, label='Normalized eigenvalues')
    ax_u.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax_u.axhline(y=0.001, color='orange', linestyle='--', alpha=0.5, label='0.1% threshold')
    ax_u.set_xlabel('Index', fontsize=10)
    ax_u.set_ylabel('Normalized Eigenvalue', fontsize=10)
    ax_u.set_title(f'Layer {layer_idx}: Disentangler u (covariance eigenvalues)',
                   fontsize=11, fontweight='bold')
    ax_u.grid(True, alpha=0.3)
    ax_u.legend(fontsize=8)

    # Add decay annotation
    # Fit power law: y = x^(-alpha)
    indices = np.arange(10, min(100, len(sv_u_norm)))
    if len(indices) > 10:
        log_idx = np.log(indices)
        log_sv = np.log(sv_u_norm[indices] + 1e-10)
        alpha_u = -np.polyfit(log_idx, log_sv, 1)[0]
        ax_u.text(0.98, 0.05, f'Power law: α≈{alpha_u:.2f}',
                 transform=ax_u.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=9)

    # Plot isometry w (right column)
    ax_w = axes[layer_idx, 1]
    ax_w.semilogy(sv_w_norm, 'g-', linewidth=2, label='Normalized singular values')
    ax_w.axhline(y=0.01, color='r', linestyle='--', alpha=0.5, label='1% threshold')
    ax_w.axhline(y=0.001, color='orange', linestyle='--', alpha=0.5, label='0.1% threshold')

    # Mark 99.9% energy cutoff
    energy = (sv_w ** 2).cumsum() / (sv_w ** 2).sum()
    chi_999 = (energy < 0.999).sum() + 1
    ax_w.axvline(x=chi_999, color='purple', linestyle=':', linewidth=2,
                 label=f'99.9% energy (χ={chi_999})')

    ax_w.set_xlabel('Singular Value Index', fontsize=10)
    ax_w.set_ylabel('Normalized Singular Value', fontsize=10)
    ax_w.set_title(f'Layer {layer_idx}: Isometry w (SVD singular values)',
                   fontsize=11, fontweight='bold')
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(fontsize=8)

    # Fit power law
    indices = np.arange(10, min(100, len(sv_w_norm)))
    if len(indices) > 10:
        log_idx = np.log(indices)
        log_sv = np.log(sv_w_norm[indices] + 1e-10)
        alpha_w = -np.polyfit(log_idx, log_sv, 1)[0]
        ax_w.text(0.98, 0.05, f'Power law: α≈{alpha_w:.2f}',
                 transform=ax_w.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                 fontsize=9)

plt.tight_layout()
plt.savefig('mera_spectra.png', dpi=300, bbox_inches='tight')
print("\nSaved plot to mera_spectra.png")

# Print summary statistics
print("\n" + "="*80)
print("SINGULAR VALUE DECAY ANALYSIS")
print("="*80)

for layer_idx in range(3):
    sv_u = layer_sv_u[layer_idx]
    sv_w = layer_sv_w[layer_idx]

    # Energy captured by top k
    sv_u_norm = sv_u / sv_u.sum()
    sv_w_sq = sv_w ** 2
    energy_w = sv_w_sq.cumsum() / sv_w_sq.sum()

    print(f"\nLayer {layer_idx}:")
    print(f"  Disentangler u:")
    print(f"    Total dimensions: {len(sv_u)}")
    print(f"    Top 10% contain: {sv_u[:len(sv_u)//10].sum() / sv_u.sum() * 100:.1f}% of trace")
    print(f"    Top 50% contain: {sv_u[:len(sv_u)//2].sum() / sv_u.sum() * 100:.1f}% of trace")

    print(f"  Isometry w:")
    print(f"    Total dimensions: {len(sv_w)}")
    chi_90 = (energy_w < 0.90).sum()
    chi_99 = (energy_w < 0.99).sum()
    chi_999 = (energy_w < 0.999).sum()
    print(f"    90% energy: χ={chi_90} ({chi_90/len(sv_w)*100:.1f}% of dims)")
    print(f"    99% energy: χ={chi_99} ({chi_99/len(sv_w)*100:.1f}% of dims)")
    print(f"    99.9% energy: χ={chi_999} ({chi_999/len(sv_w)*100:.1f}% of dims)")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print("Power law decay (α > 1): Good compressibility, truncation preserves most energy")
print("Exponential decay: Excellent compressibility, aggressive truncation viable")
print("Slow decay (α < 1): Poor compressibility, need many dimensions to preserve energy")
print("="*80)
