"""Test hard chi caps with lower energy thresholds."""

from mera_svd_deterministic import DeterministicMERA, compute_snr_db
import torch
from pathlib import Path
import numpy as np

device = torch.device('cuda')
cache_dir = Path.home() / '.cache' / 'llm_activations'
activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
activations = activations[:64, :4096, :128].to(device)

print('='*80)
print('HARD CHI CAP + ENERGY THRESHOLD SWEEP')
print('='*80)
print('\nForcing information bottleneck with hard chi caps...\n')

print(f"{'Chi Cap':>8} {'Threshold':>10} {'Final chi':>10} {'Compress':>10} {'Mean SNR':>10} {'>30dB%':>10}")
print('-'*80)

for chi_cap in [128, 64, 48, 32]:
    for threshold in [0.99, 0.95]:
        mera = DeterministicMERA(4096, 128, chi_cap, threshold).to(device)
        mera.initialize_uv_gate(activations)

        with torch.no_grad():
            latent, intermediates = mera.build_tree(activations, num_layers=3, verbose=False)
            recon = mera.reconstruct(latent, intermediates)

        snrs = [compute_snr_db(activations[i], recon[i]) for i in range(64)]
        compression = (4096 * 128) / latent[0].numel()
        pct_above_30 = 100 * sum(1 for s in snrs if s >= 30.0) / 64

        marker = '✓' if np.mean(snrs) >= 25.0 else ' '
        print(f'{marker} {chi_cap:>6} {threshold:>9.1%} {latent.shape[-1]:>10} {compression:>9.2f}x {np.mean(snrs):>9.2f}dB {pct_above_30:>9.1f}%')

print('\n' + '='*80)
print('TARGET: High compression (5-10x) + High SNR (25-30 dB)')
print('='*80)
