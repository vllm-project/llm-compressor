"""Sweep energy thresholds to find optimal compression vs SNR tradeoff."""

from mera_svd_deterministic import DeterministicMERA, compute_snr_db
import torch
from pathlib import Path

device = torch.device('cuda')
cache_dir = Path.home() / '.cache' / 'llm_activations'
activations = torch.load(cache_dir / 'activations_post_attention_ln.pt', map_location='cpu').float()
activations = activations[:8, :4096, :128].to(device)

print('='*80)
print('ENERGY THRESHOLD SWEEP')
print('='*80)
print('\nTesting different energy thresholds to find optimal compression vs SNR:\n')
print(f"{'Threshold':>10} {'L1 χ':>8} {'L2 χ':>8} {'Final χ':>10} {'Compression':>12} {'SNR (dB)':>10}")
print('-' * 80)

for threshold in [0.999, 0.99, 0.95, 0.90, 0.85]:
    mera = DeterministicMERA(4096, 128, 128, threshold).to(device)
    mera.initialize_uv_gate(activations)

    with torch.no_grad():
        latent, intermediates = mera.build_tree(activations, num_layers=3, verbose=False)
        recon = mera.reconstruct(latent, intermediates)

    snr = compute_snr_db(activations[0], recon[0])
    compression = (4096 * 128) / latent[0].numel()

    chi_str = f'{mera.chi_eff_list[0]}' if len(mera.chi_eff_list) > 0 else 'N/A'
    chi2_str = f'{mera.chi_eff_list[1]}' if len(mera.chi_eff_list) > 1 else 'N/A'
    chi_final = latent.shape[-1]

    marker = '✓' if snr >= 30.0 else ' '
    print(f'{marker} {threshold:>9.1%} {chi_str:>8} {chi2_str:>8} {chi_final:>10} {compression:>11.1f}x {snr:>10.2f}')

print('\n' + '='*80)
