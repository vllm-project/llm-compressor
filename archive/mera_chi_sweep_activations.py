"""Sweep chi values to find optimal compression/SNR trade-off."""

import torch
import numpy as np
from pathlib import Path
from mera_hierarchical import HierarchicalMERA, train_layer0, train_remaining_layers, evaluate_all_layers, global_tuning
from mera_batch_universal import compute_snr_db


def load_activations():
    """Load saved LLM activations."""
    cache_dir = Path.home() / '.cache' / 'llm_activations'
    file_path = cache_dir / 'activations_post_attention_ln.pt'

    print(f"Loading activations from {file_path}...")
    activations = torch.load(file_path, map_location='cpu')

    if activations.dtype == torch.bfloat16:
        activations = activations.float()

    return activations


def test_chi(activations, chi, device):
    """Test MERA with specific chi value."""
    batch_size, seq_len, hidden_dim = activations.shape

    print(f"\n{'='*70}")
    print(f"Testing χ = {chi}")
    print(f"{'='*70}")

    compression_l0 = hidden_dim / chi
    print(f"Layer 0 compression: {compression_l0:.1f}x")

    # Initialize MERA
    mera = HierarchicalMERA(seq_len, hidden_dim, chi).to(device)
    mera.initialize_uv_gate(activations)

    # Train
    train_layer0(mera, activations)
    train_remaining_layers(mera, activations, max_layers=2)

    # Evaluate
    mera.eval()
    with torch.no_grad():
        results = {}

        for stop_layer in range(3):  # L0, L1, L2
            latent, intermediates = mera(activations[:1], stop_layer=stop_layer)
            recon = mera.reconstruct(latent, intermediates)

            snr = compute_snr_db(activations[0], recon[0])
            compression = (seq_len * hidden_dim) / latent[0].numel()

            results[f'L{stop_layer}'] = {
                'snr': snr,
                'compression': compression,
                'latent_shape': list(latent[0].shape)
            }

    return results


def main():
    print("="*70)
    print("CHI SWEEP ON FULL ACTIVATIONS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load activations
    activations = load_activations()
    batch_size, seq_len, hidden_dim = activations.shape

    print(f"\nOriginal shape: [{batch_size}, {seq_len}, {hidden_dim}]")

    # Reduce batch size for memory
    batch_size = 8
    activations = activations[:batch_size]

    # Pad sequence
    target_seq_len = 3 ** int(np.ceil(np.log(seq_len) / np.log(3)))
    if seq_len != target_seq_len:
        print(f"Padding sequence: {seq_len} → {target_seq_len}")
        pad_len = target_seq_len - seq_len
        activations = torch.nn.functional.pad(activations, (0, 0, 0, pad_len))
        seq_len = target_seq_len

    activations = activations.to(device)

    # Analyze spectrum
    print("\n" + "="*70)
    print("SPECTRUM ANALYSIS")
    print("="*70)

    with torch.no_grad():
        act_flat = activations.reshape(-1, hidden_dim).double()
        U, S, Vt = torch.linalg.svd(act_flat, full_matrices=False)

        energy = (S ** 2).cumsum(0) / (S ** 2).sum()
        eff_rank_95 = (energy < 0.95).sum().item() + 1

        print(f"  Effective rank (95%): {eff_rank_95}/{hidden_dim} ({eff_rank_95/hidden_dim:.1%})")

    # Chi values to test
    chi_values = [256, 512, 1024, 2048]

    print(f"\nTesting chi values: {chi_values}")
    print(f"Effective rank: {eff_rank_95}")

    all_results = {}

    for chi in chi_values:
        try:
            results = test_chi(activations, chi, device)
            all_results[chi] = results

            # Print summary
            print(f"\n  Results for χ={chi}:")
            for layer, data in results.items():
                print(f"    {layer}: {data['snr']:.2f} dB at {data['compression']:.1f}x")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n  χ={chi}: OUT OF MEMORY")
                torch.cuda.empty_cache()
                continue
            else:
                raise

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Chi':>6} {'L0 SNR':>10} {'L0 Comp':>10} {'L1 SNR':>10} {'L1 Comp':>10} {'L2 SNR':>10} {'L2 Comp':>10}")
    print("-" * 70)

    for chi in chi_values:
        if chi in all_results:
            r = all_results[chi]
            print(f"{chi:6d} "
                  f"{r['L0']['snr']:10.2f} {r['L0']['compression']:9.1f}x "
                  f"{r['L1']['snr']:10.2f} {r['L1']['compression']:9.1f}x "
                  f"{r['L2']['snr']:10.2f} {r['L2']['compression']:9.1f}x")
        else:
            print(f"{chi:6d} {'OOM':>10}")

    # Find best for 5-10x target
    print("\n" + "="*70)
    print("BEST FOR 5-10x COMPRESSION TARGET")
    print("="*70)

    best_chi = None
    best_snr = -np.inf

    for chi, results in all_results.items():
        for layer, data in results.items():
            comp = data['compression']
            snr = data['snr']
            if 5 <= comp <= 10:
                if snr > best_snr:
                    best_snr = snr
                    best_chi = chi
                    best_layer = layer
                    best_comp = comp

    if best_chi:
        print(f"\n  Best: χ={best_chi}, {best_layer}")
        print(f"  SNR: {best_snr:.2f} dB")
        print(f"  Compression: {best_comp:.1f}x")
        print(f"  Gap to 30 dB target: {30 - best_snr:.2f} dB")
    else:
        print("\n  No configuration achieved 5-10x compression")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
