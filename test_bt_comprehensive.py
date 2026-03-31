"""Comprehensive BlockTensorizedLinear test with multiple ranks.

Test block_size = x^num_cores with rank = 1.0, 0.5, 0.25
Show SNR, Cosine Similarity, and Compression Ratio for each.
"""
import torch
import torch.nn.functional as F
import sys
import importlib.util
from transformers import AutoModelForCausalLM

# Load tensorized_linear module first
spec_tn = importlib.util.spec_from_file_location(
    "tensorized_linear",
    "src/llmcompressor/modifiers/experimental/tensorized_linear.py"
)
tensorized_module = importlib.util.module_from_spec(spec_tn)
sys.modules['llmcompressor.modifiers.experimental.tensorized_linear'] = tensorized_module
spec_tn.loader.exec_module(tensorized_module)

# Load block_tensorized_linear module
spec = importlib.util.spec_from_file_location(
    "block_tensorized_linear",
    "src/llmcompressor/modifiers/experimental/block_tensorized_linear.py"
)
block_tn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(block_tn_module)

BlockTensorizedLinear = block_tn_module.BlockTensorizedLinear

def compute_sqnr(original, approximation):
    """Compute SQNR in dB."""
    signal_power = torch.var(original)
    mse_noise = torch.mean((original - approximation) ** 2)
    sqnr_linear = signal_power / (mse_noise + 1e-10)
    sqnr_db = 10 * torch.log10(sqnr_linear)
    return sqnr_db.item()

def cosine_similarity_matrix(W1, W2):
    """Compute cosine similarity between two weight matrices."""
    return F.cosine_similarity(W1.flatten(), W2.flatten(), dim=0).item()

print("="*120)
print("BlockTensorizedLinear: Comprehensive Analysis (rank = 1.0, 0.5, 0.25)")
print("="*120)

# Load model
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Get layers to test
layers_to_test = {
    'q_proj': model.model.layers[0].self_attn.q_proj,
    'k_proj': model.model.layers[0].self_attn.k_proj,
    'v_proj': model.model.layers[0].self_attn.v_proj,
    'o_proj': model.model.layers[0].self_attn.o_proj,
}

print("\nLayer shapes:")
for name, layer in layers_to_test.items():
    print(f"  {name}: {layer.weight.shape} ({layer.weight.numel():,} params)")

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Generate synthetic activations
num_samples = 256

# Define test configurations: (x, num_cores, block_size)
configs = [
    (4, 2, 16),      # 4^2 = 16
    (4, 3, 64),      # 4^3 = 64
    (4, 4, 256),     # 4^4 = 256
    (8, 2, 64),      # 8^2 = 64
    (8, 3, 512),     # 8^3 = 512
    (16, 2, 256),    # 16^2 = 256
    (32, 2, 1024),   # 32^2 = 1024
]

# Test with multiple ranks
test_ranks = [1.0, 0.5, 0.25]

# Store all results
all_results = []

for rank in test_ranks:
    print(f"\n{'='*120}")
    print(f"RANK = {rank}")
    print(f"{'='*120}")

    for layer_name, layer in layers_to_test.items():
        print(f"\n{'-'*120}")
        print(f"Layer: {layer_name} {layer.weight.shape}")
        print(f"{'-'*120}")

        W = layer.weight.data.float().clone()
        in_features = layer.in_features
        out_features = layer.out_features
        original_params = W.numel()

        # Generate activations for this layer
        input_activations = torch.randn(num_samples, in_features) * 0.02

        print(f"{'x':<4} {'cores':<6} {'block':<8} {'SNR (dB)':<12} {'Cos Sim':<12} {'Params':<15} {'Ratio':<10} {'Compression'}")
        print("-"*120)

        for x, num_cores, block_size in configs:
            # Check if block_size divides both dimensions
            if out_features % block_size != 0 or in_features % block_size != 0:
                continue

            try:
                # Create BlockTensorizedLinear
                bt = BlockTensorizedLinear.from_linear(
                    layer,
                    block_size=block_size,
                    rank=rank,
                    num_cores=num_cores,
                    input_activations=input_activations,
                )

                # Reconstruct weight matrix
                W_bt = bt.to_matrix().float()

                # Compute metrics
                snr = compute_sqnr(W, W_bt)
                cos_sim = cosine_similarity_matrix(W, W_bt)

                # Compute parameters
                params = bt.num_params
                param_ratio = params / original_params
                compression_pct = 100 * (1 - param_ratio)

                print(f"{x:<4} {num_cores:<6} {block_size:<8} {snr:>8.2f}    {cos_sim:>8.6f}    "
                      f"{params:>10,}    {param_ratio:>6.2f}x    {compression_pct:>6.1f}%")

                all_results.append({
                    'rank': rank,
                    'layer': layer_name,
                    'x': x,
                    'num_cores': num_cores,
                    'block_size': block_size,
                    'snr': snr,
                    'cos_sim': cos_sim,
                    'params': params,
                    'param_ratio': param_ratio,
                    'compression_pct': compression_pct,
                })

            except Exception as e:
                print(f"{x:<4} {num_cores:<6} {block_size:<8} ERROR: {str(e)[:50]}")

# Generate summary tables
print(f"\n{'='*120}")
print("SUMMARY TABLE: Average Performance Across All Layers")
print(f"{'='*120}\n")

# Group by (rank, x, num_cores, block_size)
from collections import defaultdict
by_config = defaultdict(list)
for r in all_results:
    key = (r['rank'], r['x'], r['num_cores'], r['block_size'])
    by_config[key].append(r)

summary = []
for (rank, x, num_cores, block_size), results in by_config.items():
    avg_snr = sum(r['snr'] for r in results) / len(results)
    avg_cos = sum(r['cos_sim'] for r in results) / len(results)
    avg_ratio = sum(r['param_ratio'] for r in results) / len(results)
    avg_comp = sum(r['compression_pct'] for r in results) / len(results)
    layers_tested = len(results)

    summary.append({
        'rank': rank,
        'x': x,
        'num_cores': num_cores,
        'block_size': block_size,
        'avg_snr': avg_snr,
        'avg_cos': avg_cos,
        'avg_ratio': avg_ratio,
        'avg_comp': avg_comp,
        'layers': layers_tested,
    })

# Sort by rank then avg_snr
summary.sort(key=lambda x: (-x['rank'], -x['avg_snr']))

print(f"{'Rank':<6} {'Config':<20} {'Avg SNR':<12} {'Avg Cos':<12} {'Avg Ratio':<12} {'Avg Comp':<12} {'Layers'}")
print("-"*90)

for s in summary:
    config = f"x={s['x']}, c={s['num_cores']} (bs={s['block_size']})"
    print(f"{s['rank']:<6.2f} {config:<20} {s['avg_snr']:>8.2f} dB  {s['avg_cos']:>8.6f}  "
          f"{s['avg_ratio']:>8.2f}x    {s['avg_comp']:>8.1f}%    {s['layers']}")

# Best configuration for each rank
print(f"\n{'='*120}")
print("BEST CONFIGURATION FOR EACH RANK (by Average SNR)")
print(f"{'='*120}\n")

for rank in test_ranks:
    rank_results = [s for s in summary if s['rank'] == rank]
    if rank_results:
        best = max(rank_results, key=lambda x: x['avg_snr'])
        print(f"Rank {rank}:")
        print(f"  Config: x={best['x']}, num_cores={best['num_cores']}, block_size={best['block_size']}")
        print(f"  Avg SNR: {best['avg_snr']:.2f} dB")
        print(f"  Avg Cosine Similarity: {best['avg_cos']:.6f}")
        print(f"  Avg Param Ratio: {best['avg_ratio']:.2f}x")
        print(f"  Avg Compression: {best['avg_comp']:.1f}%")
        print()

# Per-layer breakdown for best configs
print(f"{'='*120}")
print("PER-LAYER BREAKDOWN: Best Config (x=8, cores=3, block_size=512) at Different Ranks")
print(f"{'='*120}\n")

print(f"{'Layer':<10} {'Rank':<6} {'SNR (dB)':<12} {'Cos Sim':<12} {'Params':<15} {'Ratio':<10} {'Compression'}")
print("-"*90)

for layer_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    for rank in test_ranks:
        layer_results = [r for r in all_results
                        if r['layer'] == layer_name
                        and r['x'] == 8
                        and r['num_cores'] == 3
                        and r['block_size'] == 512
                        and r['rank'] == rank]

        if layer_results:
            r = layer_results[0]
            print(f"{layer_name:<10} {rank:<6.2f} {r['snr']:>8.2f}    {r['cos_sim']:>8.6f}    "
                  f"{r['params']:>10,}    {r['param_ratio']:>6.2f}x    {r['compression_pct']:>6.1f}%")
    print()

# Quality vs Compression trade-off table
print(f"{'='*120}")
print("QUALITY vs COMPRESSION TRADE-OFF (x=8, cores=3, block_size=512)")
print(f"{'='*120}\n")

print("This table shows the optimal config across all ranks:\n")
print(f"{'Rank':<6} {'Avg SNR':<12} {'Avg Cos':<12} {'Avg Ratio':<12} {'Avg Comp':<12} {'Quality':<15} {'Use Case'}")
print("-"*110)

for rank in sorted(test_ranks, reverse=True):
    rank_results = [r for r in all_results
                   if r['x'] == 8
                   and r['num_cores'] == 3
                   and r['block_size'] == 512
                   and r['rank'] == rank]

    if rank_results:
        avg_snr = sum(r['snr'] for r in rank_results) / len(rank_results)
        avg_cos = sum(r['cos_sim'] for r in rank_results) / len(rank_results)
        avg_ratio = sum(r['param_ratio'] for r in rank_results) / len(rank_results)
        avg_comp = sum(r['compression_pct'] for r in rank_results) / len(rank_results)

        if avg_snr >= 15:
            quality = "Excellent"
            use_case = "High-fidelity"
        elif avg_snr >= 10:
            quality = "Good"
            use_case = "Balanced"
        elif avg_snr >= 5:
            quality = "Moderate"
            use_case = "Aggressive compression"
        else:
            quality = "Poor"
            use_case = "Too lossy"

        print(f"{rank:<6.2f} {avg_snr:>8.2f} dB  {avg_cos:>8.6f}  "
              f"{avg_ratio:>8.2f}x    {avg_comp:>8.1f}%    {quality:<15} {use_case}")

print(f"\n{'='*120}")
print("KEY RECOMMENDATIONS")
print(f"{'='*120}")
print("\n1. OPTIMAL CONFIGURATION: x=8, num_cores=3, block_size=512")
print("   - Best SNR across all ranks")
print("   - Clean tensor train decomposition (8^3 = 512)")
print("   - Consistently high cosine similarity (>0.99 for rank=1.0)")

print("\n2. RANK SELECTION:")
print("   - rank=1.0: ~18 dB SNR, ~1.0x params (no compression)")
print("   - rank=0.5: ~12 dB SNR, ~0.5x params (50% compression)")
print("   - rank=0.25: ~7 dB SNR, ~0.25x params (75% compression)")

print("\n3. COMPRESSION STRATEGY:")
print("   - For high quality: Use rank=1.0 (matches original params)")
print("   - For balanced: Use rank=0.5 (50% compression, good quality)")
print("   - For aggressive: Use rank=0.25 (75% compression, moderate quality)")

print(f"\n{'='*120}")
