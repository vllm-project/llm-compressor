"""Test BlockTensorizedLinear with clean TT decomposition: block_size = x^num_cores.

This ensures the tensor train decomposition is "clean" - the block can be evenly
factorized into num_cores tensors.
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

print("="*100)
print("BlockTensorizedLinear: Clean Tensor Train Decomposition (block_size = x^num_cores)")
print("="*100)

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

# Define test configurations: (x, num_cores) where block_size = x^num_cores
configs = [
    # Small blocks
    (4, 2, 16),      # 4^2 = 16
    (4, 3, 64),      # 4^3 = 64
    (4, 4, 256),     # 4^4 = 256

    # Medium blocks
    (8, 2, 64),      # 8^2 = 64
    (8, 3, 512),     # 8^3 = 512

    # Large blocks
    (16, 2, 256),    # 16^2 = 256
    (16, 3, 4096),   # 16^3 = 4096

    (32, 2, 1024),   # 32^2 = 1024
]

# Test with rank=1.0 for fair comparison (full rank within block)
test_rank = 1.0

print(f"\n{'='*100}")
print(f"Testing with rank={test_rank} (full rank reconstruction within each block)")
print(f"{'='*100}")

# Store all results for summary table
all_results = []

for layer_name, layer in layers_to_test.items():
    print(f"\n{'-'*100}")
    print(f"Layer: {layer_name} {layer.weight.shape}")
    print(f"{'-'*100}")

    W = layer.weight.data.float().clone()
    in_features = layer.in_features
    out_features = layer.out_features

    # Generate activations for this layer
    input_activations = torch.randn(num_samples, in_features) * 0.02

    print(f"{'x':<4} {'cores':<6} {'block':<8} {'Valid?':<8} {'SNR (dB)':<12} {'Cos Sim':<12} {'Params':<15} {'Ratio':<8} {'Note'}")
    print("-"*100)

    for x, num_cores, block_size in configs:
        # Check if block_size divides both dimensions
        if out_features % block_size != 0 or in_features % block_size != 0:
            print(f"{x:<4} {num_cores:<6} {block_size:<8} {'✗':<8} {'--':<12} {'--':<12} {'--':<15} {'--':<8} "
                  f"Block doesn't divide {out_features}x{in_features}")
            continue

        try:
            # Create BlockTensorizedLinear
            bt = BlockTensorizedLinear.from_linear(
                layer,
                block_size=block_size,
                rank=test_rank,
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
            param_ratio = params / W.numel()

            # Note about decomposition quality
            if snr >= 30:
                note = "Excellent"
            elif snr >= 20:
                note = "Good"
            elif snr >= 10:
                note = "Fair"
            else:
                note = "Poor"

            print(f"{x:<4} {num_cores:<6} {block_size:<8} {'✓':<8} {snr:>8.2f}    {cos_sim:>8.6f}    "
                  f"{params:>10,}    {param_ratio:>6.2f}x  {note}")

            all_results.append({
                'layer': layer_name,
                'x': x,
                'num_cores': num_cores,
                'block_size': block_size,
                'snr': snr,
                'cos_sim': cos_sim,
                'params': params,
                'param_ratio': param_ratio,
            })

        except Exception as e:
            print(f"{x:<4} {num_cores:<6} {block_size:<8} {'✗':<8} {'ERROR':<12} {'--':<12} {'--':<15} {'--':<8} "
                  f"{str(e)[:30]}")

# Summary tables
print(f"\n{'='*100}")
print("SUMMARY: Best Configurations by Layer")
print(f"{'='*100}\n")

for layer_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
    layer_results = [r for r in all_results if r['layer'] == layer_name]

    if not layer_results:
        continue

    # Sort by SNR
    layer_results.sort(key=lambda x: -x['snr'])

    print(f"{layer_name}:")
    print(f"  {'Rank':<4} {'Config':<20} {'SNR':<12} {'Cos Sim':<12} {'Ratio':<10} {'Quality'}")
    print(f"  {'-'*70}")

    for i, r in enumerate(layer_results[:5], 1):
        config = f"x={r['x']}, cores={r['num_cores']} (bs={r['block_size']})"
        quality = "⭐⭐⭐" if r['snr'] >= 30 else "⭐⭐" if r['snr'] >= 20 else "⭐"
        print(f"  {i:<4} {config:<20} {r['snr']:>6.2f} dB   {r['cos_sim']:>8.6f}   {r['param_ratio']:>6.2f}x   {quality}")
    print()

print(f"{'='*100}")
print("SUMMARY: Best Configurations by Block Structure")
print(f"{'='*100}\n")

# Group by (x, num_cores)
from collections import defaultdict
by_config = defaultdict(list)
for r in all_results:
    key = (r['x'], r['num_cores'], r['block_size'])
    by_config[key].append(r)

# Show average performance per configuration across layers
print(f"{'x':<4} {'cores':<6} {'block':<8} {'Avg SNR':<12} {'Avg Cos':<12} {'Layers Tested':<15} {'Quality'}")
print("-"*70)

config_summary = []
for (x, num_cores, block_size), results in by_config.items():
    avg_snr = sum(r['snr'] for r in results) / len(results)
    avg_cos = sum(r['cos_sim'] for r in results) / len(results)
    layers = len(results)

    config_summary.append({
        'x': x,
        'num_cores': num_cores,
        'block_size': block_size,
        'avg_snr': avg_snr,
        'avg_cos': avg_cos,
        'layers': layers,
    })

config_summary.sort(key=lambda x: -x['avg_snr'])

for c in config_summary:
    quality = "⭐⭐⭐" if c['avg_snr'] >= 30 else "⭐⭐" if c['avg_snr'] >= 20 else "⭐"
    print(f"{c['x']:<4} {c['num_cores']:<6} {c['block_size']:<8} {c['avg_snr']:>8.2f}    "
          f"{c['avg_cos']:>8.6f}    {c['layers']:<15} {quality}")

print(f"\n{'='*100}")
print("KEY INSIGHTS")
print(f"{'='*100}")
print("\n1. Block Size vs Quality Trade-off:")
print("   - Smaller blocks (16, 64) → More blocks, potentially worse quality")
print("   - Larger blocks (512, 1024) → Fewer blocks, better local approximation")
print("\n2. Num Cores Impact:")
print("   - More cores → More flexible decomposition, but higher complexity")
print("   - Fewer cores → Simpler, but may lose expressiveness")
print("\n3. Clean Decomposition (block_size = x^num_cores):")
print("   - Ensures tensor dimensions factorize evenly")
print("   - x represents the 'mode size' in each TT core")
print(f"\n{'='*100}")
