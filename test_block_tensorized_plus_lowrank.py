"""Test BlockTensorizedLinear + Low-Rank Residual Compression.

Tests whether a hybrid approach of:
1. BlockTensorizedLinear for initial approximation
2. Low-rank (SVD) approximation of residuals

can achieve SNR > 35 dB on real Llama weights while maintaining compression.
"""
import torch
import torch.nn as nn
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

print("="*80)
print("Testing BlockTensorizedLinear + Low-Rank Residual Compression")
print("="*80)

# Load real model weights
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Get first q_proj layer (2048x2048)
q_proj = model.model.layers[0].self_attn.q_proj
W = q_proj.weight.data.float().clone()
in_features = q_proj.in_features
out_features = q_proj.out_features

print(f"Loaded q_proj weight: {W.shape}")
print(f"Original params: {W.numel():,}")

# Clean up model
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

original_params = W.numel()

# Generate synthetic activations for activation-aware compression
num_samples = 256
input_activations = torch.randn(num_samples, in_features) * 0.02

print(f"\n{'='*80}")
print(f"Testing Hybrid Approach: BlockTensorizedLinear + Low-Rank Residual")
print(f"{'='*80}\n")

print(f"{'Block':<6} {'BT':<5} {'Cores':<6} {'Res':<6} {'Block SNR':<11} {'Hybrid SNR':<11} {'Total Params':<15} {'Ratio':<8} {'Status'}")
print("-"*80)

results = []

# Test different configurations
# Strategy: Use very high BlockTensorizedLinear rank for good base approximation,
# then see if moderate residuals can reach 35 dB with < 1.0x total params
block_sizes = [512]
bt_ranks = [0.85, 0.90, 0.95]  # Very high ranks for strong base
num_cores_list = [3]
residual_ranks = [100, 150, 200, 250, 300, 350, 400, 500]  # Try smaller residuals

for block_size in block_sizes:
    for bt_rank in bt_ranks:
        for num_cores in num_cores_list:
            # Create BlockTensorizedLinear
            try:
                bt_linear = BlockTensorizedLinear.from_linear(
                    q_proj,
                    block_size=block_size,
                    rank=bt_rank,
                    num_cores=num_cores,
                    input_activations=input_activations,
                )
            except Exception as e:
                print(f"Skipping block_size={block_size}, rank={bt_rank}, cores={num_cores}: {e}")
                continue

            # Reconstruct and compute residual
            W_bt = bt_linear.to_matrix().float()
            residual = W - W_bt

            # Compute BlockTensorized-only SNR
            bt_only_snr = compute_sqnr(W, W_bt)

            bt_params = bt_linear.num_params

            # Test different residual ranks
            for res_rank in residual_ranks:
                # SVD of residual
                U, S, Vh = torch.linalg.svd(residual, full_matrices=False)

                # Low-rank approximation of residual
                U_r = U[:, :res_rank]
                S_r = S[:res_rank]
                Vh_r = Vh[:res_rank, :]
                W_residual_lr = U_r @ torch.diag(S_r) @ Vh_r

                # Combined reconstruction
                W_combined = W_bt + W_residual_lr

                # Compute hybrid SNR
                hybrid_snr = compute_sqnr(W, W_combined)

                # Compute total parameters
                # BlockTensorized params + residual low-rank params
                residual_params = res_rank * (in_features + out_features)
                total_params = bt_params + residual_params
                param_ratio = total_params / original_params

                # Store result
                result = {
                    'block_size': block_size,
                    'bt_rank': bt_rank,
                    'num_cores': num_cores,
                    'res_rank': res_rank,
                    'bt_snr': bt_only_snr,
                    'hybrid_snr': hybrid_snr,
                    'total_params': total_params,
                    'param_ratio': param_ratio,
                }
                results.append(result)

                # Print if achieves > 35 dB or if notable configuration
                status = ""
                if hybrid_snr >= 35.0:
                    if param_ratio < 1.0:
                        status = "✅ TARGET (compressed)"
                    else:
                        status = "✓ TARGET"

                # Print every 5th result to avoid clutter, plus all targets
                if status or (len(results) % 10 == 0):
                    print(f"{block_size:<6} {bt_rank:<5.1f} {num_cores:<6} {res_rank:<6} "
                          f"{bt_only_snr:>6.2f} dB   {hybrid_snr:>6.2f} dB   "
                          f"{total_params:>10,}    {param_ratio:>6.2f}x  {status}")

print("\n" + "="*80)
print("Best Results (SNR >= 35 dB)")
print("="*80)

# Filter and sort results
target_results = [r for r in results if r['hybrid_snr'] >= 35.0]
target_results.sort(key=lambda x: x['param_ratio'])
compressed_results = [r for r in target_results if r['param_ratio'] < 1.0]

if target_results:
    print(f"\n{'Config':<30} {'Hybrid SNR':<12} {'Total Params':<15} {'Ratio':<10} {'Compression'}")
    print("-"*80)
    for r in target_results[:10]:  # Show top 10
        config = f"BS{r['block_size']}_R{r['bt_rank']:.1f}_C{r['num_cores']}_ResR{r['res_rank']}"
        compression = f"{100*(1-r['param_ratio']):.1f}%" if r['param_ratio'] < 1.0 else "None"
        print(f"{config:<30} {r['hybrid_snr']:>6.2f} dB    {r['total_params']:>10,}    "
              f"{r['param_ratio']:>6.2f}x   {compression}")

    # Show the best compressed result
    if compressed_results:
        best_compressed = compressed_results[0]
        print(f"\n🎉 Best Compressed Result:")
        print(f"   Config: block_size={best_compressed['block_size']}, "
              f"bt_rank={best_compressed['bt_rank']}, "
              f"num_cores={best_compressed['num_cores']}, "
              f"res_rank={best_compressed['res_rank']}")
        print(f"   SNR: {best_compressed['hybrid_snr']:.2f} dB")
        print(f"   Params: {best_compressed['total_params']:,} ({best_compressed['param_ratio']:.2f}x)")
        print(f"   Compression: {100*(1-best_compressed['param_ratio']):.1f}%")
else:
    print("\n⚠️  No configurations achieved SNR >= 35 dB")

    # Show closest results
    results.sort(key=lambda x: -x['hybrid_snr'])
    print(f"\nClosest Results:")
    print(f"{'Config':<30} {'Hybrid SNR':<12} {'Total Params':<15} {'Ratio'}")
    print("-"*80)
    for r in results[:5]:
        config = f"BS{r['block_size']}_R{r['bt_rank']:.1f}_C{r['num_cores']}_ResR{r['res_rank']}"
        print(f"{config:<30} {r['hybrid_snr']:>6.2f} dB    {r['total_params']:>10,}    "
              f"{r['param_ratio']:>6.2f}x")

print("\n" + "="*80)
print("Summary")
print("="*80)
print(f"Total configurations tested: {len(results)}")
print(f"Configurations achieving SNR >= 35 dB: {len(target_results)}")
if compressed_results:
    print(f"With compression (< 1.0x params): {len(compressed_results)}")
print("="*80)
