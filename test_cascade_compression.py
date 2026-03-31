"""Test cascaded compression: BT512 → BT256 (residual) → LowRank (residual)."""
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
print("Testing Cascaded Compression:")
print("  Stage 1: BlockTensorizedLinear (block_size=512, num_cores=3)")
print("  Stage 2: BlockTensorizedLinear on residual (block_size=256, num_cores=4)")
print("  Stage 3: Low-rank SVD on residual")
print("="*80)

# Load real model weights
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

q_proj = model.model.layers[0].self_attn.q_proj
W = q_proj.weight.data.float().clone()
in_features = q_proj.in_features
out_features = q_proj.out_features

print(f"Loaded q_proj weight: {W.shape}")
print(f"Original params: {W.numel():,}")

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

original_params = W.numel()

# Generate synthetic activations
num_samples = 256
input_activations = torch.randn(num_samples, in_features) * 0.02

print(f"\n{'='*80}")
print(f"Testing Different Rank Combinations")
print(f"{'='*80}\n")

print(f"{'Stage1':<8} {'Stage2':<8} {'LR':<6} {'S1 SNR':<10} {'S2 SNR':<10} {'Final SNR':<11} {'Total Params':<15} {'Ratio':<8} {'Status'}")
print("-"*100)

results = []

# Test different rank combinations
stage1_ranks = [0.7, 0.8, 0.9, 1.0]
stage2_ranks = [0.6, 0.7, 0.8, 0.9]
lr_ranks = [100, 200, 300, 400, 500, 600, 800, 1000]

for s1_rank in stage1_ranks:
    print(f"\n# Stage 1 rank = {s1_rank}")

    # Stage 1: BlockTensorizedLinear (512, num_cores=3)
    try:
        bt1 = BlockTensorizedLinear.from_linear(
            q_proj,
            block_size=512,
            rank=s1_rank,
            num_cores=3,
            input_activations=input_activations,
        )
    except Exception as e:
        print(f"  Skipping stage1_rank={s1_rank}: {e}")
        continue

    W_bt1 = bt1.to_matrix().float()
    residual1 = W - W_bt1
    s1_snr = compute_sqnr(W, W_bt1)
    s1_params = bt1.num_params

    for s2_rank in stage2_ranks:
        # Stage 2: BlockTensorizedLinear on residual (256, num_cores=4)
        # Create a temporary linear layer with residual1 as weight
        temp_linear = nn.Linear(in_features, out_features, bias=False)
        temp_linear.weight.data = residual1.clone()

        try:
            bt2 = BlockTensorizedLinear.from_linear(
                temp_linear,
                block_size=256,
                rank=s2_rank,
                num_cores=4,
                input_activations=input_activations,
            )
        except Exception as e:
            continue

        W_bt2 = bt2.to_matrix().float()
        residual2 = residual1 - W_bt2

        # Combined after stage 2
        W_combined_s2 = W_bt1 + W_bt2
        s2_snr = compute_sqnr(W, W_combined_s2)
        s2_params = bt2.num_params

        # Stage 3: Low-rank SVD on residual2
        U, S, Vh = torch.linalg.svd(residual2, full_matrices=False)

        for lr_rank in lr_ranks:
            # Low-rank approximation of residual2
            U_r = U[:, :lr_rank]
            S_r = S[:lr_rank]
            Vh_r = Vh[:lr_rank, :]
            W_lr = U_r @ torch.diag(S_r) @ Vh_r

            # Final combined reconstruction
            W_final = W_bt1 + W_bt2 + W_lr

            # Compute final SNR
            final_snr = compute_sqnr(W, W_final)

            # Total parameters
            lr_params = lr_rank * (in_features + out_features)
            total_params = s1_params + s2_params + lr_params
            param_ratio = total_params / original_params

            # Store result
            result = {
                's1_rank': s1_rank,
                's2_rank': s2_rank,
                'lr_rank': lr_rank,
                's1_snr': s1_snr,
                's2_snr': s2_snr,
                'final_snr': final_snr,
                'total_params': total_params,
                'param_ratio': param_ratio,
                's1_params': s1_params,
                's2_params': s2_params,
                'lr_params': lr_params,
            }
            results.append(result)

            # Print if achieves > 35 dB or notable
            status = ""
            if final_snr >= 35.0:
                if param_ratio < 1.0:
                    status = "✅ TARGET (compressed)"
                else:
                    status = "✓ TARGET"

            # Print every 10th result plus all targets
            if status or (len(results) % 10 == 0):
                print(f"{s1_rank:<8.1f} {s2_rank:<8.1f} {lr_rank:<6} "
                      f"{s1_snr:>6.2f} dB  {s2_snr:>6.2f} dB  {final_snr:>6.2f} dB   "
                      f"{total_params:>10,}    {param_ratio:>6.2f}x  {status}")

print(f"\n{'='*80}")
print("Best Results (SNR >= 35 dB)")
print(f"{'='*80}")

# Filter and sort
target_results = [r for r in results if r['final_snr'] >= 35.0]
target_results.sort(key=lambda x: x['param_ratio'])
compressed_results = [r for r in target_results if r['param_ratio'] < 1.0]

if target_results:
    print(f"\n{'Config':<40} {'Final SNR':<12} {'Total Params':<15} {'Ratio':<10} {'Compression'}")
    print("-"*100)
    for r in target_results[:10]:
        config = f"S1:{r['s1_rank']:.1f} S2:{r['s2_rank']:.1f} LR:{r['lr_rank']}"
        compression = f"{100*(1-r['param_ratio']):.1f}%" if r['param_ratio'] < 1.0 else "None"
        print(f"{config:<40} {r['final_snr']:>6.2f} dB    {r['total_params']:>10,}    "
              f"{r['param_ratio']:>6.2f}x   {compression}")

    if compressed_results:
        best = compressed_results[0]
        print(f"\n🎉 Best Compressed Result:")
        print(f"   Stage 1: block_size=512, rank={best['s1_rank']}, num_cores=3 ({best['s1_params']:,} params, {best['s1_snr']:.2f} dB)")
        print(f"   Stage 2: block_size=256, rank={best['s2_rank']}, num_cores=4 ({best['s2_params']:,} params)")
        print(f"   Stage 3: low_rank={best['lr_rank']} ({best['lr_params']:,} params)")
        print(f"   Final SNR: {best['final_snr']:.2f} dB")
        print(f"   Total Params: {best['total_params']:,} ({best['param_ratio']:.2f}x)")
        print(f"   Compression: {100*(1-best['param_ratio']):.1f}%")
else:
    print("\n⚠️  No configurations achieved SNR >= 35 dB")

    # Show closest
    results.sort(key=lambda x: -x['final_snr'])
    print(f"\nClosest Results:")
    print(f"{'Config':<40} {'Final SNR':<12} {'Total Params':<15} {'Ratio'}")
    print("-"*100)
    for r in results[:5]:
        config = f"S1:{r['s1_rank']:.1f} S2:{r['s2_rank']:.1f} LR:{r['lr_rank']}"
        print(f"{config:<40} {r['final_snr']:>6.2f} dB    {r['total_params']:>10,}    "
              f"{r['param_ratio']:>6.2f}x")

print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")
print(f"Total configurations tested: {len(results)}")
print(f"Configurations achieving SNR >= 35 dB: {len(target_results)}")
if compressed_results:
    print(f"With compression (< 1.0x params): {len(compressed_results)}")
print(f"{'='*80}")
