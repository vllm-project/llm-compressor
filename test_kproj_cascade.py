"""Test cascaded compression on k_proj - can we reach 35 dB with < 1.0x params?

k_proj requires only 1.21x params with pure SVD (rank=495).
Can BlockTensorized + SVD cascade get under 1.0x?
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
print("k_proj Cascaded Compression: Can we beat 1.21x while reaching 35 dB?")
print("="*80)

# Load model
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

k_proj = model.model.layers[0].self_attn.k_proj
W = k_proj.weight.data.float().clone()
in_features = k_proj.in_features  # 2048
out_features = k_proj.out_features  # 512

print(f"k_proj weight: {W.shape}")
print(f"Original params: {W.numel():,}")
print(f"Target: < 1.0x params ({W.numel():,})")

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

original_params = W.numel()

# Generate synthetic activations
num_samples = 256
input_activations = torch.randn(num_samples, in_features) * 0.02

# Baseline: Pure SVD
print(f"\nBaseline - Pure SVD to reach 35 dB:")
U, S, Vh = torch.linalg.svd(W, full_matrices=False)

# Binary search for 35 dB
left, right = 1, min(W.shape)
baseline_rank = None
while left <= right:
    mid = (left + right) // 2
    U_r = U[:, :mid]
    S_r = S[:mid]
    Vh_r = Vh[:mid, :]
    W_approx = U_r @ torch.diag(S_r) @ Vh_r
    snr = compute_sqnr(W, W_approx)
    if snr >= 35.0:
        baseline_rank = mid
        right = mid - 1
    else:
        left = mid + 1

baseline_params = baseline_rank * (W.shape[0] + W.shape[1])
baseline_ratio = baseline_params / original_params
print(f"  Rank: {baseline_rank}")
print(f"  Params: {baseline_params:,} ({baseline_ratio:.2f}x)")
print(f"  SNR: 35+ dB")

print(f"\n{'='*80}")
print("Strategy: BlockTensorized → SVD (residual)")
print(f"{'='*80}\n")

print(f"{'BT blk':<8} {'BT rnk':<8} {'SVD':<6} {'Stage1':<10} {'Stage2':<10} {'Final SNR':<11} {'Params':<15} {'Ratio':<8} {'Target?'}")
print("-"*95)

results = []

# Test BlockTensorized first, then SVD on residual
# k_proj is 512x2048, so block_size must divide both dimensions
bt_configs = [
    (256, 0.5),
    (256, 0.6),
    (256, 0.7),
    (256, 0.8),
    (512, 0.5),
    (512, 0.6),
    (512, 0.7),
    (512, 0.8),
]

# For each BT config, binary search for minimum SVD rank to reach 35 dB
for block_size, bt_rank in bt_configs:
    # Check if block_size divides dimensions
    if W.shape[0] % block_size != 0 or W.shape[1] % block_size != 0:
        continue

    # Stage 1: BlockTensorized
    try:
        bt = BlockTensorizedLinear.from_linear(
            k_proj,
            block_size=block_size,
            rank=bt_rank,
            num_cores=3,
            input_activations=input_activations,
        )
    except Exception as e:
        continue

    W_bt = bt.to_matrix().float()
    residual = W - W_bt
    s1_snr = compute_sqnr(W, W_bt)
    s1_params = bt.num_params

    # Stage 2: Binary search for minimum SVD rank to reach 35 dB
    U_r, S_r, Vh_r = torch.linalg.svd(residual, full_matrices=False)

    left, right = 1, min(residual.shape)
    best_svd_rank = None
    best_snr = 0

    while left <= right:
        mid = (left + right) // 2

        # Test this SVD rank
        U_test = U_r[:, :mid]
        S_test = S_r[:mid]
        Vh_test = Vh_r[:mid, :]
        W_svd = U_test @ torch.diag(S_test) @ Vh_test

        W_combined = W_bt + W_svd
        snr = compute_sqnr(W, W_combined)

        if snr >= 35.0:
            best_svd_rank = mid
            best_snr = snr
            right = mid - 1
        else:
            left = mid + 1

    if best_svd_rank is not None:
        # Compute final stats
        s2_params = best_svd_rank * (W.shape[0] + W.shape[1])
        total_params = s1_params + s2_params
        param_ratio = total_params / original_params

        # Recompute combined for accurate stage 2 SNR
        U_final = U_r[:, :best_svd_rank]
        S_final = S_r[:best_svd_rank]
        Vh_final = Vh_r[:best_svd_rank, :]
        W_svd_final = U_final @ torch.diag(S_final) @ Vh_final
        W_combined_final = W_bt + W_svd_final
        s2_snr = compute_sqnr(W, W_combined_final)

        target_mark = "✅ TARGET!" if param_ratio < 1.0 else ""

        result = {
            'block_size': block_size,
            'bt_rank': bt_rank,
            'svd_rank': best_svd_rank,
            's1_snr': s1_snr,
            's2_snr': s2_snr,
            'total_params': total_params,
            'param_ratio': param_ratio,
        }
        results.append(result)

        print(f"{block_size:<8} {bt_rank:<8.1f} {best_svd_rank:<6} "
              f"{s1_snr:>6.2f} dB  {s2_snr:>6.2f} dB  {s2_snr:>6.2f} dB   "
              f"{total_params:>10,}    {param_ratio:>6.2f}x  {target_mark}")

print(f"\n{'='*80}")
print("Results Summary")
print(f"{'='*80}\n")

# Sort by param ratio
results.sort(key=lambda x: x['param_ratio'])

if results:
    print(f"{'Config':<30} {'Final SNR':<12} {'Params':<15} {'Ratio':<10} {'vs Baseline'}")
    print("-"*85)
    for r in results[:10]:
        config = f"BT:{r['block_size']}/{r['bt_rank']:.1f} SVD:{r['svd_rank']}"
        vs_baseline = f"{r['param_ratio'] - baseline_ratio:+.2f}x"
        print(f"{config:<30} {r['s2_snr']:>6.2f} dB    {r['total_params']:>10,}    "
              f"{r['param_ratio']:>6.2f}x   {vs_baseline}")

    best = results[0]
    print(f"\n🎯 Best Cascade Result:")
    print(f"   Config: block_size={best['block_size']}, bt_rank={best['bt_rank']}, svd_rank={best['svd_rank']}")
    print(f"   SNR: {best['s2_snr']:.2f} dB")
    print(f"   Params: {best['total_params']:,} ({best['param_ratio']:.2f}x)")

    print(f"\n📊 Comparison:")
    print(f"   Pure SVD:  {baseline_params:,} ({baseline_ratio:.2f}x) for 35+ dB")
    print(f"   Cascade:   {best['total_params']:,} ({best['param_ratio']:.2f}x) for 35+ dB")

    if best['param_ratio'] < baseline_ratio:
        improvement = baseline_ratio - best['param_ratio']
        print(f"   ✅ Cascade is better by {improvement:.2f}x!")
    elif best['param_ratio'] < 1.0:
        print(f"   ✅ Cascade achieves compression (< 1.0x)!")
    else:
        print(f"   ⚠️  Cascade still requires expansion")

print(f"\n{'='*80}")
