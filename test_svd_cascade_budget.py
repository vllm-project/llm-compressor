"""Test SVD + cascaded compression with fixed param budget (0.5x).

Goal: Keep total params <= 0.5x original while maximizing SNR.

Strategies to test:
1. SVD → BlockTensorized → Low-rank (SVD-first)
2. BlockTensorized → SVD → Low-rank (BT-first)
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
print("Testing SVD + Cascaded Compression with 0.5x Param Budget")
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
target_budget = int(0.5 * original_params)  # 2,097,152 params

print(f"Target param budget (0.5x): {target_budget:,}")

# Generate synthetic activations
num_samples = 256
input_activations = torch.randn(num_samples, in_features) * 0.02

# Perform SVD once (reuse for all tests)
print("\nPerforming SVD...")
U, S, Vh = torch.linalg.svd(W, full_matrices=False)

# Baseline: Pure SVD at budget
print(f"\nBaseline - Pure SVD at 0.5x budget:")
svd_baseline_rank = target_budget // (in_features + out_features)
U_b = U[:, :svd_baseline_rank]
S_b = S[:svd_baseline_rank]
Vh_b = Vh[:svd_baseline_rank, :]
W_svd_baseline = U_b @ torch.diag(S_b) @ Vh_b
baseline_snr = compute_sqnr(W, W_svd_baseline)
baseline_params = svd_baseline_rank * (in_features + out_features)
print(f"  Rank: {svd_baseline_rank}")
print(f"  Params: {baseline_params:,} ({baseline_params/original_params:.2f}x)")
print(f"  SNR: {baseline_snr:.2f} dB")

print(f"\n{'='*80}")
print("Strategy 1: SVD → BlockTensorized (residual) → Low-rank (residual)")
print(f"{'='*80}\n")

print(f"{'SVD':<6} {'BT blk':<8} {'BT rnk':<8} {'LR':<6} {'Stage1':<10} {'Stage2':<10} {'Final SNR':<11} {'Params':<15} {'Ratio':<8} {'Budget?'}")
print("-"*110)

results_svd_first = []

# Allocate budget: Try different allocations that sum to ~0.5x
# Strategy: larger base, smaller residuals
svd_ranks = [200, 250, 300, 350, 400]  # 0.2x, 0.24x, 0.29x, 0.34x, 0.39x
bt_configs = [(512, 0.2), (512, 0.3), (1024, 0.2)]  # Small BT for residual
lr_ranks = [25, 50, 75, 100]  # Small final correction

for svd_rank in svd_ranks:
    # Stage 1: SVD
    U_r = U[:, :svd_rank]
    S_r = S[:svd_rank]
    Vh_r = Vh[:svd_rank, :]
    W_svd = U_r @ torch.diag(S_r) @ Vh_r

    residual1 = W - W_svd
    s1_snr = compute_sqnr(W, W_svd)
    s1_params = svd_rank * (in_features + out_features)

    for block_size, bt_rank in bt_configs:
        # Stage 2: BlockTensorized on residual
        temp_linear = nn.Linear(in_features, out_features, bias=False)
        temp_linear.weight.data = residual1.clone()

        try:
            bt = BlockTensorizedLinear.from_linear(
                temp_linear,
                block_size=block_size,
                rank=bt_rank,
                num_cores=3,
                input_activations=input_activations,
            )
        except Exception as e:
            continue

        W_bt = bt.to_matrix().float()
        residual2 = residual1 - W_bt

        W_combined_s2 = W_svd + W_bt
        s2_snr = compute_sqnr(W, W_combined_s2)
        s2_params = bt.num_params

        for lr_rank in lr_ranks:
            # Stage 3: Low-rank on residual2
            U2, S2, Vh2 = torch.linalg.svd(residual2, full_matrices=False)
            U2_r = U2[:, :lr_rank]
            S2_r = S2[:lr_rank]
            Vh2_r = Vh2[:lr_rank, :]
            W_lr = U2_r @ torch.diag(S2_r) @ Vh2_r

            W_final = W_svd + W_bt + W_lr
            final_snr = compute_sqnr(W, W_final)

            lr_params = lr_rank * (in_features + out_features)
            total_params = s1_params + s2_params + lr_params
            param_ratio = total_params / original_params

            within_budget = "✓" if total_params <= target_budget else ""

            result = {
                'strategy': 'SVD→BT→LR',
                'svd_rank': svd_rank,
                'bt_block': block_size,
                'bt_rank': bt_rank,
                'lr_rank': lr_rank,
                's1_snr': s1_snr,
                's2_snr': s2_snr,
                'final_snr': final_snr,
                'total_params': total_params,
                'param_ratio': param_ratio,
            }
            results_svd_first.append(result)

            if within_budget:
                print(f"{svd_rank:<6} {block_size:<8} {bt_rank:<8.1f} {lr_rank:<6} "
                      f"{s1_snr:>6.2f} dB  {s2_snr:>6.2f} dB  {final_snr:>6.2f} dB   "
                      f"{total_params:>10,}    {param_ratio:>6.2f}x  {within_budget}")

print(f"\n{'='*80}")
print("Strategy 2: BlockTensorized → SVD (residual) → Low-rank (residual)")
print(f"{'='*80}\n")

print(f"{'BT blk':<8} {'BT rnk':<8} {'SVD':<6} {'LR':<6} {'Stage1':<10} {'Stage2':<10} {'Final SNR':<11} {'Params':<15} {'Ratio':<8} {'Budget?'}")
print("-"*110)

results_bt_first = []

# Allocate budget: Try BT first then SVD
bt_configs_2 = [(512, 0.3), (512, 0.4), (1024, 0.3)]  # Start with BT
svd_ranks_2 = [100, 150, 200, 250]  # SVD on residual
lr_ranks_2 = [25, 50, 75]  # Small final correction

for block_size, bt_rank in bt_configs_2:
    # Stage 1: BlockTensorized
    try:
        bt = BlockTensorizedLinear.from_linear(
            q_proj,
            block_size=block_size,
            rank=bt_rank,
            num_cores=3,
            input_activations=input_activations,
        )
    except Exception as e:
        continue

    W_bt = bt.to_matrix().float()
    residual1 = W - W_bt
    s1_snr = compute_sqnr(W, W_bt)
    s1_params = bt.num_params

    for svd_rank in svd_ranks_2:
        # Stage 2: SVD on residual
        U_r, S_r, Vh_r = torch.linalg.svd(residual1, full_matrices=False)
        U_r = U_r[:, :svd_rank]
        S_r = S_r[:svd_rank]
        Vh_r = Vh_r[:svd_rank, :]
        W_svd = U_r @ torch.diag(S_r) @ Vh_r

        residual2 = residual1 - W_svd
        W_combined_s2 = W_bt + W_svd
        s2_snr = compute_sqnr(W, W_combined_s2)
        s2_params = svd_rank * (in_features + out_features)

        for lr_rank in lr_ranks_2:
            # Stage 3: Low-rank on residual2
            U2, S2, Vh2 = torch.linalg.svd(residual2, full_matrices=False)
            U2_r = U2[:, :lr_rank]
            S2_r = S2[:lr_rank]
            Vh2_r = Vh2[:lr_rank, :]
            W_lr = U2_r @ torch.diag(S2_r) @ Vh2_r

            W_final = W_bt + W_svd + W_lr
            final_snr = compute_sqnr(W, W_final)

            lr_params = lr_rank * (in_features + out_features)
            total_params = s1_params + s2_params + lr_params
            param_ratio = total_params / original_params

            within_budget = "✓" if total_params <= target_budget else ""

            result = {
                'strategy': 'BT→SVD→LR',
                'bt_block': block_size,
                'bt_rank': bt_rank,
                'svd_rank': svd_rank,
                'lr_rank': lr_rank,
                's1_snr': s1_snr,
                's2_snr': s2_snr,
                'final_snr': final_snr,
                'total_params': total_params,
                'param_ratio': param_ratio,
            }
            results_bt_first.append(result)

            if within_budget:
                print(f"{block_size:<8} {bt_rank:<8.1f} {svd_rank:<6} {lr_rank:<6} "
                      f"{s1_snr:>6.2f} dB  {s2_snr:>6.2f} dB  {final_snr:>6.2f} dB   "
                      f"{total_params:>10,}    {param_ratio:>6.2f}x  {within_budget}")

# Combine all results
all_results = results_svd_first + results_bt_first

# Filter to budget
budget_results = [r for r in all_results if r['total_params'] <= target_budget]
budget_results.sort(key=lambda x: -x['final_snr'])

print(f"\n{'='*80}")
print(f"Best Results Within 0.5x Param Budget ({target_budget:,} params)")
print(f"{'='*80}\n")

if budget_results:
    print(f"{'Strategy':<15} {'Config':<40} {'Final SNR':<12} {'Params':<15} {'Ratio'}")
    print("-"*100)
    for r in budget_results[:15]:
        if r['strategy'] == 'SVD→BT→LR':
            config = f"SVD:{r['svd_rank']} BT:{r['bt_block']}/{r['bt_rank']:.1f} LR:{r['lr_rank']}"
        else:
            config = f"BT:{r['bt_block']}/{r['bt_rank']:.1f} SVD:{r['svd_rank']} LR:{r['lr_rank']}"

        print(f"{r['strategy']:<15} {config:<40} {r['final_snr']:>6.2f} dB    {r['total_params']:>10,}    {r['param_ratio']:>6.2f}x")

    best = budget_results[0]
    print(f"\n🎯 Best Result at 0.5x Budget:")
    print(f"   Strategy: {best['strategy']}")
    print(f"   Final SNR: {best['final_snr']:.2f} dB")
    print(f"   Total Params: {best['total_params']:,} ({best['param_ratio']:.2f}x)")
    print(f"\n   Baseline (Pure SVD at {svd_baseline_rank}): {baseline_snr:.2f} dB")
    print(f"   Improvement: {best['final_snr'] - baseline_snr:.2f} dB")

    if best['final_snr'] > baseline_snr:
        print(f"\n   ✅ Cascade approach beats pure SVD!")
    else:
        print(f"\n   ⚠️  Pure SVD is still better at this param budget")
else:
    print("⚠️  No configurations fit within budget")

print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")
print(f"Target budget: {target_budget:,} params (0.5x)")
print(f"Total configs tested: {len(all_results)}")
print(f"Configs within budget: {len(budget_results)}")
print(f"{'='*80}")
