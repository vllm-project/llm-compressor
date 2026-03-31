"""Test k_proj and v_proj compression - do they have lower rank than q_proj?"""
import torch
from transformers import AutoModelForCausalLM

def compute_sqnr(original, approximation):
    """Compute SQNR in dB."""
    signal_power = torch.var(original)
    mse_noise = torch.mean((original - approximation) ** 2)
    sqnr_linear = signal_power / (mse_noise + 1e-10)
    sqnr_db = 10 * torch.log10(sqnr_linear)
    return sqnr_db.item()

print("="*80)
print("Testing k_proj and v_proj: Do they compress better than q_proj?")
print("="*80)

# Load model
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Get all attention projection layers
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

print("\n" + "="*80)
print("Pure SVD Compression Test")
print("="*80)

# Test ranks that correspond to different compression ratios
test_configs = [
    # (rank, target_ratio_description)
    (100, "0.24x for k/v, 0.10x for q/o"),
    (200, "0.48x for k/v, 0.20x for q/o"),
    (300, "0.72x for k/v, 0.29x for q/o"),
    (400, "0.95x for k/v, 0.39x for q/o"),
    (500, "1.19x for k/v, 0.49x for q/o"),
    (512, "~0.50x for q/o"),
]

for rank, description in test_configs:
    print(f"\n{'='*80}")
    print(f"Rank {rank} ({description})")
    print(f"{'='*80}")
    print(f"{'Layer':<10} {'Original':<15} {'Compressed':<15} {'Ratio':<10} {'SNR':<12} {'35dB?'}")
    print("-"*80)

    for layer_name, layer in layers_to_test.items():
        W = layer.weight.data.float().clone()
        original_params = W.numel()

        # Check if rank is valid for this matrix
        min_dim = min(W.shape[0], W.shape[1])
        if rank > min_dim:
            print(f"{layer_name:<10} {original_params:>10,}    {'SKIP':>12}  {'(rank > min_dim)':>8}  {'--':>10}")
            continue

        # SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # Low-rank approximation
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        W_approx = U_r @ torch.diag(S_r) @ Vh_r

        # Compute SNR
        snr = compute_sqnr(W, W_approx)

        # Params: rank * (m + n)
        compressed_params = rank * (W.shape[0] + W.shape[1])
        param_ratio = compressed_params / original_params

        target_mark = "✓" if snr >= 35.0 else ""
        compressed_mark = "🎯" if snr >= 35.0 and param_ratio < 1.0 else ""

        print(f"{layer_name:<10} {original_params:>10,}    {compressed_params:>10,}    "
              f"{param_ratio:>6.2f}x  {snr:>6.2f} dB   {target_mark} {compressed_mark}")

print("\n" + "="*80)
print("Finding Minimum Rank for 35 dB SNR (Binary Search)")
print("="*80)
print(f"{'Layer':<10} {'Min Rank':<12} {'Params':<15} {'Ratio':<10} {'SNR':<12} {'Status'}")
print("-"*80)

for layer_name, layer in layers_to_test.items():
    W = layer.weight.data.float().clone()
    original_params = W.numel()
    min_dim = min(W.shape[0], W.shape[1])

    # SVD once
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Binary search for minimum rank to achieve 35 dB
    left, right = 1, min_dim
    best_rank = None
    best_snr = 0

    while left <= right:
        mid = (left + right) // 2

        # Test this rank
        U_r = U[:, :mid]
        S_r = S[:mid]
        Vh_r = Vh[:mid, :]
        W_approx = U_r @ torch.diag(S_r) @ Vh_r
        snr = compute_sqnr(W, W_approx)

        if snr >= 35.0:
            best_rank = mid
            best_snr = snr
            right = mid - 1  # Try smaller rank
        else:
            left = mid + 1  # Need larger rank

    if best_rank is not None:
        compressed_params = best_rank * (W.shape[0] + W.shape[1])
        param_ratio = compressed_params / original_params

        status = "✅ Compressed" if param_ratio < 1.0 else "❌ Expanded"

        print(f"{layer_name:<10} {best_rank:<12} {compressed_params:>10,}    "
              f"{param_ratio:>6.2f}x  {best_snr:>6.2f} dB   {status}")
    else:
        print(f"{layer_name:<10} {'IMPOSSIBLE':<12} {'--':>10}    {'--':>8}  {'--':>10}   ❌ Cannot reach 35 dB")

print("\n" + "="*80)
print("Summary: Best Compression Ratios for 35 dB SNR")
print("="*80)

results = []
for layer_name, layer in layers_to_test.items():
    W = layer.weight.data.float().clone()
    original_params = W.numel()
    min_dim = min(W.shape[0], W.shape[1])

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    # Binary search
    left, right = 1, min_dim
    best_rank = None

    while left <= right:
        mid = (left + right) // 2
        U_r = U[:, :mid]
        S_r = S[:mid]
        Vh_r = Vh[:mid, :]
        W_approx = U_r @ torch.diag(S_r) @ Vh_r
        snr = compute_sqnr(W, W_approx)

        if snr >= 35.0:
            best_rank = mid
            right = mid - 1
        else:
            left = mid + 1

    if best_rank is not None:
        compressed_params = best_rank * (W.shape[0] + W.shape[1])
        param_ratio = compressed_params / original_params
        results.append({
            'layer': layer_name,
            'rank': best_rank,
            'params': compressed_params,
            'ratio': param_ratio,
            'achieves_target': param_ratio < 1.0
        })

if results:
    # Sort by compression ratio
    results.sort(key=lambda x: x['ratio'])

    print("\nLayers sorted by compression ratio (best to worst):")
    for r in results:
        status = "✅ Target!" if r['achieves_target'] else "Needs expansion"
        print(f"  {r['layer']:<10} rank={r['rank']:<5} params={r['params']:>10,}  "
              f"({r['ratio']:.2f}x)  {status}")

    # Check if any achieve target
    achievers = [r for r in results if r['achieves_target']]
    if achievers:
        print(f"\n🎉 SUCCESS: {len(achievers)} layer(s) can reach 35 dB with compression:")
        for r in achievers:
            compression_pct = 100 * (1 - r['ratio'])
            print(f"  - {r['layer']}: {compression_pct:.1f}% compression at 35+ dB SNR")
    else:
        print("\n⚠️  No layers can achieve 35 dB SNR with < 1.0x parameters")
        best = results[0]
        print(f"   Best layer is {best['layer']} requiring {best['ratio']:.2f}x params")
else:
    print("\n❌ No layers can achieve 35 dB SNR at any rank")

print("="*80)
