"""Test hybrid low-rank + iterative sparse approach."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

def compute_sqnr(original_output, approx_output):
    """Compute SQNR in dB."""
    y_true = original_output.detach().float()
    y_pred = approx_output.detach().float()
    signal_power = torch.var(y_true)
    mse_noise = torch.mean((y_true - y_pred) ** 2)
    sqnr_linear = signal_power / (mse_noise + 1e-10)
    sqnr_db = 10 * torch.log10(sqnr_linear)
    return sqnr_db.item()

print("="*70)
print("Testing Hybrid Low-Rank + Iterative Sparse")
print("="*70)

# Load real model weights
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Get first q_proj layer
q_proj = model.model.layers[0].self_attn.q_proj
W = q_proj.weight.data.float().clone()
in_features = q_proj.in_features
out_features = q_proj.out_features

print(f"Loaded q_proj weight: {W.shape}")

# Clean up
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Generate realistic input activations
num_samples = 1000
input_activations = torch.randn(num_samples, in_features) * 0.02
with torch.no_grad():
    output_activations = input_activations @ W.T

original_params = W.numel()

print(f"\nConfiguration:")
print(f"  Weight shape: {W.shape}")
print(f"  Original params: {original_params:,}")

# Strategy:
# 1. Low-rank base to capture bulk structure (cheap)
# 2. Iterative sparse on residual to capture fine details

# Test different low-rank base sizes
for rank_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
    rank = int(rank_ratio * min(in_features, out_features))

    print(f"\n{'='*70}")
    print(f"Testing with rank={rank} ({rank_ratio:.1%} of dimension)")
    print(f"{'='*70}")

    # Step 1: Low-rank base
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    W_lowrank = U_r @ torch.diag(S_r) @ Vh_r

    # Evaluate low-rank base
    with torch.no_grad():
        lowrank_output = input_activations @ W_lowrank.T
    lowrank_sqnr = compute_sqnr(output_activations, lowrank_output)

    lowrank_params = rank * (out_features + in_features)
    print(f"Low-rank base: {lowrank_sqnr:.2f} dB, {lowrank_params:,} params ({lowrank_params/original_params:.2f}x)")

    # Step 2: Iterative sparse on residual
    W_residual = W - W_lowrank
    W_sparse = torch.zeros_like(W)

    print(f"\nIterative sparse on residual:")
    print(f"{'Iteration':<10} {'Sparsity':<12} {'SQNR (dB)':<12} {'Total Params'}")
    print("-"*60)

    target_sqnr = 30.0
    k_per_iter = int(0.025 * original_params)  # 2.5% per iteration

    for iter_idx in range(40):
        # Find top-k elements by magnitude in residual
        residual_flat = W_residual.abs().reshape(-1)
        k_actual = min(k_per_iter, (residual_flat > 0).sum().item())

        if k_actual == 0:
            break

        topk_values, topk_indices = torch.topk(residual_flat, k=k_actual)

        # Convert flat indices to 2D
        topk_rows = topk_indices // W.shape[1]
        topk_cols = topk_indices % W.shape[1]

        # Add to sparse representation
        W_sparse[topk_rows, topk_cols] = W_residual.reshape(-1)[topk_indices]

        # Update residual
        W_residual[topk_rows, topk_cols] = 0

        # Combined approximation: low-rank + sparse
        W_hybrid = W_lowrank + W_sparse

        # Evaluate
        with torch.no_grad():
            hybrid_output = input_activations @ W_hybrid.T
        sqnr = compute_sqnr(output_activations, hybrid_output)

        # Count parameters
        num_sparse = (W_sparse != 0).sum().item()
        sparse_params = num_sparse
        total_params = lowrank_params + sparse_params
        total_ratio = total_params / original_params

        if iter_idx % 5 == 0 or sqnr >= target_sqnr:
            print(f"{iter_idx:<10} {num_sparse/original_params:<12.1%} {sqnr:<12.2f} {total_params:,} ({total_ratio:.2f}x)")

        if sqnr >= target_sqnr:
            print(f"\n✅ Target SQNR of {target_sqnr} dB achieved!")
            print(f"  Low-rank: rank={rank}, {lowrank_params:,} params ({lowrank_params/original_params:.2f}x)")
            print(f"  Sparse: {num_sparse:,} elements ({sparse_params/original_params:.2f}x)")
            print(f"  Total: {total_params:,} params ({total_ratio:.2f}x)")
            print(f"  Compression: {100*(1-total_ratio):.1f}% reduction")
            break

print("\n" + "="*70)
