"""Test iterative sparse-only compression."""
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
print("Testing Iterative Sparse-Only Compression")
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

# Iterative sparse compression:
# Each iteration, select k most important elements from residual
# Store them, then compute new residual

print(f"\nIterative sparse selection (greedy top-k)...")
print(f"{'Iteration':<10} {'Sparsity':<12} {'SQNR (dB)':<12} {'Params'}")
print("="*60)

W_sparse = torch.zeros_like(W)
W_residual = W.clone()
target_sqnr = 30.0

# Number of elements to add per iteration
k_per_iter = int(0.05 * original_params)  # 5% per iteration

for iter_idx in range(20):
    # Find top-k elements by magnitude in residual
    residual_flat = W_residual.abs().reshape(-1)

    # Get top-k
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

    # Evaluate
    with torch.no_grad():
        sparse_output = input_activations @ W_sparse.T
    sqnr = compute_sqnr(output_activations, sparse_output)

    # Count non-zero elements
    num_nonzero = (W_sparse != 0).sum().item()
    sparsity = num_nonzero / original_params

    print(f"{iter_idx:<10} {sparsity:<12.1%} {sqnr:<12.2f} {num_nonzero:,}")

    if sqnr >= target_sqnr:
        print(f"\n✅ Target SQNR of {target_sqnr} dB achieved!")
        print(f"  Sparse elements: {num_nonzero:,} ({sparsity:.1%} of original)")
        print(f"  Compression: {100*(1-sparsity):.1f}% reduction")
        break

print("="*70)
