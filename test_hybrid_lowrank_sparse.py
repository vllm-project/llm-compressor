"""Test hybrid low-rank + sparse correction approach."""
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
print("Testing Hybrid Low-Rank + Sparse Correction")
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

# Strategy: Start with low-rank base, add sparse corrections
print(f"\nStep 1: Low-rank base approximation")

# Use rank that provides compression (0.60x params, ~11 dB)
# Then add sparse to reach 30 dB while keeping total under 1.0x
rank = 614  # From earlier test: 0.60x params, 11.33 dB

# SVD
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
print(f"  Rank: {rank}")
print(f"  Low-rank SQNR: {lowrank_sqnr:.2f} dB")
print(f"  Low-rank params: {lowrank_params:,} ({lowrank_params/original_params:.2f}x)")

print(f"\nStep 2: Iterative sparse corrections")

# Compute residual
residual = W - W_lowrank

# Store sparse corrections
sparse_corrections = []
W_current = W_lowrank.clone()
target_sqnr = 30.0

# Try different sparsity levels
for sparsity_ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]:
    # Number of elements to keep
    num_sparse = int(sparsity_ratio * original_params)

    # Compute current residual
    current_residual = W - W_current

    # Find top-k elements by magnitude
    residual_flat = current_residual.abs().reshape(-1)
    topk_values, topk_indices = torch.topk(residual_flat, k=num_sparse)

    # Convert flat indices to 2D
    topk_rows = topk_indices // W.shape[1]
    topk_cols = topk_indices % W.shape[1]

    # Create sparse correction
    sparse_correction = torch.zeros_like(W)
    sparse_correction[topk_rows, topk_cols] = current_residual.reshape(-1)[topk_indices]

    # Update approximation
    W_hybrid = W_lowrank + sparse_correction

    # Evaluate
    with torch.no_grad():
        hybrid_output = input_activations @ W_hybrid.T
    hybrid_sqnr = compute_sqnr(output_activations, hybrid_output)

    # Compute total parameters
    # Low-rank: rank * (M + N) parameters
    # Sparse: num_sparse parameters (+ index overhead not counted here)
    sparse_params = num_sparse
    total_params = lowrank_params + sparse_params
    param_ratio = total_params / original_params

    print(f"  Sparse {sparsity_ratio:.1%}: SQNR = {hybrid_sqnr:6.2f} dB, "
          f"Params = {param_ratio:.2f}x ({total_params:,})")

    if hybrid_sqnr >= target_sqnr:
        success_marker = "🎉 SUCCESS" if param_ratio < 1.0 else "✓"
        reduction_msg = f"with {100*(1-param_ratio):.1f}% parameter reduction" if param_ratio < 1.0 else f"using {param_ratio:.2f}x params"
        print(f"\n{success_marker}: Achieved {hybrid_sqnr:.1f} dB SNR {reduction_msg}!")
        print(f"  - Low-rank base: rank={rank}, {lowrank_params:,} params ({lowrank_params/original_params:.2f}x)")
        print(f"  - Sparse correction: {num_sparse:,} elements ({sparse_params/original_params:.2f}x)")
        print(f"  - Total: {total_params:,} params ({param_ratio:.2f}x original)")
        break

print("\n" + "="*70)
print("Summary of Hybrid Approach")
print("="*70)
print(f"Low-rank base (rank={rank}): {lowrank_sqnr:.2f} dB @ {lowrank_params/original_params:.2f}x params")
print(f"Adding sparse corrections progressively improves SNR")
print(f"Trade-off: more sparse elements = higher SNR but more parameters")
print("="*70)
