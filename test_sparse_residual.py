"""Test sparse residual decomposition: W = (I + Δ_n) @ ... @ (I + Δ_1)"""
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
print("Testing Sparse Residual Decomposition")
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

# Clean up model
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Test parameters
num_samples = 1000
k = 64  # Number of rows/cols to update per iteration
max_iters = 20
target_sqnr = 30.0

# Generate realistic input activations
input_activations = torch.randn(num_samples, in_features) * 0.02
with torch.no_grad():
    output_activations = input_activations @ W.T

print(f"\nConfiguration:")
print(f"  Weight shape: {W.shape}")
print(f"  k (sparse update size): {k}")
print(f"  Target SQNR: {target_sqnr} dB")

# Decompose: W = (I + Δ_n) @ ... @ (I + Δ_1)
# Equivalently: W_n = W, W_{i-1} = (I + Δ_i)^{-1} @ W_i
# We want W_0 = I

# Forward: W = W_0 @ (I + Δ_1) @ (I + Δ_2) @ ... @ (I + Δ_n)
# Start with W_0 = I, then find Δ_i such that W_i = W_{i-1} @ (I + Δ_i) ≈ W_target

deltas = []  # Store sparse Δ matrices

M, N = W.shape
I = torch.eye(M, N, dtype=W.dtype, device=W.device)

# Start from identity, build up to W
W_current = I.clone()
W_target = W.clone()

print(f"\nIteratively building sparse residuals...")

for iter_idx in range(max_iters):
    # Compute residual: R = W_target - W_current
    R = W_target - W_current

    # Find k rows with largest residual norm
    row_norms = torch.norm(R, dim=1)
    top_k_row_indices = torch.topk(row_norms, k=min(k, M)).indices.sort()[0]

    # Find k columns with largest residual norm
    col_norms = torch.norm(R, dim=0)
    top_k_col_indices = torch.topk(col_norms, k=min(k, N)).indices.sort()[0]

    # Create sparse Δ: only update top-k rows and columns
    # Δ @ W_current should approximate R
    # For simplicity: Δ[i,j] = R[i,j] / W_current[i,j] if i in top_k_rows and j in top_k_cols

    # Alternative: fit Δ such that W_current @ (I + Δ) ≈ W_target
    # This means: W_current @ Δ ≈ R
    # So: Δ ≈ W_current^{-1} @ R (not always possible)

    # Simpler approach: Δ = sparse matrix that when added to I and multiplied gives improvement
    # Use least-squares on the selected rows

    # For selected rows, solve: W_current[i] @ (I + Δ) ≈ W_target[i]
    # This gives: W_current[i] @ Δ ≈ W_target[i] - W_current[i] = R[i]

    # Create sparse Δ (only non-zero for top-k rows and top-k columns)
    delta = torch.zeros_like(W)

    # For each selected row, use OLS to find best column updates
    for row_idx in top_k_row_indices:
        # We want: W_current[row_idx] @ (I + Δ) ≈ W_target[row_idx]
        # W_current[row_idx] + W_current[row_idx] @ Δ ≈ W_target[row_idx]
        # W_current[row_idx] @ Δ ≈ R[row_idx]

        # But Δ is sparse (only k columns), so:
        # W_current[row_idx, top_k_cols] @ Δ[top_k_cols, :] ≈ R[row_idx, :]

        # Simplification: just set delta to capture the residual proportionally
        # delta[row_idx, top_k_col_indices] = R[row_idx, top_k_col_indices] / (W_current[row_idx, top_k_col_indices] + 1e-10)
        pass

    # Even simpler: Just store the k×k block of R
    # delta[top_k_row_indices[:, None], top_k_col_indices] = R[top_k_row_indices[:, None], top_k_col_indices]

    # Use OLS to solve: for selected rows, (I + delta)[selected_cols] should give best fit
    # W_current @ (I + delta) = W_current + W_current @ delta ≈ W_target
    # W_current @ delta ≈ R

    # For simplicity, use the subset:
    W_sub = W_current[top_k_row_indices][:, top_k_col_indices]  # (k, k)
    R_sub = R[top_k_row_indices]  # (k, N)

    # Solve: W_sub @ delta_cols ≈ R_sub
    # delta_cols is (k, N) where only the selected columns are used
    delta_cols = torch.linalg.lstsq(W_sub, R_sub).solution  # (k, N)

    # Store sparse delta
    for i, col_idx in enumerate(top_k_col_indices):
        delta[col_idx, :] = delta_cols[i, :]

    deltas.append({
        'col_indices': top_k_col_indices.clone(),
        'delta_cols': delta_cols.clone(),  # (k, N)
    })

    # Update W_current: W_current = W_current @ (I + delta)
    W_current = W_current @ (I + delta)

    # Reconstruct W from deltas
    W_reconstructed = I.clone()
    for d in deltas:
        delta_mat = torch.zeros_like(W)
        for i, col_idx in enumerate(d['col_indices']):
            delta_mat[col_idx, :] = d['delta_cols'][i, :]
        W_reconstructed = W_reconstructed @ (I + delta_mat)

    # Test output
    with torch.no_grad():
        approx_output = input_activations @ W_reconstructed.T

    sqnr = compute_sqnr(output_activations, approx_output)

    # Compute parameter count
    params_stored = sum(
        len(d['col_indices']) + d['delta_cols'].numel()
        for d in deltas
    )
    original_params = W.numel()
    param_ratio = params_stored / original_params

    # Compute remaining error
    remaining_error = torch.norm(W_target - W_current).item()

    print(f"  Iter {iter_idx}: SQNR = {sqnr:6.2f} dB, "
          f"Params = {param_ratio:.2f}x ({params_stored:,} / {original_params:,}), "
          f"Remaining error = {remaining_error:.2f}")

    if sqnr >= target_sqnr:
        print(f"\n✅ Target SQNR of {target_sqnr} dB achieved!")
        break

    if remaining_error < 1.0:
        print(f"\n✅ Converged to target!")
        break

print(f"\nFinal Results:")
print(f"  Iterations: {len(deltas)}")
print(f"  SQNR: {sqnr:.2f} dB")
print(f"  Parameters: {param_ratio:.2f}x original ({params_stored:,} params)")

if sqnr >= 30 and param_ratio < 1.0:
    print(f"\n🎉 SUCCESS: Achieved {sqnr:.1f} dB SNR with {100*(1-param_ratio):.1f}% parameter reduction!")
elif sqnr >= 30:
    print(f"\n✓ Achieved {sqnr:.1f} dB SNR but used {param_ratio:.2f}x params")
else:
    print(f"\n⚠️  Only achieved {sqnr:.1f} dB SNR (target: 30+ dB)")

print("="*70)
