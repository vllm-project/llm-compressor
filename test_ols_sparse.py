"""Test OLS-based sparse selection for better compression."""
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
print("Testing OLS-Based Sparse Selection")
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

# OLS-based sparse selection:
# Instead of selecting by magnitude, select elements that minimize
# reconstruction error when fitted via least squares

print(f"\nOLS-based column selection (activation-aware importance)...")
print(f"Idea: Select input columns (features) that best predict outputs via OLS")
print(f"{'Iteration':<10} {'Cols Used':<12} {'SQNR (dB)':<12} {'Sparsity'}")
print("="*60)

# Instead of selecting individual elements, select entire input columns
# This creates block-sparsity and reduces index overhead

# Track which input columns are selected
selected_cols = []
target_sqnr = 30.0

# Number of columns to add per iteration
k_cols_per_iter = 32

for iter_idx in range(100):
    # Find best column to add using OLS residual reduction

    if len(selected_cols) == 0:
        # First iteration: find single best column
        best_col = None
        best_error = float('inf')

        # Try each column
        for col_idx in range(in_features):
            # Fit: output = input[:, col_idx] @ W[col_idx, :]
            # OLS: W[col_idx, :] = lstsq(input[:, col_idx], output)
            X_col = input_activations[:, col_idx:col_idx+1]  # (num_samples, 1)
            W_col = torch.linalg.lstsq(X_col, output_activations).solution  # (1, out_features)

            # Compute reconstruction error
            output_approx = X_col @ W_col
            error = torch.norm(output_activations - output_approx) ** 2

            if error < best_error:
                best_error = error
                best_col = col_idx

        selected_cols = [best_col]
    else:
        # Add k_cols_per_iter columns per iteration
        candidates = [c for c in range(in_features) if c not in selected_cols]

        if len(candidates) == 0:
            break

        # Try adding each candidate and measure improvement
        best_improvement = -float('inf')
        best_new_cols = []

        # Current reconstruction with selected columns
        X_current = input_activations[:, selected_cols]  # (num_samples, len(selected_cols))
        W_current = torch.linalg.lstsq(X_current, output_activations).solution
        current_output = X_current @ W_current
        current_error = torch.norm(output_activations - current_output) ** 2

        # Batch evaluation: try adding k_cols_per_iter at once
        # Use residual correlation to select
        residual = output_activations - current_output

        # For each candidate column, compute correlation with residual
        correlations = []
        for col_idx in candidates:
            X_col = input_activations[:, col_idx]  # (num_samples,)
            # Correlation: sum over samples and output dims
            corr = torch.abs((X_col.unsqueeze(1) * residual).sum(dim=0)).sum()
            correlations.append((corr.item(), col_idx))

        # Sort by correlation and take top k
        correlations.sort(reverse=True)
        new_cols = [col for _, col in correlations[:k_cols_per_iter]]
        selected_cols.extend(new_cols)

    # Refit with selected columns
    X_selected = input_activations[:, selected_cols]
    W_selected = torch.linalg.lstsq(X_selected, output_activations).solution

    # Reconstruct W as sparse matrix
    W_sparse = torch.zeros_like(W)
    W_sparse[:, selected_cols] = W_selected.T

    # Evaluate
    with torch.no_grad():
        sparse_output = input_activations @ W_sparse.T
    sqnr = compute_sqnr(output_activations, sparse_output)

    # Sparsity: only selected columns are non-zero
    num_params = len(selected_cols) * out_features
    sparsity = num_params / original_params

    print(f"{iter_idx:<10} {len(selected_cols):<12} {sqnr:<12.2f} {sparsity:.1%}")

    if sqnr >= target_sqnr:
        print(f"\n✅ Target SQNR of {target_sqnr} dB achieved!")
        print(f"  Selected columns: {len(selected_cols)} / {in_features}")
        print(f"  Parameters: {num_params:,} ({sparsity:.1%})")
        print(f"  Compression: {100*(1-sparsity):.1f}% reduction")
        break

    if len(selected_cols) >= in_features:
        break

print("="*70)
