"""Test block-diagonal correction approach for matrix compression."""
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

def row_deviation_from_identity(W):
    """Compute deviation of each row from corresponding identity row."""
    M, N = W.shape
    I = torch.zeros_like(W)
    diag_len = min(M, N)
    I[range(diag_len), range(diag_len)] = 1.0
    deviation = torch.norm(W - I, dim=1)
    return deviation

print("="*70)
print("Testing Block Correction Compression")
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
W = q_proj.weight.data.float().clone()  # (out_features, in_features)
in_features = q_proj.in_features
out_features = q_proj.out_features

print(f"Loaded q_proj weight: {W.shape}")

# Test parameters
num_samples = 1000
k = 64  # Block size for correction
max_iters = 20
target_sqnr = 30.0

# Clean up model to free memory
del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Generate realistic input activations (similar to hidden states)
# Use normal distribution with smaller std (typical for normalized hidden states)
input_activations = torch.randn(num_samples, in_features) * 0.02
with torch.no_grad():
    output_activations = input_activations @ W.T

print(f"\nConfiguration:")
print(f"  Weight shape: {W.shape}")
print(f"  k (block size): {k}")
print(f"  Target SQNR: {target_sqnr} dB")

# Idea: Store corrections C_i where C_i @ W brings W closer to identity
# W_new = C_i @ W_old, so W_old = inv(C_i) @ W_new
# After n iterations: W = inv(C_1) @ inv(C_2) @ ... @ inv(C_n) @ I
# Forward: x @ W = x @ inv(C_1) @ inv(C_2) @ ... @ inv(C_n)

# Each C_i is block-diagonal, only affecting k rows
corrections = []  # Store C_i as sparse blocks

M, N = W.shape
I = torch.zeros_like(W)
diag_len = min(M, N)
I[range(diag_len), range(diag_len)] = 1.0

W_current = W.clone()

print(f"\nIteratively finding block corrections...")

for iter_idx in range(max_iters):
    # Find k rows with largest deviation from identity
    deviation = row_deviation_from_identity(W_current)
    k_actual = min(k, (deviation > 1e-6).sum().item())
    if k_actual == 0:
        print(f"  Iter {iter_idx}: All rows converged to identity")
        break

    top_k_indices = torch.topk(deviation, k=k_actual).indices.sort()[0]  # Sort for consistency

    # Extract the k×N block of rows that deviate most
    W_block = W_current[top_k_indices, :]  # (k, N)
    I_block = I[top_k_indices, :]  # (k, N)

    # Goal: Find C (k×k) such that C @ W_block = I_block
    # This means: C = I_block @ W_block^T @ inv(W_block @ W_block^T)
    # But W_block is k×N (not square), so we can't directly invert

    # Alternative: Use least-squares to find C
    # We want C @ W_block ≈ I_block
    # This is equivalent to: C = I_block @ pinv(W_block)

    C_block = torch.linalg.lstsq(W_block.T, I_block.T).solution.T  # (k, k)

    # Store the correction
    correction = {
        'indices': top_k_indices.clone(),
        'C': C_block.clone(),  # (k, k) matrix
    }
    corrections.append(correction)

    # Apply correction: update the k rows
    W_current = W_current.clone()
    W_current[top_k_indices] = C_block @ W_block

    # Reconstruct W from corrections
    # Start with identity, then apply inv(C_i) in reverse order
    W_reconstructed = I.clone()

    for corr in reversed(corrections):
        # Apply inv(C_i) to the k rows
        indices = corr['indices']
        C_inv = torch.linalg.inv(corr['C'])  # (k, k)
        W_reconstructed[indices] = C_inv @ W_reconstructed[indices]

    # Test output
    with torch.no_grad():
        approx_output = input_activations @ W_reconstructed.T

    sqnr = compute_sqnr(output_activations, approx_output)

    # Compute parameter count
    params_stored = sum(
        len(corr['indices']) + corr['C'].numel()
        for corr in corrections
    )
    original_params = W.numel()
    param_ratio = params_stored / original_params

    # Compute remaining deviation
    remaining_deviation = row_deviation_from_identity(W_current).sum().item()

    print(f"  Iter {iter_idx}: SQNR = {sqnr:6.2f} dB, "
          f"Params = {param_ratio:.2f}x ({params_stored:,} / {original_params:,}), "
          f"Remaining deviation = {remaining_deviation:.2f}")

    if sqnr >= target_sqnr:
        print(f"\n✅ Target SQNR of {target_sqnr} dB achieved!")
        break

    if remaining_deviation < 1e-6:
        print(f"\n✅ Converged to identity!")
        break

print(f"\nFinal Results:")
print(f"  Iterations: {len(corrections)}")
print(f"  SQNR: {sqnr:.2f} dB")
print(f"  Parameters: {param_ratio:.2f}x original ({params_stored:,} params)")

if sqnr >= 30 and param_ratio < 1.0:
    print(f"\n🎉 SUCCESS: Achieved {sqnr:.1f} dB SNR with {100*(1-param_ratio):.1f}% parameter reduction!")
elif sqnr >= 30:
    print(f"\n✓ Achieved {sqnr:.1f} dB SNR but used {param_ratio:.2f}x params")
else:
    print(f"\n⚠️  Only achieved {sqnr:.1f} dB SNR (target: 30+ dB)")

print("="*70)
