"""Test identity refinement approach for matrix compression."""
import torch
import torch.nn as nn

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
    """
    Compute deviation of each row from corresponding identity row.
    For M×N matrix, identity has 1 on diagonal (min(M,N)) and 0 elsewhere.
    """
    M, N = W.shape
    # Create identity pattern
    I = torch.zeros_like(W)
    diag_len = min(M, N)
    I[range(diag_len), range(diag_len)] = 1.0

    # Compute row-wise L2 deviation
    deviation = torch.norm(W - I, dim=1)  # (M,)
    return deviation

def low_rank_approx(X, rank):
    """Low-rank approximation via SVD."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    rank = min(rank, len(S))
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]
    return U_r @ torch.diag(S_r) @ Vh_r, U_r, S_r, Vh_r

print("="*70)
print("Testing Identity Refinement Compression")
print("="*70)

# Test parameters
in_features = 256
out_features = 128
num_samples = 1000
k = 16  # Number of rows to update per iteration
rank = 16  # Rank for low-rank correction of k rows (full rank)
max_iters = 20
target_sqnr = 30.0

# Create linear layer and data
linear = nn.Linear(in_features, out_features, bias=False)
W = linear.weight.data.clone()  # (out_features, in_features) = (128, 256)

input_activations = torch.randn(num_samples, in_features)
with torch.no_grad():
    output_activations = linear(input_activations)

print(f"\nConfiguration:")
print(f"  Weight shape: {W.shape}")
print(f"  k (rows per iteration): {k}")
print(f"  rank (for low-rank correction): {rank}")
print(f"  Target SQNR: {target_sqnr} dB")

# Track correction matrices
corrections = []  # Each correction stores indices + low-rank factors

W_current = W.clone()

print(f"\nIteratively finding corrections...")

for iter_idx in range(max_iters):
    # Find k rows with largest deviation from identity
    deviation = row_deviation_from_identity(W_current)
    k_actual = min(k, (deviation > 1e-6).sum().item())  # Only correct rows that need it
    if k_actual == 0:
        print(f"  Iter {iter_idx}: All rows converged to identity")
        break

    top_k_indices = torch.topk(deviation, k=k_actual).indices

    # Create identity matrix for baseline
    M, N = W.shape
    I = torch.zeros_like(W)
    diag_len = min(M, N)
    I[range(diag_len), range(diag_len)] = 1.0

    # Compute deviation of selected rows from identity
    delta = W_current[top_k_indices] - I[top_k_indices]  # (k, N)

    # Low-rank approximation of delta
    _, U_r, S_r, Vh_r = low_rank_approx(delta, rank)

    # Store correction as low-rank factors
    correction = {
        'indices': top_k_indices.clone(),
        'U': U_r.clone(),  # (k, rank)
        'S': S_r.clone(),  # (rank,)
        'Vh': Vh_r.clone(),  # (rank, N)
    }
    corrections.append(correction)

    # Update W_current: set corrected rows to identity
    W_current = W_current.clone()
    W_current[top_k_indices] = I[top_k_indices]

    # Reconstruct W from corrections
    W_reconstructed = torch.zeros_like(W)
    diag_len = min(M, N)
    W_reconstructed[range(diag_len), range(diag_len)] = 1.0  # Start with identity

    # Apply corrections in reverse order
    for corr in reversed(corrections):
        # Reconstruct delta from low-rank factors
        delta_approx = corr['U'] @ torch.diag(corr['S']) @ corr['Vh']
        # Add to identity at the specified indices
        W_reconstructed[corr['indices']] = I[corr['indices']] + delta_approx

    # Test output
    with torch.no_grad():
        approx_output = input_activations @ W_reconstructed.T

    sqnr = compute_sqnr(output_activations, approx_output)

    # Compute parameter count (low-rank storage)
    params_stored = sum(
        len(corr['indices']) +  # indices
        corr['U'].numel() +      # k × rank
        corr['S'].numel() +      # rank
        corr['Vh'].numel()       # rank × N
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
