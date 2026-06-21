"""Debug identity initialization."""

import torch
import numpy as np

# Test identity pass-through
chi = 128
seq_len = 4096

# Create fake data
X = torch.randn(1, seq_len, chi)

# Layer 0: u = identity [2*chi, 2*chi], w = [I; 0] [2*chi, chi]
u = torch.eye(2 * chi)
w = torch.zeros(2 * chi, chi)
w[:chi, :] = torch.eye(chi)

print("u shape:", u.shape)
print("w shape:", w.shape)
print("u orthog check (should be 0):", (u @ u.T - torch.eye(2*chi)).abs().max().item())
print("w orthog check (should be 0):", (w.T @ w - torch.eye(chi)).abs().max().item())

# Forward pass
n_pairs = seq_len // 2

# Step 1: Disentangle pairs
pair_indices = torch.arange(0, n_pairs * 2, 2)
pairs_left = X[:, pair_indices, :]
pairs_right = X[:, pair_indices + 1, :]
pairs = torch.cat([pairs_left, pairs_right], dim=-1)  # [1, 2048, 256]

print("\npairs shape:", pairs.shape)

# Apply disentangler (should be identity)
pairs_disentangled = pairs @ u.T
print("pairs_disentangled shape:", pairs_disentangled.shape)
print("disentangle changed data (should be 0):", (pairs - pairs_disentangled).abs().max().item())

X_disentangled = X.clone()
X_disentangled[:, pair_indices, :] = pairs_disentangled[:, :, :chi]
X_disentangled[:, pair_indices + 1, :] = pairs_disentangled[:, :, chi:]

print("X_disentangled changed data (should be 0):", (X - X_disentangled).abs().max().item())

# Step 2: Isometry
X_trim = X_disentangled[:, :n_pairs * 2, :]
pairs = X_trim.reshape(1, n_pairs, 2 * chi)  # [1, 2048, 256]

print("\nIsometry input pairs shape:", pairs.shape)

X_compressed = pairs @ w  # [1, 2048, 128]
print("X_compressed shape:", X_compressed.shape)

# Backward pass - WRONG: using w.T doesn't recover the lost information!
# The isometry is LOSSY - it projects 2*chi → chi
# We can only get back a projection: X_compressed @ w.T gives us [1, 2048, 256]
# but the second half (bottom chi dims) is all zeros

# What we DO have is pairs shape [1, 2048, 256] → w → [1, 2048, 128]
# The pseudo-inverse of w is w.T, so pairs ≈ X_compressed @ w.T
# But pairs originally had BOTH left and right, and we only kept one half

print("\nAttempting reconstruction (will fail - lossy operation!):")
X_expanded = X_compressed @ w.T  # [1, 2048, 256]
print("X_expanded shape:", X_expanded.shape)
print("X_expanded second half all zeros?", X_expanded[:, :, chi:].abs().max().item())

X_back = X_expanded.reshape(1, n_pairs * 2, chi)  # [1, 4096, 128]
print("X_back shape:", X_back.shape)

# Inverse disentangler
pairs_left = X_back[:, pair_indices, :]
pairs_right = X_back[:, pair_indices + 1, :]
pairs = torch.cat([pairs_left, pairs_right], dim=-1)

pairs_original = pairs @ u  # Apply u (not u.T since u is symmetric)
X_final = X_back.clone()
X_final[:, pair_indices, :] = pairs_original[:, :, :chi]
X_final[:, pair_indices + 1, :] = pairs_original[:, :, chi:]

print("\nReconstruction error:", (X - X_final).abs().max().item())
print("Relative error:", ((X - X_final).norm() / X.norm()).item())

# Compute SNR
signal_power = (X ** 2).mean()
noise_power = ((X - X_final) ** 2).mean()
snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
print(f"SNR: {snr.item():.2f} dB")
