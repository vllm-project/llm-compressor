"""Test different isometry initializations for 2→1 compression."""

import torch
import numpy as np

chi = 128

# Original data: two vectors
x1 = torch.randn(chi)
x2 = torch.randn(chi)
pair = torch.cat([x1, x2])  # [256]

print("Testing different isometry initializations for 2→1 compression:\n")

# Option 1: [I; 0] - keep only first half
w1 = torch.zeros(2 * chi, chi)
w1[:chi, :] = torch.eye(chi)

y1 = pair @ w1  # [128]
pair_recon1 = y1 @ w1.T  # [256]
error1 = (pair - pair_recon1).norm() / pair.norm()
print(f"1. w = [I; 0] (keep x1, drop x2)")
print(f"   Reconstruction error: {error1.item():.4f}")
print(f"   SNR: {-20*np.log10(error1.item()):.2f} dB\n")

# Option 2: [I/√2; I/√2] - average of x1 and x2
w2 = torch.zeros(2 * chi, chi)
w2[:chi, :] = torch.eye(chi) / np.sqrt(2)
w2[chi:, :] = torch.eye(chi) / np.sqrt(2)

y2 = pair @ w2  # [128]
pair_recon2 = y2 @ w2.T  # [256]
error2 = (pair - pair_recon2).norm() / pair.norm()
print(f"2. w = [I/√2; I/√2] (average x1 and x2)")
print(f"   Reconstruction error: {error2.item():.4f}")
print(f"   SNR: {-20*np.log10(error2.item()):.2f} dB")
print(f"   w.T @ w = I? {(w2.T @ w2 - torch.eye(chi)).abs().max().item():.2e}\n")

# Option 3: Random orthogonal
U, _, Vt = torch.linalg.svd(torch.randn(2 * chi, chi))
w3 = U @ Vt

y3 = pair @ w3
pair_recon3 = y3 @ w3.T
error3 = (pair - pair_recon3).norm() / pair.norm()
print(f"3. w = random orthogonal (SVD)")
print(f"   Reconstruction error: {error3.item():.4f}")
print(f"   SNR: {-20*np.log10(error3.item()):.2f} dB\n")

print("Key insight:")
print("- Any isometry that compresses 2→1 is LOSSY by definition")
print("- w @ w.T is a projection, not identity")
print("- Best we can do is averaging: w = [I/√2; I/√2]")
print("- This gives ~3 dB loss (50% error) at epoch 0, which is expected")
