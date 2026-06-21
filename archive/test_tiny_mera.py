"""Test MERA on tiny synthetic data first."""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/brian-dellabetta/projects/llm-compressor')
from binary_mera_correct import BinaryMERACorrect, compute_snr_db

device = torch.device('cpu')  # Use CPU for easier debugging
print("Testing on tiny synthetic data...")

# Tiny problem: batch=2, seq=8, hidden=16
batch_size = 2
seq_len = 8
hidden_dim = 16
chi = 16

# Synthetic data
data = torch.randn(batch_size, seq_len, hidden_dim)
print(f"Data shape: {data.shape}")

# Create MERA
mera = BinaryMERACorrect(seq_len, hidden_dim, chi, 0.999).to(device)
mera.initialize_uv_gate(data)

print("\nLayer 0 (UV gate)...")
X = data @ mera.U_features
recon = X @ mera.Vt_features
snr = compute_snr_db(data[0], recon[0])
print(f"SNR: {snr:.2f} dB")

print("\nAdding Layer 0...")
mera.add_layer(device)

print("Forward pass...")
mera.eval()
with torch.no_grad():
    latent, intermediates = mera(data, stop_layer=1, adaptive=False)
    print(f"Latent shape: {latent.shape}")
    print(f"Intermediates: {[x.shape for x in intermediates]}")

    print("\nReconstruction...")
    recon = mera.reconstruct(latent, intermediates)
    print(f"Recon shape: {recon.shape}")

    snr = compute_snr_db(data[0], recon[0])
    print(f"SNR: {snr:.2f} dB")

print("\nSuccess!")
