"""Test low-rank approximation on real Llama weights."""
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
print("Testing Low-Rank on Real Llama Weights")
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

print(f"\nTesting different ranks...")
print(f"{'Rank':<10} {'Params':<15} {'SQNR (dB)':<12} {'Compression'}")
print("="*60)

original_params = W.numel()

# Test different ranks
for rank_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    rank = int(rank_ratio * min(in_features, out_features))

    # SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vh_r = Vh[:rank, :]

    # Reconstruct
    W_approx = U_r @ torch.diag(S_r) @ Vh_r

    # Test
    with torch.no_grad():
        approx_output = input_activations @ W_approx.T

    sqnr = compute_sqnr(output_activations, approx_output)

    # Parameters: rank * (M + N)
    params = rank * (out_features + in_features)
    param_ratio = params / original_params

    print(f"{rank:<10} {params:<15,} {sqnr:<12.2f} {param_ratio:.2f}x")

print("="*70)
