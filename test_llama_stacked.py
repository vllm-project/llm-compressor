"""Test stacked low-rank on real Llama weights."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import sys
import importlib.util

# Load modules
spec = importlib.util.spec_from_file_location(
    "adtn_linear",
    "src/llmcompressor/modifiers/experimental/adtn_linear.py"
)
adtn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adtn_module)

StackedLowRankLinear = adtn_module.StackedLowRankLinear
LowRankLayer = adtn_module.LowRankLayer

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
print("Testing Stacked Low-Rank on Real Llama Weights")
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

# Test with rank that gives ~0.6x params per layer
rank = 614  # From previous test, gives 0.60x per layer

print(f"\nConfiguration:")
print(f"  in_features: {in_features}")
print(f"  out_features: {out_features}")
print(f"  rank: {rank}")
print(f"  rank per layer params: {rank * (in_features + out_features):,} (0.60x)")

# Create stacked low-rank model
stacked = StackedLowRankLinear(
    in_features=in_features,
    out_features=out_features,
    layers=[],
)

print(f"\nAdding low-rank layers iteratively...")

max_layers = 5
target_sqnr = 30.0
original_params = W.numel()

for layer_idx in range(max_layers):
    # Compute residual
    with torch.no_grad():
        if layer_idx == 0:
            residual = output_activations.clone()
        else:
            current_approx = stacked(input_activations)
            residual = output_activations - current_approx

    # Fit low-rank to residual via SVD of OLS solution
    X = input_activations.float()
    Y = residual.float()

    # OLS solution
    W_full = torch.linalg.lstsq(X, Y).solution

    # SVD for low-rank
    U_svd, S, Vh = torch.linalg.svd(W_full, full_matrices=False)

    # Truncate to rank
    U_svd = U_svd[:, :rank]
    S_trunc = S[:rank]
    Vh_trunc = Vh[:rank, :]

    # Absorb singular values into V
    V_weighted = torch.diag(S_trunc) @ Vh_trunc

    # Create layer
    layer = LowRankLayer(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
    )
    layer.U.weight.data = U_svd.T
    layer.V.weight.data = V_weighted.T

    # Add to stack
    stacked.append_layer(layer)

    # Evaluate
    with torch.no_grad():
        approx = stacked(input_activations)

    sqnr = compute_sqnr(output_activations, approx)

    # Cosine similarity
    orig_flat = output_activations.reshape(-1)
    approx_flat = approx.reshape(-1)
    similarity = (orig_flat * approx_flat).sum() / (torch.norm(orig_flat) * torch.norm(approx_flat))

    # Parameter count
    stacked_params = stacked.num_params
    param_ratio = stacked_params / original_params

    print(f"  Layer {layer_idx}: SQNR = {sqnr:6.2f} dB, "
          f"Cos Sim = {similarity:.4f}, "
          f"Params = {param_ratio:.2f}x ({stacked_params:,} / {original_params:,})")

    if sqnr >= target_sqnr:
        print(f"\n✅ Target SQNR of {target_sqnr} dB achieved!")
        break

print(f"\nFinal Results:")
print(f"  Layers: {len(stacked.layers)}")
print(f"  SQNR: {sqnr:.2f} dB")
print(f"  Cosine Similarity: {similarity:.4f}")
print(f"  Parameters: {param_ratio:.2f}x original ({stacked_params:,} params)")

if sqnr >= 30 and param_ratio < 1.0:
    print(f"\n🎉 SUCCESS: Achieved {sqnr:.1f} dB SNR with {100*(1-param_ratio):.1f}% parameter reduction!")
elif sqnr >= 30:
    print(f"\n✓ Achieved {sqnr:.1f} dB SNR but used {param_ratio:.2f}x params")
else:
    print(f"\n⚠️  Only achieved {sqnr:.1f} dB SNR (target: 30+ dB)")

print("="*70)
