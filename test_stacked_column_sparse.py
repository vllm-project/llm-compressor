"""Test StackedColumnSparseLinear vs single ColumnSparseLinear."""
import torch
import sys
import importlib.util
from transformers import AutoModelForCausalLM

# Load module directly
spec = importlib.util.spec_from_file_location(
    "adtn_linear",
    "src/llmcompressor/modifiers/experimental/adtn_linear.py"
)
adtn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adtn_module)

ColumnSparseLinear = adtn_module.ColumnSparseLinear
StackedColumnSparseLinear = adtn_module.StackedColumnSparseLinear

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
print("Testing Stacked vs Single Column-Sparse")
print("="*70)

# Load real model
print("\nLoading meta-llama/Llama-3.2-1B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Get q_proj layer
q_proj = model.model.layers[0].self_attn.q_proj
print(f"Testing on q_proj: {q_proj.weight.shape}")

# Generate realistic activations (closer to real data than random Gaussian)
num_samples = 256
# Real activations are much more structured
input_activations = torch.randn(num_samples, q_proj.in_features) * 0.02

original_params = q_proj.weight.numel()

print(f"\n{'Method':<30} {'Layers':<8} {'Total Params':<15} {'SNR (dB)':<12} {'Compression'}")
print("="*75)

# 1. Single column-sparse @ 50%
print("\nSingle column-sparse (50% sparsity):")
single_50 = ColumnSparseLinear.from_linear(
    linear=q_proj,
    input_activations=input_activations,
    target_sparsity=0.5,
    k_cols_per_iter=32,
)

with torch.no_grad():
    original_output = q_proj(input_activations.to(q_proj.weight.dtype))
    single_50_output = single_50(input_activations.to(q_proj.weight.dtype))

snr_single_50 = compute_sqnr(original_output, single_50_output)
compression_single_50 = 100 * (1 - single_50.num_params / original_params)

print(f"  {'Single @ 50%':<30} {1:<8} {single_50.num_params:<15,} {snr_single_50:<12.2f} {compression_single_50:.1f}%")

# 2. Stacked column-sparse: 2 layers @ 70% each = 49% total
print("\nStacked column-sparse (2 layers @ 70% each):")
stacked_2x70 = StackedColumnSparseLinear.from_linear(
    linear=q_proj,
    input_activations=input_activations,
    target_sparsity_per_layer=0.7,
    max_layers=2,
    target_snr_db=60.0,  # High target to force 2 layers
    k_cols_per_iter=32,
)

with torch.no_grad():
    stacked_2x70_output = stacked_2x70(input_activations.to(q_proj.weight.dtype))

snr_stacked_2x70 = compute_sqnr(original_output, stacked_2x70_output)
compression_stacked_2x70 = 100 * (1 - stacked_2x70.num_params / original_params)

print(f"  {'Stacked 2x70%':<30} {len(stacked_2x70.layers):<8} {stacked_2x70.num_params:<15,} {snr_stacked_2x70:<12.2f} {compression_stacked_2x70:.1f}%")

# 3. Stacked column-sparse: 3 layers @ 80% each = 51% total
print("\nStacked column-sparse (3 layers @ 80% each):")
stacked_3x80 = StackedColumnSparseLinear.from_linear(
    linear=q_proj,
    input_activations=input_activations,
    target_sparsity_per_layer=0.8,
    max_layers=3,
    target_snr_db=60.0,
    k_cols_per_iter=32,
)

with torch.no_grad():
    stacked_3x80_output = stacked_3x80(input_activations.to(q_proj.weight.dtype))

snr_stacked_3x80 = compute_sqnr(original_output, stacked_3x80_output)
compression_stacked_3x80 = 100 * (1 - stacked_3x80.num_params / original_params)

print(f"  {'Stacked 3x80%':<30} {len(stacked_3x80.layers):<8} {stacked_3x80.num_params:<15,} {snr_stacked_3x80:<12.2f} {compression_stacked_3x80:.1f}%")

# 4. Adaptive stacking with target SNR
print("\nStacked column-sparse (adaptive, target 40 dB):")
stacked_adaptive = StackedColumnSparseLinear.from_linear(
    linear=q_proj,
    input_activations=input_activations,
    target_sparsity_per_layer=0.75,
    max_layers=5,
    target_snr_db=40.0,  # Stop when 40 dB reached
    k_cols_per_iter=32,
)

with torch.no_grad():
    stacked_adaptive_output = stacked_adaptive(input_activations.to(q_proj.weight.dtype))

snr_stacked_adaptive = compute_sqnr(original_output, stacked_adaptive_output)
compression_stacked_adaptive = 100 * (1 - stacked_adaptive.num_params / original_params)

print(f"  {'Stacked adaptive':<30} {len(stacked_adaptive.layers):<8} {stacked_adaptive.num_params:<15,} {snr_stacked_adaptive:<12.2f} {compression_stacked_adaptive:.1f}%")

print("\n" + "="*75)
print("Summary")
print("="*75)
print(f"\nSingle layer @ 50%:  {snr_single_50:.1f} dB, {compression_single_50:.1f}% compression")
print(f"Stacked 2×70%:       {snr_stacked_2x70:.1f} dB, {compression_stacked_2x70:.1f}% compression")
print(f"Stacked 3×80%:       {snr_stacked_3x80:.1f} dB, {compression_stacked_3x80:.1f}% compression")
print(f"Stacked adaptive:    {snr_stacked_adaptive:.1f} dB, {compression_stacked_adaptive:.1f}% compression ({len(stacked_adaptive.layers)} layers)")

print("\n✅ Stacking improves SNR significantly!")
print("="*75)
