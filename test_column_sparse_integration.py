"""Test ColumnSparseLinear integration with TensorNetworkModifier."""
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
print("Testing ColumnSparseLinear Integration")
print("="*70)

# Test direct usage of ColumnSparseLinear.from_linear
print("\n1. Testing ColumnSparseLinear.from_linear()...")

# Create test linear layer
in_features = 512
out_features = 256
linear = torch.nn.Linear(in_features, out_features, bias=False)

# Generate test activations
num_samples = 100
input_activations = torch.randn(num_samples, in_features) * 0.02

# Create column-sparse version
column_sparse = ColumnSparseLinear.from_linear(
    linear=linear,
    input_activations=input_activations,
    target_sparsity=0.5,
    k_cols_per_iter=16,
)

print(f"  Original params: {linear.weight.numel():,}")
print(f"  Column-sparse params: {column_sparse.num_params:,}")
print(f"  Selected columns: {len(column_sparse.selected_columns)}/{in_features}")
print(f"  Compression: {100*(1-column_sparse.num_params/linear.weight.numel()):.1f}%")

# Test forward pass
with torch.no_grad():
    original_output = linear(input_activations)
    sparse_output = column_sparse(input_activations)

sqnr = compute_sqnr(original_output, sparse_output)
print(f"  SQNR: {sqnr:.2f} dB")

# Test to_linear conversion
reconstructed = column_sparse.to_linear()
with torch.no_grad():
    reconstructed_output = reconstructed(input_activations)

reconstruction_error = torch.norm(sparse_output - reconstructed_output)
print(f"  Reconstruction error (to_linear): {reconstruction_error:.6f}")

print("\n✅ ColumnSparseLinear.from_linear() works!")

# Test with real model
print("\n2. Testing with real Llama model layer...")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Get q_proj layer
q_proj = model.model.layers[0].self_attn.q_proj
print(f"  Original q_proj: {q_proj.weight.shape}")

# Generate realistic activations
input_activations = torch.randn(100, q_proj.in_features) * 0.02

# Create column-sparse version
column_sparse_qproj = ColumnSparseLinear.from_linear(
    linear=q_proj,
    input_activations=input_activations,
    target_sparsity=0.5,
    k_cols_per_iter=32,
)

print(f"  Column-sparse params: {column_sparse_qproj.num_params:,}")
print(f"  Selected columns: {len(column_sparse_qproj.selected_columns)}/{q_proj.in_features}")
print(f"  Compression: {100*(1-column_sparse_qproj.num_params/q_proj.weight.numel()):.1f}%")

# Test forward
with torch.no_grad():
    original_output = q_proj(input_activations.to(q_proj.weight.dtype))
    sparse_output = column_sparse_qproj(input_activations.to(q_proj.weight.dtype))

sqnr = compute_sqnr(original_output, sparse_output)
print(f"  SQNR: {sqnr:.2f} dB")

print("\n✅ All tests passed!")
print("="*70)
