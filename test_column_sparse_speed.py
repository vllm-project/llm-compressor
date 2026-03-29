"""Test optimized ColumnSparseLinear speed."""
import torch
import time
import sys
import importlib.util

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
print("Testing Optimized ColumnSparseLinear Speed")
print("="*70)

# Test on different sizes
test_configs = [
    (512, 256, 100, "Small"),
    (2048, 2048, 100, "Medium (Llama q_proj)"),
    (4096, 4096, 100, "Large"),
]

for in_features, out_features, num_samples, label in test_configs:
    print(f"\n{label}: {in_features}×{out_features}")
    print("-" * 60)

    # Create test linear layer
    linear = torch.nn.Linear(in_features, out_features, bias=False)
    input_activations = torch.randn(num_samples, in_features) * 0.02

    # Time the compression
    start = time.time()
    column_sparse = ColumnSparseLinear.from_linear(
        linear=linear,
        input_activations=input_activations,
        target_sparsity=0.5,
        k_cols_per_iter=32,
    )
    elapsed = time.time() - start

    # Verify quality
    with torch.no_grad():
        original_output = linear(input_activations)
        sparse_output = column_sparse(input_activations)

    sqnr = compute_sqnr(original_output, sparse_output)
    num_selected = len(column_sparse.selected_columns)
    compression = 100 * (1 - column_sparse.num_params / linear.weight.numel())

    print(f"  Time: {elapsed:.2f}s")
    print(f"  Selected: {num_selected}/{in_features} columns ({100*num_selected/in_features:.1f}%)")
    print(f"  Compression: {compression:.1f}%")
    print(f"  SQNR: {sqnr:.2f} dB")

print("\n" + "="*70)
print("✅ Speed test complete!")
print("="*70)
