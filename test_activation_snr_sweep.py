"""Test BlockTensorizedLinear targeting output activation SNR, not weight SNR.

Use realistic activations from calibration data to find configs that maintain
high activation SNR with parameter compression.
"""
import torch
import torch.nn as nn
import sys
import importlib.util
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Load tensorized_linear module first
spec_tn = importlib.util.spec_from_file_location(
    "tensorized_linear",
    "src/llmcompressor/modifiers/experimental/tensorized_linear.py"
)
tensorized_module = importlib.util.module_from_spec(spec_tn)
sys.modules['llmcompressor.modifiers.experimental.tensorized_linear'] = tensorized_module
spec_tn.loader.exec_module(tensorized_module)

# Load block_tensorized_linear module
spec = importlib.util.spec_from_file_location(
    "block_tensorized_linear",
    "src/llmcompressor/modifiers/experimental/block_tensorized_linear.py"
)
block_tn_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(block_tn_module)

BlockTensorizedLinear = block_tn_module.BlockTensorizedLinear

def compute_sqnr(original, approximation):
    """Compute SQNR in dB."""
    signal_power = torch.var(original)
    mse_noise = torch.mean((original - approximation) ** 2)
    sqnr_linear = signal_power / (mse_noise + 1e-10)
    sqnr_db = 10 * torch.log10(sqnr_linear)
    return sqnr_db.item()

def get_activation_samples(model, tokenizer, num_samples=32):
    """Get realistic activation samples from calibration data."""
    print(f"\nCollecting {num_samples} activation samples from calibration data...")

    # Load calibration dataset
    ds = load_dataset(
        "mit-han-lab/pile-val-backup",
        split=f"validation[:{num_samples*10}]",
    )

    def preprocess(example):
        return {"input_ids": tokenizer.encode(example["text"].strip()[:512])}  # Limit length

    ds = (
        ds.shuffle(seed=42)
        .map(preprocess, remove_columns=ds.column_names)
        .select(range(num_samples))
    )

    # Prepare batched inputs
    max_len = 512
    input_ids_list = []

    for item in ds:
        ids = item["input_ids"][:max_len]
        input_ids_list.append(torch.tensor(ids))

    # Pad to same length
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id or 0)

    return input_ids

def capture_layer_activations(model, layer_path, input_ids):
    """Capture input activations for a specific layer.

    Returns: input_activations (num_samples, seq_len, hidden_size)
    """
    activations = []

    # Get the layer
    parts = layer_path.split('.')
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)

    # Hook to capture input
    def hook(module, input, output):
        # input is a tuple, take first element
        activations.append(input[0].detach().cpu())

    handle = layer.register_forward_hook(hook)

    # Forward pass
    with torch.no_grad():
        model(input_ids.to(model.device))

    handle.remove()

    # Concatenate all activations: (batch*seq_len, hidden_size)
    all_acts = torch.cat(activations, dim=0)

    # Reshape to (num_tokens, hidden_size)
    if len(all_acts.shape) == 3:
        batch, seq, hidden = all_acts.shape
        all_acts = all_acts.reshape(batch * seq, hidden)

    return all_acts

print("="*120)
print("Activation SNR Test: Finding Configs that Preserve Output Quality with Compression")
print("="*120)

# Load model
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
print(f"\nLoading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Get calibration data
input_ids = get_activation_samples(model, tokenizer, num_samples=32)
print(f"Calibration data shape: {input_ids.shape}")

# Layers to test
layers_to_test = {
    'q_proj': 'model.layers.0.self_attn.q_proj',
    'k_proj': 'model.layers.0.self_attn.k_proj',
    'v_proj': 'model.layers.0.self_attn.v_proj',
    'o_proj': 'model.layers.0.self_attn.o_proj',
}

print(f"\n{'='*120}")
print("Testing Configurations")
print(f"{'='*120}")

# Test configurations: (x, num_cores, block_size, rank)
configs = [
    # Optimal structure: x=8, cores=3, block=512
    (8, 3, 512, 1.0),
    (8, 3, 512, 0.9),
    (8, 3, 512, 0.8),
    (8, 3, 512, 0.7),
    (8, 3, 512, 0.6),
    (8, 3, 512, 0.5),
    (8, 3, 512, 0.4),
    (8, 3, 512, 0.3),

    # Alternative: x=4, cores=3, block=64
    (4, 3, 64, 1.0),
    (4, 3, 64, 0.8),
    (4, 3, 64, 0.6),
    (4, 3, 64, 0.5),
    (4, 3, 64, 0.4),
    (4, 3, 64, 0.3),
]

all_results = []

for layer_name, layer_path in layers_to_test.items():
    print(f"\n{'-'*120}")
    print(f"Testing Layer: {layer_name} ({layer_path})")
    print(f"{'-'*120}")

    # Get the actual layer module
    parts = layer_path.split('.')
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)

    # Capture input activations for this layer
    print("Capturing input activations...")
    input_activations = capture_layer_activations(model, layer_path, input_ids)
    print(f"Input activations shape: {input_activations.shape}")

    # Get original output activations
    print("Computing original output activations...")
    with torch.no_grad():
        original_output = layer(input_activations.to(layer.weight.dtype))

    original_params = layer.weight.numel()

    print(f"\n{'x':<4} {'cores':<6} {'block':<8} {'rank':<6} {'Act SNR':<12} {'Weight SNR':<12} {'Params':<15} {'Ratio':<10} {'Compression':<12} {'Target?'}")
    print("-"*120)

    for x, num_cores, block_size, rank in configs:
        # Check if valid for this layer
        if layer.out_features % block_size != 0 or layer.in_features % block_size != 0:
            continue

        try:
            # Create BlockTensorizedLinear
            bt_layer = BlockTensorizedLinear.from_linear(
                layer,
                block_size=block_size,
                rank=rank,
                num_cores=num_cores,
                input_activations=input_activations.float(),  # Use real activations
            )

            # Compute compressed output activations
            with torch.no_grad():
                compressed_output = bt_layer(input_activations.to(layer.weight.dtype))

            # Compute activation SNR
            activation_snr = compute_sqnr(original_output, compressed_output)

            # Also compute weight SNR for comparison
            W_original = layer.weight.data.float()
            W_compressed = bt_layer.to_matrix().float()
            weight_snr = compute_sqnr(W_original, W_compressed)

            # Parameter stats
            compressed_params = bt_layer.num_params
            param_ratio = compressed_params / original_params
            compression_pct = 100 * (1 - param_ratio)

            # Check if meets target
            target_mark = "✅" if activation_snr >= 30.0 and param_ratio < 1.0 else "✓" if activation_snr >= 30.0 else ""

            result = {
                'layer': layer_name,
                'x': x,
                'num_cores': num_cores,
                'block_size': block_size,
                'rank': rank,
                'activation_snr': activation_snr,
                'weight_snr': weight_snr,
                'params': compressed_params,
                'param_ratio': param_ratio,
                'compression_pct': compression_pct,
            }
            all_results.append(result)

            print(f"{x:<4} {num_cores:<6} {block_size:<8} {rank:<6.2f} {activation_snr:>8.2f}    {weight_snr:>8.2f}    "
                  f"{compressed_params:>10,}    {param_ratio:>6.2f}x    {compression_pct:>6.1f}%    {target_mark}")

        except Exception as e:
            print(f"{x:<4} {num_cores:<6} {block_size:<8} {rank:<6.2f} ERROR: {str(e)[:50]}")

# Summary
print(f"\n{'='*120}")
print("SUMMARY: Configurations Achieving Activation SNR >= 30 dB")
print(f"{'='*120}\n")

target_results = [r for r in all_results if r['activation_snr'] >= 30.0]
target_results.sort(key=lambda x: x['param_ratio'])

if target_results:
    print(f"{'Layer':<10} {'Config':<30} {'Act SNR':<12} {'Weight SNR':<12} {'Ratio':<10} {'Compression':<12} {'Status'}")
    print("-"*110)

    for r in target_results:
        config = f"x={r['x']}, c={r['num_cores']}, bs={r['block_size']}, r={r['rank']:.1f}"
        status = "🎯 COMPRESSED!" if r['param_ratio'] < 1.0 else "No compression"
        print(f"{r['layer']:<10} {config:<30} {r['activation_snr']:>6.2f} dB   {r['weight_snr']:>6.2f} dB   "
              f"{r['param_ratio']:>6.2f}x    {r['compression_pct']:>6.1f}%    {status}")

    # Best compressed results
    compressed_results = [r for r in target_results if r['param_ratio'] < 1.0]
    if compressed_results:
        print(f"\n{'='*120}")
        print("🎉 SUCCESS: Configurations with Activation SNR >= 30 dB AND Compression")
        print(f"{'='*120}\n")

        for r in compressed_results[:10]:
            print(f"{r['layer']} with x={r['x']}, cores={r['num_cores']}, block={r['block_size']}, rank={r['rank']:.1f}:")
            print(f"  Activation SNR: {r['activation_snr']:.2f} dB")
            print(f"  Weight SNR: {r['weight_snr']:.2f} dB")
            print(f"  Compression: {r['compression_pct']:.1f}% ({r['param_ratio']:.2f}x params)")
            print()
else:
    print("⚠️  No configurations achieved activation SNR >= 30 dB")

    # Show best results
    all_results.sort(key=lambda x: -x['activation_snr'])
    print(f"\nBest Activation SNR Results:")
    print(f"{'Layer':<10} {'Config':<30} {'Act SNR':<12} {'Weight SNR':<12} {'Ratio':<10} {'Compression'}")
    print("-"*110)

    for r in all_results[:15]:
        config = f"x={r['x']}, c={r['num_cores']}, bs={r['block_size']}, r={r['rank']:.1f}"
        print(f"{r['layer']:<10} {config:<30} {r['activation_snr']:>6.2f} dB   {r['weight_snr']:>6.2f} dB   "
              f"{r['param_ratio']:>6.2f}x    {r['compression_pct']:>6.1f}%")

print(f"\n{'='*120}")
print("KEY INSIGHTS")
print(f"{'='*120}")
print("\n1. Activation SNR vs Weight SNR:")
print("   - Activation SNR measures output quality (what matters for model performance)")
print("   - Weight SNR measures weight similarity (less directly relevant)")
print("   - We can have low weight SNR but high activation SNR if structure is preserved")

print("\n2. Compression Trade-offs:")
print("   - Higher rank → Better activation SNR but less compression")
print("   - Lower rank → More compression but worse activation SNR")
print("   - Goal: Find minimum rank that maintains activation SNR >= 30 dB")

print(f"\n{'='*120}")
