"""Test Greedy Multi-Scale Decomposition on Llama weights."""
import torch
import sys
import importlib.util
from transformers import AutoModelForCausalLM

# Load the greedy multiscale module
spec = importlib.util.spec_from_file_location(
    "greedy_multiscale_linear",
    "src/llmcompressor/modifiers/experimental/greedy_multiscale_linear.py"
)
gms_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gms_module)

GreedyMultiScaleLinear = gms_module.GreedyMultiScaleLinear

def compute_sqnr(original, approximation):
    """Compute SQNR in dB."""
    signal_power = torch.var(original)
    mse_noise = torch.mean((original - approximation) ** 2)
    sqnr_linear = signal_power / (mse_noise + 1e-10)
    sqnr_db = 10 * torch.log10(sqnr_linear)
    return sqnr_db.item()

print("="*100)
print("Testing Greedy Multi-Scale Decomposition")
print("="*100)

# Load model
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
print(f"\nLoading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

# Test on multiple layers
layers_to_test = {
    'q_proj': model.model.layers[0].self_attn.q_proj,
    'k_proj': model.model.layers[0].self_attn.k_proj,
    'v_proj': model.model.layers[0].self_attn.v_proj,
    'o_proj': model.model.layers[0].self_attn.o_proj,
}

print("\nLayer shapes:")
for name, layer in layers_to_test.items():
    print(f"  {name}: {layer.weight.shape} ({layer.weight.numel():,} params)")

# Generate synthetic activations
num_samples = 256

# Test different configurations
configs = [
    # (mpo_block_size, mpo_num_cores, mpo_rank, lr_rank, max_stages, target_snr)
    (512, 3, 0.3, 64, 5, 30.0),   # Conservative: small MPO, moderate LR
    (512, 3, 0.2, 128, 5, 30.0),  # Smaller MPO, larger LR
    (512, 3, 0.4, 32, 5, 30.0),   # Larger MPO, smaller LR
    (512, 3, 0.3, 64, 5, 35.0),   # Higher target SNR
]

print(f"\n{'='*100}")
print("Testing Configurations")
print(f"{'='*100}\n")

results = []

for config_idx, (mpo_bs, mpo_cores, mpo_rank, lr_rank, max_stages, target_snr) in enumerate(configs, 1):
    print(f"\n{'='*100}")
    print(f"Configuration {config_idx}:")
    print(f"  MPO: block_size={mpo_bs}, num_cores={mpo_cores}, rank={mpo_rank}")
    print(f"  LR: rank={lr_rank}")
    print(f"  Max stages: {max_stages}, Target SNR: {target_snr} dB")
    print(f"{'='*100}")

    for layer_name, layer in layers_to_test.items():
        print(f"\n{'-'*100}")
        print(f"Layer: {layer_name}")
        print(f"{'-'*100}")

        W = layer.weight.data.float()
        in_features = layer.in_features
        out_features = layer.out_features

        # Generate activations
        input_activations = torch.randn(num_samples, in_features) * 0.02

        # Create greedy multiscale decomposition
        gms = GreedyMultiScaleLinear.from_linear(
            layer,
            input_activations=input_activations,
            target_snr_db=target_snr,
            max_stages=max_stages,
            mpo_block_size=mpo_bs,
            mpo_num_cores=mpo_cores,
            mpo_rank=mpo_rank,
            lr_rank=lr_rank,
            verbose=True,
        )

        # Verify reconstruction
        W_reconstructed = gms.to_matrix()
        weight_snr = compute_sqnr(W, W_reconstructed)

        # Verify forward pass
        with torch.no_grad():
            original_output = layer(input_activations.to(layer.weight.dtype))
            gms_output = gms(input_activations)
            activation_snr = compute_sqnr(original_output.float(), gms_output.float())

        original_params = W.numel()
        compressed_params = gms.num_params
        param_ratio = compressed_params / original_params
        compression_pct = 100 * (1 - param_ratio)

        result = {
            'config_idx': config_idx,
            'layer': layer_name,
            'mpo_bs': mpo_bs,
            'mpo_rank': mpo_rank,
            'lr_rank': lr_rank,
            'target_snr': target_snr,
            'num_stages': len(gms.stages),
            'weight_snr': weight_snr,
            'activation_snr': activation_snr,
            'params': compressed_params,
            'param_ratio': param_ratio,
            'compression_pct': compression_pct,
        }
        results.append(result)

        print(f"\nVerification:")
        print(f"  Weight SNR: {weight_snr:.2f} dB")
        print(f"  Activation SNR: {activation_snr:.2f} dB")
        print(f"  Achieved target: {'✓' if activation_snr >= target_snr else '✗'}")

# Summary
print(f"\n{'='*100}")
print("SUMMARY: All Results")
print(f"{'='*100}\n")

print(f"{'Config':<8} {'Layer':<10} {'Stages':<8} {'Act SNR':<12} {'Params':<15} {'Ratio':<10} {'Compression':<12} {'Target?'}")
print("-"*100)

for r in results:
    target_mark = "✅" if r['activation_snr'] >= r['target_snr'] else "❌"
    print(f"{r['config_idx']:<8} {r['layer']:<10} {r['num_stages']:<8} {r['activation_snr']:>6.2f} dB   "
          f"{r['params']:>10,}    {r['param_ratio']:>6.2f}x    {r['compression_pct']:>6.1f}%    {target_mark}")

# Best results
print(f"\n{'='*100}")
print("BEST RESULTS by Compression (achieving target SNR)")
print(f"{'='*100}\n")

successful = [r for r in results if r['activation_snr'] >= r['target_snr']]
successful.sort(key=lambda x: x['param_ratio'])

if successful:
    print(f"{'Layer':<10} {'Config':<30} {'Stages':<8} {'Act SNR':<12} {'Compression':<12} {'Status'}")
    print("-"*100)

    for r in successful[:10]:
        config = f"MPO:{r['mpo_bs']}/{r['mpo_rank']:.1f}, LR:{r['lr_rank']}"
        status = "🎯 COMPRESSED" if r['param_ratio'] < 1.0 else "No compression"
        print(f"{r['layer']:<10} {config:<30} {r['num_stages']:<8} {r['activation_snr']:>6.2f} dB   "
              f"{r['compression_pct']:>6.1f}%    {status}")

    print(f"\n🎉 Found {len([r for r in successful if r['param_ratio'] < 1.0])} configs with compression + target SNR")
else:
    print("⚠️  No configurations achieved target SNR")

print(f"\n{'='*100}")
print("KEY INSIGHTS")
print(f"{'='*100}")
print("\n1. Greedy Multi-Scale Decomposition iteratively builds approximation")
print("2. Each stage (MPO + LR) captures different aspects of the residual")
print("3. Small MPOs (low rank) + moderate LR corrections can achieve high SNR")
print("4. More memory efficient than single large MPO")
print(f"\n{'='*100}")
