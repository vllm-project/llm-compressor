"""
HIGGS: ILP-Based Mixed-Precision Quantization Example

This example demonstrates how to use HIGGS to automatically select optimal
quantization schemes for each layer in Llama-3-8B.
"""

import os
import time
import torch

from llmcompressor.transformers.compression.higgs import ilp_quantize

# Select model
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Define candidate quantization schemes
# HIGGS will evaluate each scheme on each layer and select the optimal assignment
# Note: Use schemes with different quality/size tradeoffs for mixed-precision
CANDIDATE_SCHEMES = ["W4A16", "W8A16"]  # 4-bit vs 8-bit weights

# Run HIGGS quantization
# Phase 1: Collect MSE data and solve ILP
# Phase 2: Apply selected schemes and save
print("Starting HIGGS quantization...")

# Reset peak memory stats
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

start_time = time.time()

optimal_config = ilp_quantize(
    model_stub=MODEL_ID,
    save_directory=os.path.expanduser("~/hf_hub/Meta-Llama-3-8B-Instruct-HIGGS"),
    candidate_schemes=CANDIDATE_SCHEMES,
    targets="Linear",  # Quantize all Linear layers
    ignore=["lm_head", "model.embed_tokens"],  # Skip output projection and embeddings
    enforce_fused_layer_constraints=True,  # gate_proj+up_proj, q+k+v same scheme
    target_avg_bitwidth=6.0,  # Target 6-bit average
    max_workers=8,  # Parallel workers for phase 2
    device="cuda:0",  # Use GPU for faster processing
)

end_time = time.time()
elapsed_time = end_time - start_time

# Get peak memory
if torch.cuda.is_available():
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak CUDA memory: {peak_memory_gb:.2f} GB")

# Print summary
print("\n" + "=" * 80)
print("HIGGS Quantization Complete!")
print("=" * 80)
print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"Generated {len(optimal_config.config_groups)} config groups:")
for group_name, scheme in optimal_config.config_groups.items():
    num_layers = len(scheme.targets)
    bitwidth = scheme.weights.num_bits if scheme.weights else "N/A"
    print(f"  {group_name}: {num_layers} layers @ {bitwidth}-bit")
print(f"\nQuantized model saved to: ~/hf_hub/Meta-Llama-3-8B-Instruct-HIGGS")
print("=" * 80)

# Verify the quantized model was saved
print("\nVerifying quantized model files...")
import os
output_dir = os.path.expanduser("~/hf_hub/Meta-Llama-3-8B-Instruct-HIGGS")
if os.path.exists(output_dir):
    files = os.listdir(output_dir)
    safetensor_files = [f for f in files if f.endswith('.safetensors')]
    print(f"  Found {len(safetensor_files)} safetensors files")
    print(f"  Total files: {len(files)}")

    # Check config
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
        if "quantization_config" in config:
            print(f"  Quantization config present: {config['quantization_config']['format']}")
print("=" * 80)

"""
lm_eval --model vllm \
    --model_args "pretrained=/home/HDCharles/repos/llm-compressor/Meta-Llama-3-8B-Instruct-HIGGS,dtype=auto,max_model_len=4096,add_bos_token=True,gpu_memory_utilization=0.85" \
    --tasks "wikitext" \
    --batch_size auto
"""