"""
LongCat-2.0 NVFP4A16 Quantization Example

This example demonstrates how to quantize LongCat-2.0 (1.6T total params, 48B active)
to NVFP4A16 using the model_free_ptq flow with multi-GPU acceleration.

LongCat-2.0 is a Mixture-of-Experts model with:
- 38 transformer layers
- 768 routed experts (top-12 routing)
- Multi-head Latent Attention (MLA) mechanism
- 262K max context length

Usage:
   ```bash
   python longcat_2_nvfp4a16.py
   ```

Note: Reindexing is no longer required - model_free_ptq now handles fused weights directly.
"""

import os
import torch

# Disable torch.compile to avoid dynamo recompilation issues with FP4 casting
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from llmcompressor import model_free_ptq

# Memory monitoring function
def log_gpu_memory(stage=""):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"[{stage}] GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    torch.cuda.empty_cache()

# Source model - can be HuggingFace repo or local path
MODEL_ID = os.environ.get("MODEL_ID", "meituan-longcat/LongCat-2.0")

# Save quantized model (use HF_HUB_DIR if set, otherwise HF_HOME, otherwise local)
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~"))
HF_HUB_DIR = os.environ.get("HF_HUB_DIR", HF_HOME)
SAVE_DIR = os.path.join(HF_HUB_DIR, "LongCat-2.0-NVFP4A16")

print(f"Quantizing model from: {MODEL_ID}")
print(f"Saving quantized model to: {SAVE_DIR}")

# Apply NVFP4A16 quantization using model_free_ptq
# This quantizes most linear layers to 4-bit floating point with 16-bit activations
# Multi-GPU: omitting device parameter will auto-detect and use all available GPUs
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="NVFP4A16",
    ignore=[
        # Ignore embedding and output layers (standard practice)
        "model.embed_tokens",
        "lm_head",

        # Ignore routing gates (critical for MoE routing decisions)
        "re:.*gate$",

        # Ignore normalization layers (1D tensors, incompatible with linear weight quantization)
        "re:.*norm.*",

        # Ignore MLA-specific layers that may be incompatible
        # These are part of the latent attention mechanism
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",

        # Multi-token prediction head (if present, may need fp16 precision)
        "re:.*mtp.*",
    ],
    max_workers=2,  # Adjust based on available CPU cores
    # device parameter omitted - will auto-detect and use all available GPUs
    # To use specific GPUs, pass: device=["cuda:0", "cuda:1", "cuda:2"]
)

log_gpu_memory("After quantization")

print(f"\nQuantized model saved to: {SAVE_DIR}")
print("The model can now be loaded with vLLM or other inference engines that support NVFP4.")
