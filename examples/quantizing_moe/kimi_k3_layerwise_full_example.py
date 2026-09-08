"""
Full layerwise decompression + compression example for Kimi-K3.

NOTE: This example requires the layerwise_decompression and layerwise_compression
flags to be available on your branch. These flags are currently being developed and
will be merged soon.

This example demonstrates the most explicit form of layerwise recalibration:
1. Load a pre-compressed Kimi-K3 model
2. For each layer sequentially:
   - Decompress the layer (strip all quantization metadata)
   - Re-apply quantization config & observers scoped to that layer
   - Calibrate with fresh data
   - Run error propagation
   - Recompress the layer
3. Save the recalibrated model

This approach keeps peak memory at just 1-2 layers, making it feasible for
100B+ parameter models.

When the flags are not yet available, use kimi_k3_layerwise_example.py which
demonstrates the sequential pipeline approach (available now).
"""

import torch
from compressed_tensors.quantization import QuantizationConfig
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "moonshotai/Kimi-K3"
num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPUs")

# Load quantization config from pretrained and add ignore patterns
qconfig = QuantizationConfig.from_pretrained(MODEL_ID)
qconfig.ignore += [
    "re:.*mlp_res_proj.*",
    "re:.*self_attention_res_proj.*",
    "re:.*routed_expert.*",
    "re:.*output_attn_res_proj.*",
]

# Load model with the modified quantization config
with load_context(KimiK3ForConditionalGeneration):
    model = KimiK3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=qconfig,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure quantization for layerwise approach
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        r"re:.*block_sparse_moe\.gate",
        "re:.*vision_tower.*",
    ],
)

# Apply quantization with full layerwise decompression/compression.
#
# When layerwise_decompression and layerwise_compression flags are available,
# this enables the most explicit per-layer recalibration workflow:
#
# For each KimiK3DecoderLayer:
#   1. Decompress: Strip quantization metadata (leave_decompressed=False)
#   2. Calibrate: Re-apply config, run observations, update quantization params
#   3. Propagate: Adjust next layer's inputs based on this layer's errors
#   4. Compress: Pack weights back to compressed format
#
# This is the most memory-efficient approach for large models because:
# - Only ~1-2 layers are ever decompressed in GPU memory
# - Other layers remain in compressed format or on disk
# - Explicit control over when decompression/compression happens
#
# Try this example when the flags become available:
try:
    oneshot(
        model=model,
        processor=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        # Use sequential pipeline with per-layer processing
        pipeline="sequential",
        sequential_targets=["KimiK3DecoderLayer"],
        batch_size=1,
        shuffle_calibration_samples=True,
        propagate_error=True,
        # Layerwise flags (when available):
        # These enable explicit decompress→calibrate→compress per layer
        layerwise_decompression=True,  # Decompress before each layer's calibration
        layerwise_compression=True,    # Recompress after calibration/propagation
    )
except TypeError as e:
    if "layerwise_decompression" in str(e) or "layerwise_compression" in str(e):
        print("\n⚠ Layerwise flags not yet available on this branch.")
        print("  These flags are currently being developed and will be merged soon.")
        print("\n  In the meantime, use kimi_k3_layerwise_example.py which demonstrates")
        print("  the sequential pipeline approach (available now).")
        print("\n  Error:", str(e))
        raise
    else:
        raise

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-Layerwise-Full"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✓ Model saved to {SAVE_DIR}")
print("  Full layerwise decompression recalibrated the model with explicit")
print("  per-layer decompress→calibrate→compress cycles, minimizing peak memory.")
