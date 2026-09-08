"""
Layerwise decompression and recalibration example for Kimi-K3.

This example demonstrates how to:
1. Load a pre-compressed Kimi-K3 model
2. Use the sequential pipeline to process one transformer layer at a time
3. Recalibrate with different quantization settings while keeping memory low

The sequential pipeline ensures only one layer is decompressed and in GPU memory
at any given time, making it feasible to recalibrate large models without OOM.

Future versions (when layerwise_decompression and layerwise_compression flags
are available) will add per-layer decompress→calibrate→compress cycles for even
more granular control and memory efficiency.
"""

from compressed_tensors.quantization import QuantizationConfig
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "moonshotai/Kimi-K3"

# Load quantization config from pretrained and add ignore patterns
# for modules that should not be quantized
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

# Configure quantization with sequential pipeline targets.
# Using the sequential pipeline means one transformer layer at a time
# is loaded into GPU memory during calibration, reducing peak memory usage.
#
# The sequential pipeline with KimiK3 processes one KimiK3DecoderLayer at a time,
# keeping only that layer decompressed in VRAM while others remain in compressed
# format on disk.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        r"re:.*block_sparse_moe\.gate",
        "re:.*vision_tower.*",
    ],
)

# Apply quantization using the sequential pipeline.
# This is memory-efficient for large models because:
# 1. Only one layer is loaded into GPU memory at a time
# 2. Other layers stay compressed on disk or in CPU memory
# 3. Intermediate activations are streamed between layers
#
# When layerwise_decompression and layerwise_compression flags become available,
# you can enable them as kwargs:
#   layerwise_decompression=True,
#   layerwise_compression=True,
# This will add per-layer decompress→calibrate→compress cycles for even more
# granular memory control.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # Use sequential pipeline to process one layer at a time
    pipeline="sequential",
    sequential_targets=["KimiK3DecoderLayer"],
    batch_size=1,
    shuffle_calibration_samples=True,
    # Error propagation adjusts next layer's inputs for better accuracy
    propagate_error=True,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-Layerwise"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✓ Model saved to {SAVE_DIR}")
print("  Sequential pipeline recalibrated the model layer-by-layer,")
print("  keeping memory usage low throughout the process.")
