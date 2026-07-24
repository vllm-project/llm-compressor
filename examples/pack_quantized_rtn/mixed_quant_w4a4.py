"""
Mixed-quantization checkpoint: one W4A4 layer + W5-W8 for the rest.

Tests the WNA4Int code path (INT4 activation quant) alongside WNA16 and
WNA8Int, without destroying model quality by using low-bit weights
everywhere.

Layer 0 gets W4A4; the remaining layers cycle through:
    W5A16, W6A16, W7A16, W8A16, W5A8, W6A8, W7A8

Usage:
    python mixed_quant_w4a4.py
    python mixed_quant_w4a4.py --model_id Qwen/Qwen3-4B
"""

import argparse
import os

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_id",
    type=str,
    default="Qwen/Qwen3-4B",
)
args = parser.parse_args()

SAVE_DIR = (
    args.model_id.rstrip("/").split("/")[-1] + "-mixed-quant-RTN-w4a4"
)

if os.path.exists(SAVE_DIR):
    print(f"Output already exists at {SAVE_DIR!r}, skipping.")
    exit(0)

REMAINING_FORMATS = [
    # (label, weight_bits, act_bits_or_None)
    ("W5A16", 5, None),
    ("W6A16", 6, None),
    ("W7A16", 7, None),
    ("W8A16", 8, None),
    ("W5A8",  5, 8),
    ("W6A8",  6, 8),
    ("W7A8",  7, 8),
]

num_layers = AutoConfig.from_pretrained(args.model_id).num_hidden_layers

config_groups = {}
for i in range(num_layers):
    if i == 0:
        label, wbits, abits = "W4A4", 4, 4
    else:
        label, wbits, abits = REMAINING_FORMATS[(i - 1) % len(REMAINING_FORMATS)]

    weights = QuantizationArgs(
        num_bits=wbits,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
    )

    input_activations = None
    if abits is not None:
        input_activations = QuantizationArgs(
            num_bits=abits,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            dynamic=True,
            symmetric=True,
        )

    config_groups[f"layer_{i}_{label}"] = QuantizationScheme(
        targets=[f"re:model\\.layers\\.{i}\\..*_proj$"],
        weights=weights,
        input_activations=input_activations,
    )

    print(f"  layer {i:2d} -> {label}")

recipe = QuantizationModifier(
    config_groups=config_groups,
    ignore=["lm_head"],
)

model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

oneshot(model=model, recipe=recipe)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer(
    "Hello my name is", return_tensors="pt"
).input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

model.save_pretrained(
    SAVE_DIR,
    save_compressed=True,
    quantization_format="pack-quantized",
)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved to {SAVE_DIR}")
