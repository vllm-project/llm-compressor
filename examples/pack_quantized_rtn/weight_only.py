"""
RTN weight-only quantization to int3 / int5 / int7 using pack-quantized format.

Usage:
    python weight_only.py --num_bits 7
    python weight_only.py --num_bits 5 --model_id meta-llama/Meta-Llama-3-8B-Instruct
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

parser = argparse.ArgumentParser()
parser.add_argument("--num_bits", type=int, required=True, choices=[2, 3, 5, 7])
parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

SAVE_DIR = args.model_id.rstrip("/").split("/")[-1] + f"-W{args.num_bits}A16-RTN"

if os.path.exists(SAVE_DIR):
    print(f"Output already exists at {SAVE_DIR!r}, skipping quantization.")
    exit(0)

model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

recipe = QuantizationModifier(
    config_groups={
        "group_0": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=args.num_bits,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=True,
            ),
        )
    },
    ignore=["lm_head"],
)

oneshot(model=model, recipe=recipe)

print("\n\n========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

model.save_pretrained(SAVE_DIR, save_compressed=True, quantization_format="pack-quantized")
tokenizer.save_pretrained(SAVE_DIR)
print(f"Saved to {SAVE_DIR}")
