"""
RTN weight-only quantization to int2-7 using pack-quantized format.

Usage:
    python weight_only.py --scheme W7A16
    python weight_only.py --scheme W3A16 --model_id meta-llama/Meta-Llama-3-8B-Instruct
    python weight_only.py --scheme W4A16 --asymmetric
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
parser.add_argument("--scheme", type=str, required=True)
parser.add_argument("--asymmetric", action="store_true")
parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

suffix = "asym" if args.asymmetric else ""
SAVE_DIR = args.model_id.rstrip("/").split("/")[-1] + f"-{args.scheme}{suffix}-RTN"

if os.path.exists(SAVE_DIR):
    print(f"Output already exists at {SAVE_DIR!r}, skipping quantization.")
    exit(0)

model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_id)

if args.asymmetric:
    num_bits = int(args.scheme.split("A")[0].replace("W", ""))
    recipe = QuantizationModifier(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=num_bits,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.CHANNEL,
                    symmetric=False,
                ),
            )
        },
        ignore=["lm_head"],
    )
else:
    recipe = QuantizationModifier(
        targets=["Linear"],
        scheme=args.scheme,
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
