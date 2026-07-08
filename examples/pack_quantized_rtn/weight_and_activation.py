"""
RTN weight + dynamic activation quantization using pack-quantized format.

Suggested pairs:
    W3A4  --weight_bits 3 --act_bits 4
    W5A8  --weight_bits 5 --act_bits 8
    W7A8  --weight_bits 7 --act_bits 8

Usage:
    python weight_and_activation.py --weight_bits 7 --act_bits 8
    python weight_and_activation.py --weight_bits 3 --act_bits 4 --model_id meta-llama/Meta-Llama-3-8B-Instruct
    python weight_and_activation.py --weight_bits 5 --act_bits 8 --asymmetric
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
parser.add_argument("--weight_bits", type=int, required=True, choices=[2, 3, 5, 6, 7])
parser.add_argument("--act_bits", type=int, required=True, choices=[4, 8])
parser.add_argument("--asymmetric", action="store_true")
parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
args = parser.parse_args()

suffix = "asym" if args.asymmetric else ""
SAVE_DIR = args.model_id.rstrip("/").split("/")[-1] + f"-W{args.weight_bits}A{args.act_bits}{suffix}-RTN"

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
                num_bits=args.weight_bits,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.CHANNEL,
                symmetric=not args.asymmetric,
            ),
            input_activations=QuantizationArgs(
                num_bits=args.act_bits,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.TOKEN,
                dynamic=True,
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
