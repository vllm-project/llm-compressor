"""
HIGGS Mixed-Precision: NVFP4 + FP8_DYNAMIC via oneshot

1. get_higgs_config(): model-free ILP selects optimal per-layer schemes
2. oneshot() + QuantizationModifier: loads model, applies config with calibration

NVFP4 (W4A4) requires calibration data for activation global scales.

Usage:
    python higgs_nvfp4_fp8_oneshot.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --target-bits 6.0
"""

import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.entrypoints.higgs import get_higgs_config

IGNORE = [
    "lm_head",
    "re:.*embed_tokens",
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*multi_modal_projector.*",
]

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-bits", type=float, default=6.0)
    args = parser.parse_args()

    schemes = ["NVFP4", "FP8_DYNAMIC"]
    model_short = args.model.rstrip("/").split("/")[-1]
    tag = "+".join(sorted(schemes))
    save_dir = os.path.expanduser(
        f"~/hf_hub/{model_short}-HIGGS-{tag}-W{args.target_bits}avg"
    )

    # Step 1: get optimal mixed-precision config (model-free, no GPU model load)
    config = get_higgs_config(
        model_stub=args.model,
        candidate_schemes=schemes,
        targets="Linear",
        ignore=IGNORE,
        target_avg_bitwidth=args.target_bits,
    )

    print(f"\nHIGGS config: {len(config.config_groups)} groups")
    for name, scheme in config.config_groups.items():
        print(f"  {name}: {len(scheme.targets)} layers")

    # Step 2: load model, prepare calibration data, apply via oneshot
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
    )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False
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

    recipe = QuantizationModifier(
        config_groups=config.config_groups,
        ignore=config.ignore,
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        output_dir=save_dir,
    )

    tokenizer.save_pretrained(save_dir)
    print(f"\nSaved to: {save_dir}")


if __name__ == "__main__":
    main()
