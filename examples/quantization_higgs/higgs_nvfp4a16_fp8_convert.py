"""
HIGGS Mixed-Precision: NVFP4A16 + FP8_DYNAMIC via convert_checkpoint

1. get_higgs_config(): model-free ILP selects optimal per-layer schemes
2. convert_checkpoint() + HiggsQuantizationConverter: applies quantization
   directly to safetensors without loading the full model into GPU memory

NVFP4A16 (W4A16) and FP8_DYNAMIC are data-free schemes, so no calibration
dataset is needed. This path is faster and uses less memory.

Usage:
    python higgs_nvfp4a16_fp8_convert.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --target-bits 6.0
"""

import argparse
import os

from compressed_tensors.entrypoints.convert import convert_checkpoint

from llmcompressor.transformers.compression.higgs import (
    HiggsQuantizationConverter,
    get_higgs_config,
)

IGNORE = [
    "lm_head",
    "re:.*embed_tokens",
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*multi_modal_projector.*",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-bits", type=float, default=6.0)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    schemes = ["NVFP4A16", "FP8_DYNAMIC"]
    model_short = args.model.rstrip("/").split("/")[-1]
    tag = "+".join(sorted(schemes))
    save_dir = os.path.expanduser(
        f"~/hf_hub/{model_short}-HIGGS-{tag}-W{args.target_bits}avg-convert"
    )

    # Step 1: get optimal mixed-precision config (model-free)
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

    # Step 2: apply quantization via convert_checkpoint (no model load)
    converter = HiggsQuantizationConverter(
        optimal_config=config,
        targets="Linear",
        ignore=IGNORE,
        device="cuda:0",
    )

    convert_checkpoint(
        model_stub=args.model,
        save_directory=save_dir,
        converter=converter,
        max_workers=args.max_workers,
    )

    print(f"\nSaved to: {save_dir}")


if __name__ == "__main__":
    main()
