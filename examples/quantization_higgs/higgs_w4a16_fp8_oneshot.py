"""
HIGGS Mixed-Precision: W4A16 + FP8_DYNAMIC via oneshot

1. get_higgs_config(): model-free ILP selects optimal per-layer schemes
2. oneshot(): loads the model and applies either QuantizationModifier or
   GPTQModifier to the selected allocation

GPTQ uses calibration activations to build Hessian matrices for weight
optimization. QuantizationModifier applies the same allocation directly.

Usage:
    python higgs_w4a16_fp8_oneshot.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --target-bits 6.0 \
        --method qmod
"""

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
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
    parser.add_argument(
        "--method",
        choices=("qmod", "gptq"),
        default="qmod",
        help="oneshot quantization method (default: qmod)",
    )
    args = parser.parse_args()

    schemes = ["W4A16", "FP8_DYNAMIC"]
    method_name = "GPTQ" if args.method == "gptq" else "QMod"
    model_short = args.model.rstrip("/").split("/")[-1]
    scheme_tag = "W4A16+FP8_DYNAMIC"
    save_dir = os.path.expanduser(
        f"~/hf_hub/{model_short}-HIGGS-{scheme_tag}-W{args.target_bits}avg-{method_name}"
    )

    config = get_higgs_config(
        model_stub=args.model,
        candidate_schemes=schemes,
        targets="Linear",
        ignore=IGNORE,
        target_avg_bitwidth=args.target_bits,
        allow_unquantized=False,
    )

    print(f"\nHIGGS config: {len(config.config_groups)} groups")
    for name, scheme in config.config_groups.items():
        print(f"  {name}: {len(scheme.targets)} layers")

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.method == "gptq":
        from llmcompressor.modifiers.gptq import GPTQModifier

        recipe = GPTQModifier(
            config_groups=config.config_groups,
            ignore=config.ignore,
        )
    else:
        from llmcompressor.modifiers.quantization import QuantizationModifier

        recipe = QuantizationModifier(
            config_groups=config.config_groups,
            ignore=config.ignore,
        )

    oneshot(
        model=model,
        dataset="perfectblend",
        splits=f"train[:{NUM_CALIBRATION_SAMPLES}]",
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        output_dir=save_dir,
    )

    tokenizer.save_pretrained(save_dir)
    print(f"\nSaved to: {save_dir}")


if __name__ == "__main__":
    main()
