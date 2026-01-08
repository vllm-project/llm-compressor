"""
GPT-OSS Model Quantization Example

This script demonstrates quantizing GPT-OSS models using various quantization
algorithms: W4A8, AWQ, and GPTQ.

Usage:
    # Basic W4A8 quantization
    python gpt_oss_20b.py --algorithm w4a8

    # AWQ quantization
    python gpt_oss_20b.py --algorithm awq

    # GPTQ quantization
    python gpt_oss_20b.py --algorithm gptq

    # Custom options
    python gpt_oss_20b.py \
        --algorithm gptq \
        --model openai/gpt-oss-20b \
        --num-samples 512 \
        --max-seq-length 2048 \
        --output-dir my-quantized-model
"""

import argparse
from enum import Enum

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.gpt_oss import (
    convert_model_for_quantization_gptoss,
)
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.quantization import (
    GPTQModifier,
    QuantizationModifier,
)
from llmcompressor.utils import dispatch_for_generation


class QuantizationAlgorithm(str, Enum):
    """Supported quantization algorithms for GPT-OSS."""

    W4A8 = "w4a8"
    AWQ = "awq"
    GPTQ = "gptq"


def create_recipe(algorithm):
    """Create quantization recipe based on algorithm."""

    # Shared weights configuration for all algorithms
    weights_args = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.CHANNEL,
        symmetric=True,
        dynamic=False,
    )

    if algorithm == QuantizationAlgorithm.W4A8:
        # W4A8 is unique - includes 8-bit activation quantization
        activations_args = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=False,
            dynamic=True,
            observer=None,
        )

        scheme = QuantizationScheme(
            targets=["Linear"],
            weights=weights_args,
            input_activations=activations_args,
        )

        return QuantizationModifier(
            config_groups={"group_0": scheme},
            ignore=["lm_head"],
        )

    # AWQ and GPTQ share the same config_groups pattern
    config_groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": weights_args,
        }
    }

    if algorithm == QuantizationAlgorithm.AWQ:
        return AWQModifier(
            targets=["Linear"],
            ignore=["lm_head", "re:.*router$"],
            config_groups=config_groups,
        )

    elif algorithm == QuantizationAlgorithm.GPTQ:
        return GPTQModifier(
            targets=["Linear"],
            ignore=["lm_head", "re:.*router$"],
            config_groups=config_groups,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantize GPT-OSS models with various algorithms"
    )
    parser.add_argument(
        "--algorithm",
        type=QuantizationAlgorithm,
        choices=list(QuantizationAlgorithm),
        default=QuantizationAlgorithm.W4A8,
        help="Quantization algorithm to use (default: w4a8)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model ID from HuggingFace Hub (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {model_name}-{algorithm})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset ID (default: HuggingFaceH4/ultrachat_200k)",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train_sft",
        help="Dataset split to use (default: train_sft)",
    )
    parser.add_argument(
        "--no-calibrate-all-experts",
        action="store_true",
        help="Disable calibrate_all_experts mode (not recommended)",
    )
    parser.add_argument(
        "--skip-generation-test",
        action="store_true",
        help="Skip generation test after quantization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Use sensible defaults if not provided
    num_samples = args.num_samples
    max_seq_length = args.max_seq_length

    # Set output directory
    base_name = args.model.rstrip("/").split("/")[-1]
    output_dir = args.output_dir or f"{base_name}-{args.algorithm.value}"

    print("=" * 70)
    print(f"GPT-OSS {args.algorithm.value.upper()} Quantization")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Algorithm: {args.algorithm.value.upper()}")
    print(f"Calibration samples: {num_samples}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Output directory: {output_dir}")
    print(
        f"Calibrate all experts: {not args.no_calibrate_all_experts} (recommended)"
    )
    print("=" * 70)

    print(f"\n[1/6] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    print("Model loaded successfully")

    print("\n[2/6] Converting MoE experts for quantization...")
    print(
        "      This linearizes fused expert weights into separate projections"
    )
    convert_model_for_quantization_gptoss(
        model, calibrate_all_experts=not args.no_calibrate_all_experts
    )
    print("Conversion completed")

    print(f"\n[3/6] Loading calibration dataset: {args.dataset}")
    ds = load_dataset(
        args.dataset, split=f"{args.dataset_split}[:{num_samples}]"
    )
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    # Tokenize for GPTQ (required for GPTQ, optional for others)
    if args.algorithm == QuantizationAlgorithm.GPTQ:

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names)

    print(f"Loaded {len(ds)} calibration samples")

    algo_name = args.algorithm.value.upper()
    print(f"\n[4/6] Creating {algo_name} quantization recipe...")
    recipe = create_recipe(args.algorithm)
    print("Recipe created")

    print(f"\n[5/6] Running {algo_name} quantization...")
    print("      This will calibrate all experts for optimal quantization")
    if args.algorithm == QuantizationAlgorithm.GPTQ:
        print(
            "      GPTQ uses layer-wise reconstruction (this may take a while)"
        )
    elif args.algorithm == QuantizationAlgorithm.AWQ:
        print("      AWQ analyzes activation patterns for optimal scales")

    # GPTQ requires pre-tokenized dataset, so we pass None for tokenizer
    use_tokenizer = (
        None if args.algorithm == QuantizationAlgorithm.GPTQ else tokenizer
    )

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        tokenizer=use_tokenizer,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_samples,
        save_compressed=False,
        output_dir=output_dir,
    )
    print("Quantization completed")

    if not args.skip_generation_test:
        print("\n[6/6] Testing generation with quantized model...")
        dispatch_for_generation(model)
        test_prompt = "Hello, my name is"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.decode(output[0])
        print(f"      Prompt: {test_prompt}")
        print(f"      Generated: {generated_text}")
        print("Generation test passed")
    else:
        print("\n[6/6] Skipping generation test")

    print(f"\nSaving quantized model to: {output_dir}")
    print("Model saved successfully")

    # ---- Display vLLM Instructions ----
    print("\n" + "=" * 70)
    print("Quantization Complete!")
    print("=" * 70)
    print(f"Quantized model saved to: {output_dir}")
    print("\nTo run inference with vLLM:")
    print("-" * 70)
    print("from vllm import LLM, SamplingParams\n")
    print(f'model = LLM(model="{output_dir}", trust_remote_code=True)')
    print('prompts = ["Hello, my name is"]')
    print("sampling_params = SamplingParams(temperature=0.7, max_tokens=100)")
    print("outputs = model.generate(prompts, sampling_params)\n")
    print("for output in outputs:")
    print("    print(output.outputs[0].text)")
    print("=" * 70)


if __name__ == "__main__":
    main()
