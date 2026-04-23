"""
Quantize models with GPTQ + nvfp4a16 with and without scale rounding

This script creates two versions of the quantized model:
1. Without scale rounding (round_scales=False) - baseline
2. With scale rounding (round_scales=True) - improved

Both models are saved and can be evaluated using lm_eval.
"""

import sys
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from llmcompressor import oneshot

# Model configuration - can be overridden via command line
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Dataset configuration
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512  # Increase for better calibration
MAX_SEQUENCE_LENGTH = 2048

print("=" * 80)
print("GPTQ + nvfp4a16 Quantization (Both Versions)")
print("=" * 80)
print(f"Model: {MODEL_ID}")
print(f"Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print("=" * 80)


def load_and_prepare_dataset(tokenizer):
    """Load and prepare the calibration dataset."""
    print(f"\nLoading {NUM_CALIBRATION_SAMPLES} samples from {DATASET_ID}...")
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
    return ds


def quantize_and_save(model_id, dataset, round_scales, output_suffix):
    """Quantize a model using GPTQ with nvfp4a16 and save it."""
    print(f"\n{'=' * 80}")
    print(f"Quantizing with round_scales={round_scales}")
    print(f"{'=' * 80}")

    # Load fresh model
    print(f"Loading model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

    # Configure GPTQ with nvfp4a16 quantization
    # Uses tensor_group strategy for proper nvfp4 with global_scale fusion
    recipe = f"""
quant_stage:
    quant_modifiers:
        GPTQModifier:
            ignore: ["lm_head"]
            round_scales: {str(round_scales).lower()}
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: false
                        symmetric: true
                        group_size: 128
                        scale_dtype: float8_e4m3fn
                        zp_dtype: float8_e4m3fn
                    targets: ["Linear"]
"""

    print("Applying GPTQ quantization...")
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save the quantized model
    save_dir = f"{model_id.split('/')[-1]}-GPTQ-NVFP4A16-{output_suffix}"
    print(f"\nSaving model to {save_dir}...")
    model.save_pretrained(save_dir, save_compressed=True)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return save_dir


def main():
    # Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained("tokenizer_cache")

    # Prepare calibration dataset
    calib_dataset = load_and_prepare_dataset(tokenizer)

    # Quantize both versions
    print("\n" + "=" * 80)
    print("STEP 1: Quantize WITHOUT scale rounding (baseline)")
    print("=" * 80)
    baseline_dir = quantize_and_save(
        MODEL_ID,
        calib_dataset,
        round_scales=False,
        output_suffix="NoRounding"
    )

    # Save tokenizer to baseline
    tokenizer.save_pretrained(baseline_dir)

    print("\n" + "=" * 80)
    print("STEP 2: Quantize WITH scale rounding (improved)")
    print("=" * 80)
    improved_dir = quantize_and_save(
        MODEL_ID,
        calib_dataset,
        round_scales=True,
        output_suffix="WithRounding"
    )

    # Save tokenizer to improved
    tokenizer.save_pretrained(improved_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE")
    print("=" * 80)
    print(f"Baseline (no rounding):  {baseline_dir}")
    print(f"Improved (with rounding): {improved_dir}")
    print("=" * 80)
    print("\nUse lm_eval to evaluate both models and compare results.")


if __name__ == "__main__":
    main()
