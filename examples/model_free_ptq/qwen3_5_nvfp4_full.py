"""
Qwen3.5-27B: Full NVFP4 quantization including DeltaNet linear_attn layers.

This example demonstrates quantizing ALL linear layers in Qwen3.5 to NVFP4,
including the DeltaNet linear attention projections (in_proj_qkv, in_proj_z,
out_proj) that are typically excluded.

Quantizing these layers reduces model size by ~30% and decode latency by ~35%
on bandwidth-limited hardware (e.g., DGX Spark with 273 GB/s LPDDR5x).

Usage:
    # Weight-only (fast, no calibration, uses Marlin W4A16 in vLLM):
    python qwen3_5_nvfp4_full.py --scheme NVFP4A16

    # Full W4A4 (slower, requires calibration, uses CUTLASS FP4 in vLLM):
    python qwen3_5_nvfp4_full.py --scheme NVFP4

Notes:
    - in_proj_a and in_proj_b (N=48) are excluded because the CUTLASS FP4
      tile requires N divisible by 64. These are tiny (<0.1% of params).
    - conv1d layers are excluded (3D tensors, not supported by NVFP4).
    - The full NVFP4 scheme requires the oneshot pipeline with calibration
      data for proper input_global_scale computation.
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scheme", default="NVFP4A16", choices=["NVFP4", "NVFP4A16"])
parser.add_argument("--model", default="Qwen/Qwen3.5-27B")
parser.add_argument("--num-samples", type=int, default=256)
parser.add_argument("--output", default=None)
args = parser.parse_args()

MODEL_ID = args.model
SAVE_DIR = args.output or f"{MODEL_ID.split('/')[-1]}-{args.scheme}-full"

# Common ignore list for Qwen3.5
IGNORE = [
    "re:.*lm_head",
    "re:visual.*",
    "re:model.visual.*",
    "re:.*mlp.gate$",
    "re:.*embed_tokens$",
    "re:.*shared_expert_gate$",
    "re:.*linear_attn.conv1d$",      # 3D conv, not supported
    "re:.*linear_attn.in_proj_a$",   # N=48, below CUTLASS FP4 tile minimum
    "re:.*linear_attn.in_proj_b$",   # N=48, below CUTLASS FP4 tile minimum
    "re:.*norm.*",                    # 1D norm weights
    "re:.*A_log$",                    # DeltaNet state params
    "re:.*dt_bias$",                  # DeltaNet state params
]

if args.scheme == "NVFP4A16":
    # Weight-only: calibration-free, fast
    from llmcompressor import model_free_ptq

    model_free_ptq(
        model_stub=MODEL_ID,
        save_directory=SAVE_DIR,
        scheme="NVFP4A16",
        ignore=IGNORE,
        device="cuda:0",
    )

else:
    # Full NVFP4 (W4A4): requires calibration for input_global_scale
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=IGNORE,
    )

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{args.num_samples}]")

    def preprocess(example):
        return {"text": processor.apply_chat_template(example["messages"], tokenize=False)}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return processor(text=sample["text"], padding=False, max_length=4096, truncation=True)

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    oneshot(
        model=model,
        recipe=recipe,
        dataset=ds,
        max_seq_length=4096,
        num_calibration_samples=args.num_samples,
    )

    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    processor.save_pretrained(SAVE_DIR)

    try:
        from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
        save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
    except Exception as e:
        print(f"Warning: Could not save MTP tensors: {e}")

print(f"Saved to {SAVE_DIR}")
