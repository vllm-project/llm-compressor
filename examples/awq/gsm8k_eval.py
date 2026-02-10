"""
GSM8K evaluation script for AWQ+FP8 quantized models.

Usage:
    python gsm8k_eval.py <model_path>

Example:
    python gsm8k_eval.py ./Qwen2.5-0.5B-Instruct-awq-fp8-dynamic
"""

import argparse
import os
import subprocess
import sys


def evaluate_model(model_path):
    """Run GSM8K eval using lm-eval."""
    print(f"\nEvaluating {model_path} on GSM8K...")

    # Output dir based on model path
    output_dir = os.path.basename(model_path.rstrip("/")) + "_gsm8k_results"

    # Run lm-eval with batch_size=16
    # Note: Don't use batch_size=auto, it defaults to 1 which is super slow
    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model_path},dtype=auto",
        "--tasks",
        "gsm8k",
        "--batch_size",
        "16",
        "--output_path",
        output_dir,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\nResults saved to {output_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval quantized models on GSM8K")
    parser.add_argument("model_path", help="Path to quantized model directory")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        print(f"Error: Model path not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)

if not os.path.isdir(args.model_path):
    print(f"Error: Model path not found: {args.model_path}", file=sys.stderr)
    sys.exit(1)

evaluate_model(args.model_path)
