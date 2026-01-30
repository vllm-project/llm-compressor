#!/usr/bin/env python3
"""
Quantize a model for lm-eval testing.

This script reads a config file (same format as test_lmeval.py),
quantizes the model, and saves it to disk WITHOUT cleanup.
The quantized model can then be evaluated separately using eval_with_vllm.py.

Usage:
    python quantize_for_lmeval.py --config tests/lmeval/configs/w4a16_awq_sym.yaml
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy
import torch
import yaml
from loguru import logger

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llmcompressor.core import active_session
from tests.e2e.e2e_utils import run_oneshot_for_e2e_testing


def quantize_model(config_path: str) -> str:
    """
    Quantize a model based on config file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Path to saved quantized model
    """
    # Load config
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    # Extract config values
    model = config["model"]
    model_class = config.get("model_class", "AutoModelForCausalLM")
    scheme = config.get("scheme")
    dataset_id = config.get("dataset_id")
    dataset_config = config.get("dataset_config")
    dataset_split = config.get("dataset_split")
    recipe = config.get("recipe")
    quant_type = config.get("quant_type")
    save_dir = config.get("save_dir")
    seed = config.get("seed", None)
    num_calibration_samples = config.get("num_calibration_samples", 512)
    max_seq_length = config.get("max_seq_length", 2048)

    # Set seed if specified
    if seed is not None:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)

    # Determine save directory
    if not save_dir:
        save_dir = model.split("/")[1] + f"-{scheme}"

    logger.info("=" * 80)
    logger.info("QUANTIZING MODEL")
    logger.info("=" * 80)
    logger.info(f"Model: {model}")
    logger.info(f"Scheme: {scheme}")
    logger.info(f"Recipe: {recipe}")
    logger.info(f"Dataset: {dataset_id}")
    logger.info(f"Save directory: {save_dir}")
    logger.info("=" * 80)

    # Run quantization
    logger.info("Running oneshot quantization...")
    oneshot_model, processor = run_oneshot_for_e2e_testing(
        model=model,
        model_class=model_class,
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
        scheme=scheme,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        recipe=recipe,
        quant_type=quant_type,
    )

    logger.info("=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)

    # Save model
    oneshot_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    # Save recipe
    recipe_path = os.path.join(save_dir, "recipe.yaml")
    session = active_session()
    recipe_yaml_str = session.get_serialized_recipe()
    with open(recipe_path, "w") as fp:
        fp.write(recipe_yaml_str)
    session.reset()

    logger.info(f"Model saved to: {save_dir}")
    logger.info("=" * 80)
    logger.info("QUANTIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Quantized model location: {save_dir}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Evaluate with HuggingFace:")
    logger.info(f"   python tests/lmeval/eval_with_vllm.py --model {save_dir} --task gsm8k")
    logger.info("")
    logger.info("2. Evaluate with vLLM and compare to baseline:")
    logger.info(f"   python tests/lmeval/eval_with_vllm.py --config {config_path} --compare-baseline")
    logger.info("=" * 80)

    return save_dir


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a model for lm-eval testing"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., tests/lmeval/configs/w4a16_awq_sym.yaml)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        save_dir = quantize_model(args.config)
        logger.success(f"✓ Quantization successful! Model saved to: {save_dir}")
    except Exception as e:
        logger.error(f"✗ Quantization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
