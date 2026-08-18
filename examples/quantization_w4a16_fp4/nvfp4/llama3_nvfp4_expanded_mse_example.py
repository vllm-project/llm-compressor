"""NVFP4 quantization with expanded MSE observer.

The ``nvfp4_expanded_mse`` observer searches over a range of per-group
scale expansions (1.8x down to 0.8x of the observed range) to find the
scale that minimizes quantization error.  This typically gives better
quality than the default minmax observer at the cost of a longer
calibration pass.

Usage::

    python llama3_nvfp4_expanded_mse_example.py
"""

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.quant_args import FP8_E4M3_DATA
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    scheme="NVFP4",
    weight_observer="nvfp4_expanded_mse",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    dataset="ultrachat_200k",
    splits="train_sft",
    num_calibration_samples=512,
    max_seq_length=2048,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-expanded-mse"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
