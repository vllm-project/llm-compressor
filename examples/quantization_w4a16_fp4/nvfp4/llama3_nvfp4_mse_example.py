"""NVFP4 quantization with memoryless MSE range search.

The standard ``memoryless_mse`` observer uses the NVFP4 search defaults
provided by ``QuantizationModifier`` to search expanded per-group ranges.

Usage::

    python llama3_nvfp4_mse_example.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    scheme="NVFP4",
    weight_observer="memoryless_mse",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    num_calibration_samples=512,
    max_seq_length=2048,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-expanded-mse"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
