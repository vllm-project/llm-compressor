# NOTE: The following example no longer includes finetuning as training.

# Training support has been deprecated as of v0.9.0. To apply finetuning
# to your sparse model, see the Axolotl integration blog post for best
# fine tuning practices
# https://developers.redhat.com/articles/2025/06/17/axolotl-meets-llm-compressor-fast-sparse-open

# DEPRECATION WARNING: The marlin24 compression format is deprecated and will
# be removed in a future release, as vLLM no longer supports marlin24 models.
# See https://github.com/vllm-project/llm-compressor/issues/2267 for details.

import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

# load the model in as bfloat16 to save on memory and compute
model_stub = "neuralmagic/Llama-2-7b-ultrachat200k"
model = AutoModelForCausalLM.from_pretrained(model_stub, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_stub)

# uses LLM Compressor's built-in preprocessing for ultra chat
dataset = "ultrachat-200k"

# Select the recipe for 2 of 4 sparsity and 4-bit activation quantization
recipe = "2of4_w4a16_recipe.yaml"

# save location of quantized model
output_dir = "output_llama7b_2of4_w4a16_channel"
output_path = Path(output_dir)

# set dataset config parameters
splits = {"calibration": "train_gen[:5%]"}
max_seq_length = 512
num_calibration_samples = 10
preprocessing_num_workers = 64

oneshot_kwargs = dict(
    dataset=dataset,
    recipe=recipe,
    num_calibration_samples=num_calibration_samples,
    preprocessing_num_workers=preprocessing_num_workers,
    splits=splits,
)

# Models are automatically saved in
# ./output_llama7b_2of4_w4a16_channel/ + (sparsity/quantization)_stage

# Oneshot sparsification
oneshot(
    model=model,
    **oneshot_kwargs,
    output_dir=output_dir,
    stage="sparsity_stage",
)

# Oneshot quantization
quantized_model = oneshot(
    model=(output_path / "sparsity_stage"),
    **oneshot_kwargs,
    stage="quantization_stage",
)
quantized_model.save_pretrained(
    f"{output_dir}/quantization_stage", skip_sparsity_compression_stats=False
)
tokenizer.save_pretrained(f"{output_dir}/quantization_stage")

warnings.warn(
    "The marlin24 compression format is deprecated and will be removed in a future "
    "release, as vLLM no longer supports marlin24 models. "
    "See https://github.com/vllm-project/llm-compressor/issues/2267 for details.",
    DeprecationWarning,
    stacklevel=2,
)
