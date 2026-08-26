# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

# Select model and load it.

# This script takes about 48 hours on 1xA100 to complete.
# Future improvements will reduce this runtime (#1561, #1558).

# For DeepSeek-R1, we require a full precision model in order to properly calibrate
# `DeepSeek-R1-0528-BF16` is a DeepSeek-V3 FP8 model which has been converted to BF16

model_id = "unsloth/DeepSeek-R1-0528-BF16"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationDeepseekV3MoE` modules (from `llmcompressor.modeling.deepseek_v3`)
# will be applied during calibration to enable proper expert calibration.
# These replace the original `DeepseekV3MoE` class from
# `transformers.models.deepseek_v3.modeling_deepseek_v3`.

# Configure the quantization algorithm to run.
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = GPTQModifier(
    targets="Linear", scheme="W4A16", ignore=["lm_head", "re:.*mlp.gate$"]
)

# Apply algorithms.
# due to the large size of DeepSeekV3, we specify sequential targets such that
# only one MLP is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    sequential_targets=["DeepseekV3Attention", "DeepseekV3MLP"],
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
