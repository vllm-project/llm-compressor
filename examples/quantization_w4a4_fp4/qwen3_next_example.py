# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
# NOTE: Requires a minimum of transformers 4.57.0
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"

# Load model.
with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

# Apply quantization.
# MoE calibration is now handled automatically by the pipeline.
# We set `moe_calibrate_all_experts` to True to ensure all experts receive
# calibration data. This temporarily updates the model definition to use
# `CalibrationQwen3NextSparseMoeBlock` (from `llmcompressor.modeling.qwen3_next_moe`)
# which replaces the original `Qwen3NextSparseMoeBlock` class.
# This updates how the forward pass is handled in the MoE block during calibration.
# Feel free to update the definition under
# llm-compressor/src/llmcompressor/modeling/qwen3_next_moe.py to play around with
# this behavior and evaluate its impact on quantization performance.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=20,
    moe_calibrate_all_experts=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
