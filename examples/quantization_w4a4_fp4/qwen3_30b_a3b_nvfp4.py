from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3-30B-A3B"

# Load model. Qwen3-30B-A3B is a MoE model, so we wrap the load in load_context.
with load_context(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

MAX_SEQUENCE_LENGTH = 2048

# MoE models benefit from more samples for better expert calibration.
NUM_CALIBRATION_SAMPLES = 1024

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
# The router/gate layers (mlp.gate) control expert routing and must not be
# quantized, so they are added to the ignore list along with lm_head.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", r"re:.*mlp\.gate$"],
)

# Apply quantization using the prebaked "perfectblend" calibration dataset.
# MoE calibration is handled automatically by the pipeline. We set
# `moe_calibrate_all_experts` to True to ensure all experts receive
# calibration data.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
