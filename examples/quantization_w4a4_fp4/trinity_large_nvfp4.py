# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "arcee-ai/Trinity-Large-Thinking"

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# MoE calibration is now handled automatically by the pipeline.
# The `CalibrationAfmoeMoE` module (from `llmcompressor.modeling.afmoe`)
# will be applied during calibration to enable proper expert calibration.
# This replaces the original `AfmoeMoE` class during calibration.

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize all expert layers (routed + shared) to nvfp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
#   * skip attention layers and lm_head

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head", "re:.*self_attn.*", "re:.*mlp.router.*"],
)

# Apply quantization.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=256,
    tokenizer=tokenizer,
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
