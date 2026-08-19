# NOTE: to use a custom dataset with your own data, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize all weights excluding down_proj layers
#       to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
#   * quantize all down_proj layer weights to fp8
#   * dynamically quantize all down_proj activations to fp8 dynamic
#       per token
scheme_0 = FP8_DYNAMIC
scheme_0["targets"] = ["re:.*down_proj.*"]
scheme_1 = NVFP4
scheme_1["targets"] = [
    "re:.*self_attn.k_proj.*",
    "re:.*self_attn.o_proj.*",
    "re:.*self_attn.q_proj.*",
    "re:.*self_attn.v_proj.*",
    "re:.*gate_proj.*",
    "re:.*up_proj.*",
]

recipe = QuantizationModifier(
    config_groups={"group_0": scheme_0, "group_1": scheme_1}, ignore=["lm_head"]
)
# Apply quantization.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=20,
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

# The model produced is compressed using two different compressors
# with two different formats: nvfp4-pack-quantized and float-quantized.
# The presence of multiple compressors is indicated by the
# `mixed-precision` format in the model's config.json.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
