from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: Qwen3-VL-MoE support is not in transformers<=4.56.2
# you may need to install transformers from source


MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# Load model.
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = replace_modules_for_calibration(model)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with channel-wise quantization
#   * quantize the activations to fp8 with dynamic token activations
# NOTE: only datafree quantization is supported for Qwen3-VL-MoE currently
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-DYNAMIC"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
