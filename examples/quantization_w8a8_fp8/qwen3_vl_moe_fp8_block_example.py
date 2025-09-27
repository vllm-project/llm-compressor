
from transformers import Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# Load model.
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
model = replace_modules_for_calibration(model)
# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with block size 128 via ptq
#   * quantize the activations to fp8 with dynamic group activations
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*lm_head",
        "re:.*router",
        "re:visual.*",
        "re:model.visual.*",
        're:.*mlp.gate$',
        're:.*mlp.shared_expert_gate$',
        "re:multi_modal_projector.*",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = "/proving-grounds/engine/hub_cache/Qwen3-VL-235B-A22B-Instruct" + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR)
