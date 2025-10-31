from transformers import AutoProcessor, AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "//proving-grounds/engine/hub_cache/models--moonshotai--Kimi-Linear-48B-A3B-Instruct/snapshots/fd1de6347c9d3896f6df8edc529c68942bdd58f6"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

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
        "re:.*gate$",
        "re:.*self_attn$",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = "/raid/engine/hub_cache/Kimi-Linear-48B-A3B-Instruct" + "-FP8-DYNAMIC"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
