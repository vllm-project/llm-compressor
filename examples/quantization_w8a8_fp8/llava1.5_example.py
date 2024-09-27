from transformers import AutoProcessor, LlavaForConditionalGeneration

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot, wrap_hf_model_class

MODEL_ID = "llava-hf/llava-1.5-7b-hf"

# Load model.
model_class = wrap_hf_model_class(LlavaForConditionalGeneration)
model = model_class.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_tower.*"],
)

# Apply quantization and save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
oneshot(model=model, recipe=recipe, output_dir=SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")
