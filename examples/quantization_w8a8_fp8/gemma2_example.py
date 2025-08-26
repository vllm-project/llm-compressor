from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "google/gemma-2-27b-it"

# 1) Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# 3) Apply quantization and save in compressed-tensors format.
oneshot(
    model=model,
    recipe=recipe,
    tokenizer=tokenizer,
)

# Confirm generations of the quantized model look sane.
# NOTE: transformers 4.49.0 results in a generation error with gemma2.
# Consider either downgrading your transformers version to a previous version
# or use vLLM for sample generation.
# Note: compile is disabled: https://github.com/huggingface/transformers/issues/38333
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20, disable_compile=True)
print(tokenizer.decode(output[0]))
print("==========================================")

# 4) Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
