from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-2-27b-it"

# 1) Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# 3) Apply quantization and save in compressed-tensors format.
OUTPUT_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
oneshot(
    model=model,
    recipe=recipe,
    tokenizer=tokenizer,
    output_dir=OUTPUT_DIR,
)

# Confirm generations of the quantized model look sane.
# NOTE: transformers 4.49.0 results in a generation error with gemma2.
# Consider either downgrading your transformers version to a previous version
# or use vLLM for sample generation.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")
