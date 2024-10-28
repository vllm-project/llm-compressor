from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
OUTPUT_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"

# Load model
# Note: device_map="auto" will offload to CPU if not enough space on GPU.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto", trust_remote_code=True
)

# Configure the quantization scheme and algorithm (PTQ + FP8_DYNAMIC).
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply quantization and save in `compressed-tensors` format.
oneshot(
    model=model,
    recipe=recipe,
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID),
)
model.save_pretrained(OUTPUT_DIR)
