import sys
import pdb

def exception_handler(exc_type, exc_value, exc_traceback):
    """Custom exception handler to invoke pdb on error."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow KeyboardInterrupt to exit normally
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    print(f"\nUnhandled exception: {exc_value}")
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = exception_handler



from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot

# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
