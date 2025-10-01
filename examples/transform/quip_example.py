"""
NOTE: Models produced by this example will not be runnable in vLLM without
the following changes: https://github.com/vllm-project/vllm/pull/22486
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import QuIPModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
# NOTE: because the datafree pipeline is being used in this
# example, you can use additional GPUs to support larger models
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
#   * apply quip transforms to model in order to make quantization easier
#   * quantize the weights to 4 bit with a group size 128
#   * NOTE: if a model has activation shapes not divisble by 2^N, consider using
#           `random-hadamard` (random hadamard kernels will be added in the future)
recipe = [
    QuIPModifier(
        rotations=["v", "u"], transform_block_size=128, transform_type="hadamard"
    ),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
]

# Apply algorithms.
oneshot(model=model, recipe=recipe, pipeline="datafree")

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-quip-w4a16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
