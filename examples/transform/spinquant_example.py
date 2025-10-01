from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform import SpinQuantModifier
from llmcompressor.utils import dispatch_for_generation

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# NOTE: currently only rotations R1, R2, and R4 are available
# R3 and learned R1/R2 rotations will be added in a future release.
# Configure the quantization algorithm to run.
#   * apply spinquant transforms to model to reduce quantization loss
#   * quantize the weights to 4 bit with group size 128
recipe = [
    SpinQuantModifier(
        rotations=["R1", "R2", "R4"],
        transform_block_size=128,
        transform_type="hadamard",
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
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-spinquantR1R2R4-w4a16"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
