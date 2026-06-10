from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.moe.linearize import load_quantizable_moe
from llmcompressor.modifiers.quantization import QuantizationModifier

"""Please see details in `README_granite4.md`."""

# Load model.
model_id = "ibm-granite/granite-4.0-tiny-preview"
with load_quantizable_moe():
    model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

recipe = QuantizationModifier(
    targets=["Linear"],
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", "re:.*block_sparse_moe.router"],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer(
    "What is your favorite TV show?", return_tensors="pt"
).input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = "ibm-granite-4-tiny-fp8-dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
