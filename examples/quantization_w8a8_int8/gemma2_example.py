# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# 1) Select model and load it.
MODEL_ID = "google/gemma-2-2b-it"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Select quantization algorithms. In this case, we:
#   * quantize the weights to int8 with GPTQ (static per channel)
#   * quantize the activations to int8 (dynamic per token)
recipe = GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])

# 3) Apply quantization
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Confirm generations of the quantized model look sane.
# NOTE: transformers 4.49.0 results in a generation error with gemma2.
# Consider either downgrading your transformers version to a previous version
# or use vLLM for sample generation.
# Note: compile is disabled: https://github.com/huggingface/transformers/issues/38333
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20, disable_compile=True)
print(tokenizer.decode(output[0]))
print("==========================================")

# 4) Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-INT8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
