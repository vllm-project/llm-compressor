# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

# Select model and load it.
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure the quantization algorithm to run.
# W4AFP8 scheme: 4-bit integer weights (group 128) + FP8 dynamic per-token activations
recipe = GPTQModifier(targets="Linear", scheme="W4AFP8", ignore=["lm_head"])

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
# Use quantization_format="pack-quantized" for vLLM compatibility
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4AFP8"
model.save_pretrained(
    SAVE_DIR, save_compressed=True, quantization_format="pack-quantized"
)
tokenizer.save_pretrained(SAVE_DIR)
