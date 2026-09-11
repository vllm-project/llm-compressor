# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
# W4AFP8 scheme: 4-bit integer weights (group 128) + FP8 dynamic per-token activations
# AWQ smooths the weights before quantization to reduce quantization error.
recipe = [
    AWQModifier(duo_scaling=True),
    QuantizationModifier(
        ignore=["lm_head"],
        scheme="W4AFP8",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=256,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
# Use quantization_format="pack-quantized" for vLLM compatibility
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-w4a8-fp8"
model.save_pretrained(
    SAVE_DIR, save_compressed=True, quantization_format="pack-quantized"
)
tokenizer.save_pretrained(SAVE_DIR)
