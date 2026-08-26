# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4 (fp4 with per-group-16 scales)
#   * use imatrix_mse observer to weight quantization error by channel importance
recipe = [
    QuantizationModifier(
        scheme="NVFP4A16",
        targets="Linear",
        ignore=["lm_head"],
        weight_observer="imatrix_mse",
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    preprocessing_num_workers=32,
    dataloader_num_workers=32,
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
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16-imatrix"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
