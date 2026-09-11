# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import preset_name_to_scheme
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
model_id = "meta-llama/Meta-Llama-3.1-8B"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure the quantization algorithm to run.
#   * collect E[x²] during calibration through the imatrix_mse observer
#   * quantize the weights to 4 bit with group size 128
#   * use imatrix_mse observer to weight quantization error by channel importance
scheme = preset_name_to_scheme("W4A16", ["Linear"])
scheme.weights.observer = "imatrix_mse"

recipe = [
    QuantizationModifier(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
    ),
]

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
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128-imatrix"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
