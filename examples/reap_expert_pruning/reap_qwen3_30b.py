# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier

# Select model and load it.
model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prune 25% of the experts in each MoE layer, based on saliency.
# You can adjust this value to prune more or less aggressively.
recipe = REAPPruningModifier(sparsity=0.25)

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
    moe_calibrate_all_experts=False,  # Disable calibrating all experts for REAP
)

# Confirm generations of the compressed model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
