# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

# Select model and load it.
MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Thinking"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(),
    QuantizationModifier(
        ignore=["lm_head", "re:.*mlp.gate$", "re:.*mlp.shared_expert_gate$"],
        scheme="W4A16",
        targets=["Linear"],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=256,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-sym"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
