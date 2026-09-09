# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# select a Mixture of Experts model for quantization
MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = QuantizationModifier(
    scheme="FP8",
    targets="Linear",
    ignore=["lm_head", "re:.*mlp.gate"],
)

oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
