# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

# Load the model
model_id = "zai-org/GLM-4.7"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

moe_ignores = [
    # Layers 0-2: Dense layers - ignore entire layers
    "re:model.layers.0.*",
    "re:model.layers.1.*",
    "re:model.layers.2.*",
    # Ignore the output head
    "lm_head",
]

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = [
    AWQModifier(),
    QuantizationModifier(targets="Linear", scheme="W4A16", ignore=moe_ignores),
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

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
