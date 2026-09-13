import time

from compressed_tensors.offload import (
    get_device_map,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

start_time = time.time()

# Select model and load it in the `load_context` context
# In this example, we emulate large model quantization with disk offloading by
# restricting the theoretical size of CPU RAM to be smaller than the size of the model
model_id = "google/gemma-4-26B-A4B"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        max_memory={
            0: 0,
            "cpu": 16e9,
        },  # remove this line to use as much cpu as possible
        offload_folder="./offload_folder",  # folder to store offloaded weights
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

devices = {offloaded for _onloaded, offloaded in get_device_map(model).values()}
print(f"Model was offloaded to the following devices: {devices}")

# Select calibration dataset.
DATASET_ID = "wikitext"
DATASET_SUBSET = "wikitext-2-v1"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

# Apply algorithms.
oneshot(
    model=model,
    dataset=DATASET_ID,
    dataset_config_name=DATASET_SUBSET,
    splits=f"train[:{NUM_CALIBRATION_SAMPLES}]",
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_linearize_moe=False,  # True or False
)

print(f"Quantization completed in {time.time() - start_time:.2f} seconds.")

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
