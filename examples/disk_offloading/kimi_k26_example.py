from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    convert_checkpoint,
)
from compressed_tensors.offload import get_device_map, load_offloaded_model
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# moonshotai/Kimi-K2.6 checkpoint is published in W4A16 compressed-tensors format.
# This script will first upconvert to bfloat16 so that the model can be compressed
# in another configuration. Note that the dequantized model is 1.9TB, whereas the
# original and the nvfp4 checkpoint are <600GB.

MODEL_ID = "moonshotai/Kimi-K2.6"
DEQUANTIZED_SAVE_DIR = "Kimi-K2.6-bf16"
SAVE_DIR = "Kimi-K2.6-NVFP4"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*self_attn.*",
    "re:.*embed_tokens$",
    # ignore anything not in language_model
    "re:.*mm_projector.*",
    "re:.*vision.*",
]

# Convert to dense bfloat16 format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=DEQUANTIZED_SAVE_DIR,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=4,
)

# Quantize bfloat16 checkpoint to NVFP4, limiting CPU RAM usage to 500GB
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        DEQUANTIZED_SAVE_DIR,
        dtype="auto",
        device_map="auto_offload",
        max_memory={"cpu": 500e9},
        trust_remote_code=True,
        offload_folder="./offload_folder",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        DEQUANTIZED_SAVE_DIR, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        DEQUANTIZED_SAVE_DIR, trust_remote_code=True
    )

# Confirm that model is dispatched correctly
devices = {offloaded for _onloaded, offloaded in get_device_map(model).values()}
print(f"Model was offloaded to the following devices: {devices}")

# Select calibration dataset.
DATASET_ID = "ultrachat-200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=ignore,
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
