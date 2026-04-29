from compressed_tensors.offload import get_device_map, load_offloaded_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "/mnt/data/engine/brian-dellabetta/Kimi-K2.6-bf16"
SAVE_DIR = "Kimi-K2.6-NVFP4"

# Select model and load it in the `load_offloaded_model` context
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        max_memory={"cpu": 500e9},  # remove this line to use as much cpu as possible
        trust_remote_code=True,
        offload_folder="./offload_folder",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

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
    ignore=[
        "re:.*mlp.gate$",
        "re:.*lm_head",
        "re:.*kv_a_proj_with_mqa$",
        "re:.*q_a_proj$",
        "re:.*vision_tower.*",
        "re:.*embed_tokens$",
        "re:.*norm$",
        # ignore anything not in language_model
        "re:.*mm_projector.*",
        "re:.*vision.*",
    ],
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
