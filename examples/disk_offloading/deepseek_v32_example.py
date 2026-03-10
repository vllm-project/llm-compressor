from compressed_tensors.offload import get_device_map, load_offloaded_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

from compressed_tensors.modeling.deepseekv32.config import (
    ModelConfig as DeepseekV32Config,
)
from compressed_tensors.modeling.deepseekv32.model import DeepseekV32ForCausalLM

# from transformers.models.deepseek_v3 import (
#     DeepseekV3Config as DeepseekV32Config,
#     DeepseekV3ForCausalLM as DeepseekV32ForCausalLM,
# )


from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("deepseek_v32", DeepseekV32Config)
AutoModelForCausalLM.register(DeepseekV32Config, DeepseekV32ForCausalLM)


model_id = "/mnt/nvme_stripe/playground/brian-dellabetta/DeepSeek-V3.2-bf16"
# model_id = "nvidia/DeepSeek-R1-NVFP4"

SAVE_DIR = "DeepSeek-V3.2-NVFP4"

# Select model and load it in the `load_offloaded_model` context
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        trust_remote_code=True,
        offload_folder="./offload_folder",
        # max_memory={"cpu": 500e9},  # don't exceed 500GB RAM
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("LOADED")

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
    targets=[
        # "Linear",
        r"re:model.layers.1\..*proj.*",
    ],
    scheme="NVFP4",
    ignore=["lm_head"],
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
