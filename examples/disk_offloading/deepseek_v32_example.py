import torch
from compressed_tensors.modeling.deepseekv32.model import DeepseekV32ForCausalLM
from compressed_tensors.offload import get_device_map, load_offloaded_model
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# from transformers import AutoConfig, AutoModelForCausalLM
# AutoConfig.register("deepseek_v32", DeepseekV32Config)
# AutoModelForCausalLM.register(DeepseekV32Config, DeepseekV32ForCausalLM)


model_id = "/mnt/data/brian-dellabetta/DeepSeek-V3.2-bf16"
# model_id = "nvidia/DeepSeek-R1-NVFP4"

SAVE_DIR = "DeepSeek-V3.2-NVFP4-FP8-BLOCK"
# SAVE_DIR = "DeepSeek-V3.2-W4A16"

# Select model and load it in the `load_offloaded_model` context
with load_offloaded_model(), torch.no_grad():
    model = DeepseekV32ForCausalLM.from_pretrained(
        model_id,
        dtype="auto",
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        trust_remote_code=True,
        offload_folder="./offload_folder",
        max_memory={"cpu": 500e9},  # don't exceed 500GB RAM
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

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
#   * quantize mlp weights to NVFP4
#   * quantize self_attn weights to FP8_BLOCK
recipe = QuantizationModifier(
    config_groups={
        "config_group_0": QuantizationScheme(
            targets=[
                r"re:model.*mlp.*(gate|up|down|gate_up)_proj$",
            ],
            **NVFP4,
        ),
        "config_group_1": QuantizationScheme(
            targets=[
                # NOTE: leaving weights_proj in bf16
                r"re:model.*self_attn.indexer.(wk|wq_b)$",
                r"re:model.*self_attn.kv_a_proj_with_mqa$",
                r"re:model.*self_attn.(kv_b|o|q_a|q_b)_proj$",
            ],
            **FP8_BLOCK,
        ),
    },
    # targets=[
    #     r"re:model.*mlp.*(gate|up|down|gate_up)_proj$",
    #     r"re:model.*self_attn.indexer.(wk|wq_b)$",
    #     r"re:model.*self_attn.kv_a_proj_with_mqa$",
    #     r"re:model.*self_attn.(kv_b|o|q_a|q_b)_proj$",
    #     r"re:model.*self_attn.fused_qkv_a_proj$",
    # ],
    # scheme="W4A16",
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
