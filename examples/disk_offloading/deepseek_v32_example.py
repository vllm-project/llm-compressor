import json

import torch
from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    convert_checkpoint,
)
from compressed_tensors.offload import load_offloaded_model
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.deepseekv32.model import DeepseekV32ForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier

# This script first dequantizes the original DeepSeek-V3.2 checkpoint to bfloat16,
# then quantizes attention layers to FP8_BLOCK and mlp layers to NVFP4.
# Result is available at https://huggingface.co/RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK

# `deepseek-ai/DeepSeek-V3.2` has layers that are quantized in the FP8 quant method's
# FP8_BLOCK scheme. This example will first convert the checkpoint to bfloat16 so that
# the model can be compressed in compressed-tensors format, in another configuration.
MODEL_ID = "deepseek-ai/DeepSeek-V3.2"
BFLOAT16_SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-bf16"
SAVE_DIR = "DeepSeek-V3.2-NVFP4-FP8-BLOCK"

# 1) Convert DeepSeek-V3.2 back to dense bfloat16 format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=BFLOAT16_SAVE_DIR,
    converter=FP8BlockDequantizer(
        # `deepseek-ai/DeepSeek-V3.2` fp8-block-quantized layers, found by inspection
        targets=[
            r"re:.*mlp.*\.(gate_up|gate|up|down)_proj$",
            r"re:.*self_attn.*\.(kv_b|o|q_a|q_b)_proj$",
            r"re:.*self_attn.kv_a_proj_with_mqa$",
            r"re:.*self_attn.indexer.(wk|wq_b)$",
        ],
    ),
    max_workers=4,
)

# 2) For some reason DeepSeek splits important config info into a separate file that
#    will break loading in transformers if not merged into config.json
with open(f"{BFLOAT16_SAVE_DIR}/config.json", "r") as f:
    orig_config = json.load(f)
with open(f"{BFLOAT16_SAVE_DIR}/inference/config_671B_v3.2.json", "r") as f:
    additional_config_data = json.load(f)
    additional_config_data.pop("dtype")
with open(f"{BFLOAT16_SAVE_DIR}/config.json", "w") as f:
    config = orig_config | additional_config_data
    json.dump(config, f)


# 3) Apply oneshot to bfloat16 model
with load_offloaded_model(), torch.no_grad():
    model = DeepseekV32ForCausalLM.from_pretrained(
        BFLOAT16_SAVE_DIR,
        dtype="auto",
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        trust_remote_code=True,
        offload_folder="./offload_folder",
        max_memory={"cpu": 500e9},  # don't exceed 500GB RAM
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

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
