"""Quantize GLM-4.5-Air and its Transformers-supported MTP layer together.

For unsupported FP8 MTP layouts, see examples/model_free_ptq/mtp_fp8_fallback.py.
"""

import os

import torch
from compressed_tensors.offload import init_dist, set_onload_device
from compressed_tensors.quantization import preset_name_to_scheme
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "zai-org/GLM-4.5-Air"
SAVE_DIR = os.environ.get("MTP_OUTPUT_DIR", "GLM-4.5-Air-FP8-Dynamic-MTP")
OFFLOAD_DIR = os.environ.get("MTP_OFFLOAD_DIR", "offload_folder")

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder=OFFLOAD_DIR,
    )
set_onload_device(model, "cuda")

recipe = QuantizationModifier(
    config_groups={
        "mtp": preset_name_to_scheme("FP8_DYNAMIC", targets=[r"re:^mtp\.layers\."]),
        "backbone": preset_name_to_scheme("FP8_DYNAMIC", targets=["Linear"]),
    },
    ignore=["lm_head", r"re:.*\.eh_proj$"],
)
oneshot(model=model, recipe=recipe, output_dir=SAVE_DIR)
torch.distributed.destroy_process_group()
