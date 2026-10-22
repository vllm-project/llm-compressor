"""Quantize GLM-4.5-Air and its MTP layer to FP8_DYNAMIC."""

import torch
from compressed_tensors.offload import init_dist, set_onload_device
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "zai-org/GLM-4.5-Air"
SAVE_DIR = "GLM-4.5-Air-FP8-Dynamic-MTP"

init_dist()
with load_context(load_mtp=True):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder="offload_folder",
    )
set_onload_device(model, "cuda")

recipe = QuantizationModifier(
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", r"re:.*\.eh_proj$"],
)
oneshot(model=model, recipe=recipe, output_dir=SAVE_DIR)
torch.distributed.destroy_process_group()
