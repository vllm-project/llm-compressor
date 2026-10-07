"""Calibrate GLM-4.5-Air in NVFP4 and quantize its MTP layer in FP8_DYNAMIC.

Use separate modifiers so the independent pipeline calibrates only the backbone.
MTP supports data-free schemes; it does not run during backbone calibration.
"""

import os

import torch
from compressed_tensors.offload import init_dist, set_onload_device
from transformers import AutoModelForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = os.environ.get("MTP_MODEL_ID", "zai-org/GLM-4.5-Air")
SAVE_DIR = os.environ.get("MTP_OUTPUT_DIR", "GLM-4.5-Air-NVFP4-FP8-MTP")

init_dist()
with load_context(load_mtp=True):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={"cpu": os.environ.get("MTP_CPU_MEMORY", "500GiB")},
        offload_folder=os.environ.get("MTP_OFFLOAD_DIR", "offload_folder"),
    )
set_onload_device(model, "cuda")

recipe = [
    QuantizationModifier(
        scheme="NVFP4",
        targets=["Linear"],
        ignore=["lm_head", r"re:^mtp\."],
    ),
    QuantizationModifier(
        scheme="FP8_DYNAMIC",
        targets=[r"re:^mtp\.layers\."],
        ignore=[r"re:.*\.eh_proj$"],
    ),
]
oneshot(
    model=model,
    recipe=recipe,
    pipeline="independent",
    # The attached MTP decoder is not part of the backbone forward pass.
    sequential_targets=[r"re:^model\.layers\.\d+$"],
    dataset="perfectblend",
    splits="train[:512]",
    num_calibration_samples=512,
    max_seq_length=2048,
    output_dir=SAVE_DIR,
)
torch.distributed.destroy_process_group()
