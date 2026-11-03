"""Quantize GLM-4.7-Flash's backbone to FP8 and copy its BF16 MTP unchanged."""

import torch
from compressed_tensors.offload import set_onload_device
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "zai-org/GLM-4.7-Flash"
SAVE_DIR = "GLM-4.7-Flash-FP8-Dynamic-BF16-MTP"

# Keep MTP unloaded: saving copies its source tensors and ignores them for
# quantization. Only the backbone participates in oneshot.
with load_context(load_mtp=False):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder="offload_folder",
    )
set_onload_device(model, "cuda")

recipe = QuantizationModifier(
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", r"re:.*\.mlp\.gate$"],
)
oneshot(
    model=model,
    processor=AutoTokenizer.from_pretrained(MODEL_ID),
    recipe=recipe,
    output_dir=SAVE_DIR,
)
