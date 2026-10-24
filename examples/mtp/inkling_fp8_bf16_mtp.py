"""Quantize Inkling's text backbone to FP8 and preserve its BF16 MTP weights.

Inkling supports loading MTP through load_context. Set load_mtp=False here
to quantize only the backbone and copy the original MTP tensors on save.
"""

import json
from pathlib import Path

import torch
from compressed_tensors.offload import set_onload_device
from transformers import AutoProcessor, InklingForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "thinkingmachines/Inkling"
SAVE_DIR = "Inkling-FP8-Dynamic-BF16-MTP"

# Leave MTP unloaded so save_pretrained copies its source tensors unchanged.
with load_context(InklingForConditionalGeneration, load_mtp=False):
    model = InklingForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder="offload_folder",
    )
set_onload_device(model, "cuda")

recipe = QuantizationModifier(
    scheme="FP8_DYNAMIC",
    targets="Linear",
    ignore=[
        "lm_head",
        r"re:.*\.mlp\.gate$",
        r"re:.*(vision|visual|audio).*",
    ],
)
oneshot(
    model=model,
    processor=AutoProcessor.from_pretrained(MODEL_ID),
    recipe=recipe,
    output_dir=SAVE_DIR,
)

# Saving adds ignore entries for the copied MTP modules. Add Inkling's
# model.mtp prefix explicitly to keep the whole MTP subtree outside FP8.
config_path = Path(SAVE_DIR) / "config.json"
config = json.loads(config_path.read_text())
ignore = config["quantization_config"]["ignore"]
mtp_ignore = r"re:^model\.mtp\..*"
if mtp_ignore not in ignore:
    ignore.append(mtp_ignore)
config_path.write_text(json.dumps(config, indent=2) + "\n")
