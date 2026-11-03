"""Quantize Qwen3.8-27B's text backbone and MTP decoder to FP8_DYNAMIC."""

import json
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"
SAVE_DIR = "Qwen3.8-27B-FP8-Dynamic-MTP"

with load_context(Qwen3_5ForConditionalGeneration, load_mtp=True):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu"
    )
assert hasattr(model, "mtp"), "This example requires MTP loading support."

# Keep vision, gated DeltaNet attention, the shared head, and MTP fusion in BF16.
recipe = QuantizationModifier(
    scheme="FP8_DYNAMIC",
    targets=[r"re:^model\.language_model\.layers\.", r"re:^mtp\.layers\."],
    ignore=[r"re:.*\.linear_attn\..*", r"re:.*\.eh_proj$"],
)
oneshot(model=model, recipe=recipe, output_dir=SAVE_DIR)

# vLLM names the text backbone language_model.model; preserve HF names too.
target_map = {
    r"re:^model\.language_model\.layers\.": (
        r"re:^(?:model\.language_model|language_model\.model)\.layers\."
    ),
}
config_path = Path(SAVE_DIR) / "config.json"
config = json.loads(config_path.read_text())
for scheme in config["quantization_config"]["config_groups"].values():
    scheme["targets"] = [target_map.get(target, target) for target in scheme["targets"]]
config_path.write_text(json.dumps(config, indent=2) + "\n")
