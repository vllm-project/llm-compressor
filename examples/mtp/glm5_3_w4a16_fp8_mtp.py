"""Quantize GLM-5.3's backbone to W4A16 and its MTP decoder to FP8.

RTN and FP8_DYNAMIC quantize both parts without calibration data.
"""

import json
from pathlib import Path

import torch
from compressed_tensors.offload import set_onload_device
from transformers import GlmMoeDsaForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "zai-org/GLM-5.3-BF16"
SAVE_DIR = "GLM-5.3-W4A16-G128-FP8-Dynamic-MTP"

with load_context(GlmMoeDsaForCausalLM, load_mtp=True):
    model = GlmMoeDsaForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder="offload_folder",
    )
assert hasattr(model, "mtp"), "This example requires MTP loading support."
set_onload_device(model, "cuda")

# The source sets top_p without enabling sampling; clear it before saving.
model.generation_config.top_p = None

# Separate groups preserve both modifiers in the saved recipe.
recipe = [
    QuantizationModifier(
        group="backbone",
        scheme="W4A16",
        targets=[r"re:^model\.layers\."],
        ignore=[
            r"re:.*\.mlp\.gate$",
            # vLLM fuses these indexer projections into an unquantized Linear.
            r"re:.*\.self_attn\.indexer\.(?:wk|weights_proj)$",
            # Keep the MLA projection used to build BF16 decoding weights dense.
            r"re:^model\.layers\.\d+\.self_attn\.kv_b_proj$",
        ],
    ),
    QuantizationModifier(
        group="mtp",
        scheme="FP8_DYNAMIC",
        targets=[r"re:^mtp\.layers\."],
        ignore=[
            r"re:.*\.mlp\.gate$",
            r"re:.*\.self_attn\.indexer\.(?:wk|weights_proj)$",
            # vLLM uses an unquantized Linear for the MTP input projection.
            r"re:.*\.eh_proj$",
        ],
    ),
]
oneshot(
    model=model,
    recipe=recipe,
    # Apply each data-free modifier in its own epoch.
    pipeline="independent",
    output_dir=SAVE_DIR,
)

# MTP loads under mtp.layers.0, but saves as model.layers.78. Saving remaps
# weights and ignores; scheme targets still need checkpoint names. Exclude
# that final layer from the INT4 target so the two schemes remain disjoint.
mtp_layer = model.config.num_hidden_layers
target_map = {
    r"re:^model\.layers\.": rf"re:^model\.layers\.(?!{mtp_layer}\.)",
    r"re:^mtp\.layers\.": rf"re:^model\.layers\.{mtp_layer}\.",
}
config_path = Path(SAVE_DIR) / "config.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
for scheme in config["quantization_config"]["config_groups"].values():
    scheme["targets"] = [target_map.get(target, target) for target in scheme["targets"]]
config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
