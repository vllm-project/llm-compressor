"""Quantize DeepSeek V3's backbone to NVFP4 and its MTP decoder to FP8.

NVFP4 requires backbone calibration; FP8_DYNAMIC keeps MTP quantization data-free.
"""

import json
from pathlib import Path

import torch
from compressed_tensors.offload import set_onload_device
from transformers import AutoTokenizer, DeepseekV3ForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Use the BF16 conversion, which includes MTP, instead of the original FP8 source.
MODEL_ID = "RedHatAI/DeepSeek-V3-BF16"
SAVE_DIR = "DeepSeek-V3-NVFP4-FP8-Dynamic-MTP"

with load_context(DeepseekV3ForCausalLM, load_mtp=True):
    model = DeepseekV3ForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder="offload_folder",
    )
assert hasattr(model, "mtp"), "This example requires Transformers MTP loading support."
set_onload_device(model, "cuda")

# Separate groups preserve both QuantizationModifiers in the saved recipe.
recipe = [
    QuantizationModifier(
        group="backbone",
        scheme="NVFP4",
        targets=[r"re:^model\.layers\."],
        ignore=[
            r"re:.*\.mlp\.gate$",
            # vLLM uses BF16 copies of this projection for MLA decoding.
            r"re:^model\.layers\.\d+\.self_attn\.kv_b_proj$",
        ],
    ),
    QuantizationModifier(
        group="mtp",
        scheme="FP8_DYNAMIC",
        targets=[r"re:^mtp\.layers\."],
        # vLLM uses an unquantized Linear for the MTP input projection.
        ignore=[r"re:.*\.mlp\.gate$", r"re:.*\.eh_proj$"],
    ),
]
oneshot(
    model=model,
    processor=AutoTokenizer.from_pretrained(MODEL_ID),
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    num_calibration_samples=512,
    max_seq_length=2048,
    # Calibrate NVFP4 first, then observe MTP weights without a forward pass.
    pipeline="independent",
    # Calibrate one decoder layer at a time, including all of its experts.
    # Treat MTP as a leaf so backbone calibration does not trace its forward.
    sequential_targets=["DeepseekV3DecoderLayer", "MtpModel"],
    output_dir=SAVE_DIR,
)

# MTP loads under mtp.layers.0, but saves as model.layers.61. Saving remaps
# weights and ignores; scheme targets still need checkpoint names. Exclude
# that final layer from the NVFP4 target so the two schemes remain disjoint.
mtp_layer = model.config.num_hidden_layers
target_map = {
    r"re:^model\.layers\.": rf"re:^model\.layers\.(?!{mtp_layer}\.)",
    r"re:^mtp\.layers\.": rf"re:^model\.layers\.{mtp_layer}\.",
}
config_path = Path(SAVE_DIR) / "config.json"
config = json.loads(config_path.read_text())
for scheme in config["quantization_config"]["config_groups"].values():
    scheme["targets"] = [target_map.get(target, target) for target in scheme["targets"]]
config_path.write_text(json.dumps(config, indent=2) + "\n")
