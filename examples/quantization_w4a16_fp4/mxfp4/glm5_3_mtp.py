"""Quantize full GLM-5.3 and its MTP layer using the same MXFP4 recipe.

The native FP8 backbone load requires Transformers with padded-block
dequantization support; 5.17.0 fails on GLM-5.3's 576-row kv_a projection.
"""

import os

import torch
from compressed_tensors.offload import init_dist, set_onload_device
from transformers import AutoModelForCausalLM, FineGrainedFP8Config

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "zai-org/GLM-5.3"
SAVE_DIR = os.environ.get("MTP_OUTPUT_DIR", "GLM-5.3-MXFP4-MTP")

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=FineGrainedFP8Config(dequantize=True),
        dtype=torch.bfloat16,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder=os.environ.get("MTP_OFFLOAD_DIR", "offload_folder"),
    )
set_onload_device(model, "cuda")

recipe = QuantizationModifier(
    scheme="MXFP4",
    targets=["Linear", r"re:^mtp\.layers\."],
    ignore=[
        "lm_head",
        r"re:.*\.eh_proj$",
        r"re:.*\.mlp\.gate$",
        # vLLM packs wk and weights_proj into a dense indexer projection.
        r"re:.*\.indexer\.(?:wk|weights_proj)$",
    ],
)
oneshot(model=model, recipe=recipe, output_dir=SAVE_DIR)
torch.distributed.destroy_process_group()
