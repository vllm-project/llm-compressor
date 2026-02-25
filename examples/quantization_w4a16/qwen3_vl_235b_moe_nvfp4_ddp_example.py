###############################################################################
# This script quantizes Qwen3-VL-235B-MoE with GPTQ + NVFP4 using DDP.
# run this with `torchrun --nproc_per_node=8 qwen3_vl_235b_moe_nvfp4_ddp_example.py`
# or change nproc_per_node to your desired configuration
# NOTE: Currently uses data-free GPTQ as only data-free quantization is supported for Qwen3-VL-MoE
###############################################################################

from compressed_tensors.offload import init_dist, load_offloaded_model
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

###### DDP MODEL LOAD CHANGE #####
init_dist()
with load_offloaded_model():
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto_offload"
    )
##################################

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Recipe: GPTQ + NVFP4 (data-free)
# NOTE: only datafree quantization is supported for Qwen3-VL-MoE currently
recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4A16",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)

# Apply quantization (no dataset needed for data-free GPTQ)
oneshot(model=model, recipe=recipe)

import torch

# Save to disk in compressed-tensors format.
SAVE_DIR = (
    MODEL_ID.rstrip("/").split("/")[-1]
    + "-GPTQ-NVFP4A16-DDP"
    + str(torch.distributed.get_world_size())
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
