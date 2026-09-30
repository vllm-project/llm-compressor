# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Run this example with `torchrun --nproc_per_node=N zaya1_nvfp4.py`
# WARNING: Zyphra model support in vLLM is currently under review
import torch
from compressed_tensors.offload import init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import load_context

# Load model
model_id = "Zyphra/ZAYA1-74B-preview"
init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure the quantization algorithm and scheme.
# Ignore `lm_head` and the MoE router/gate layers (`model.layers.*.mlp.gate.*`).
recipe = GPTQModifier(
    targets="Linear", scheme="NVFP4", ignore=["lm_head", r"re:.*mlp\.gate\..*"]
)

# Apply quantization.
oneshot(
    model=model,
    processor=tokenizer,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=256,
    batch_size=16,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
