# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
#############################################################################
# This script is adapted from ./qwen3_moe_example.py and adds DDP functionality.
# run this with `torchrun --nproc_per_node=2 qwen3_moe_example_ddp.py`
# or change nproc_per_node to your desired configuration
#############################################################################

import time

import torch
from compressed_tensors.offload import dispatch_model, init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

# Select model and load it.
MODEL_ID = "Qwen/Qwen3-30B-A3B"

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto_offload")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
# NOTE: vllm currently does not support asym MoE, using symmetric here
recipe = [
    AWQModifier(),
    QuantizationModifier(
        scheme="W4A16",
        targets=["Linear"],
        ignore=["lm_head", "re:.*mlp.gate$"],
    ),
]

torch.accelerator.reset_peak_memory_stats()
start_time = time.time()

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=512,
    num_calibration_samples=256,
)

elapsed_time = time.time() - start_time
peak_memory_gb = torch.accelerator.max_memory_allocated() / (1024**3)
print("Quantization Complete")
print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = (
    MODEL_ID.rstrip("/").split("/")[-1]
    + "-awq-sym-DDP"
    + str(torch.distributed.get_world_size())
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
