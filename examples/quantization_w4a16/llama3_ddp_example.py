# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
#############################################################################
# This script is adapted from ./llama3_example.py and adds DDP functionality.
# run this with `torchrun --nproc_per_node=2 llama3_ddp_example.py`
# or change nproc_per_node to your desired configuration
#############################################################################

import time

import torch
from compressed_tensors.offload import dispatch_model, init_dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto_offload")

tokenizer = AutoTokenizer.from_pretrained(model_id)

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])


torch.accelerator.reset_peak_memory_stats()
start_time = time.time()


# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=512,
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
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

print("Saving...")
# Save to disk compressed.
SAVE_DIR = (
    model_id.rstrip("/").split("/")[-1]
    + "-W4A16-G128-DDP"
    + str(torch.distributed.get_world_size())
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
