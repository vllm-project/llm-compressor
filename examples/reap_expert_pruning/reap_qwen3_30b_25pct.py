"""
REAP Expert Pruning: Qwen3-30B-A3B at 25% sparsity.

Usage:
CUDA_VISIBLE_DEVICES=0,1 python examples/reap_expert_pruning/reap_qwen3_30b_25pct.py

The model is loaded from the Hugging Face Hub by default. Override the source
model or output location with the MODEL_ID / OUTPUT_DIR environment variables.
"""

import inspect
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
SPARSITY = 0.25
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"{MODEL_ID.split('/')[-1]}_REAP_sp{int(SPARSITY * 100)}",
)
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH = 2048
DATASET = "open_platypus"

RECIPE = f"""
pruning_stage:
  pruning_modifiers:
    REAPPruningModifier:
      sparsity: {SPARSITY}
"""

oneshot(
    model=MODEL_ID,
    recipe=RECIPE,
    dataset=DATASET,
    output_dir=OUTPUT_DIR,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQ_LENGTH,
)

print(f"\nDone! Pruned model saved to: {OUTPUT_DIR}")
print("\nRunning generation smoke test...")

tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

prompt = "Write a Python function that checks if a number is prime."
messages = [{"role": "user", "content": prompt}]
chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
# enable_thinking is only accepted by some Qwen tokenizers
if "enable_thinking" in inspect.signature(tokenizer.apply_chat_template).parameters:
    chat_kwargs["enable_thinking"] = False
text = tokenizer.apply_chat_template(messages, **chat_kwargs)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"\nPrompt: {prompt}")
print(f"Response: {response}")
