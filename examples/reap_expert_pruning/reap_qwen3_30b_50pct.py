"""
REAP Expert Pruning: Qwen3-30B-A3B at 25% sparsity.

Usage:
CUDA_VISIBLE_DEVICES=0,1 python examples/reap_expert_pruning/reap_qwen3_30b_25pct.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

MDL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SPARSITY = 0.5
BASE_PATH = "/home/eldarkurtic/"
OUT_PATH = "github/eldarkurtic/llm-compressor/output_dir/"
MODEL_ID = BASE_PATH + "hf_models/" + MDL
OUTPUT_DIR = BASE_PATH + OUT_PATH + MDL + "_REAP_sp" + str(int(SPARSITY * 100))
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH = 2048
DATASET = "open_platypus"

RECIPE = """
pruning_stage:
  pruning_modifiers:
    REAPPruningModifier:
      sparsity: {SPARSITY}
""".format(SPARSITY=SPARSITY)

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
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=1000, do_sample=False)

new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
response = tokenizer.decode(new_tokens, skip_special_tokens=True)

print(f"\nPrompt: {prompt}")
print(f"Response: {response}")
