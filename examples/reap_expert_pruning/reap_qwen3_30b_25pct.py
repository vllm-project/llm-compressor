"""
REAP Expert Pruning: Qwen3-30B-A3B at 25% compression.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python examples/reap_expert_pruning/reap_qwen3_30b_25pct.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

# ── Configuration ──────────────────────────────────────────────────────────

MODEL_ID = "/home/eldarkurtic/hf_models/Qwen/Qwen3-30B-A3B-Instruct-2507"
OUTPUT_DIR = "/home/eldarkurtic/github/eldarkurtic/llm-compressor/output_dir/Qwen3-30B-A3B-Instruct-2507_REAP_sp25"
COMPRESSION_RATIO = 0.25
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQ_LENGTH = 2048
DATASET = "open_platypus"

# ── Recipe ─────────────────────────────────────────────────────────────────

RECIPE = """
pruning_stage:
  pruning_modifiers:
    REAPPruningModifier:
      compression_ratio: 0.25
"""

# ── Run pruning ────────────────────────────────────────────────────────────

print(f"Model:             {MODEL_ID}")
print(f"Compression ratio: {COMPRESSION_RATIO}")
print(f"Output:            {OUTPUT_DIR}")
print(f"Calibration:       {NUM_CALIBRATION_SAMPLES} samples, {MAX_SEQ_LENGTH} max seq len")
print()

oneshot(
    model=MODEL_ID,
    recipe=RECIPE,
    dataset=DATASET,
    output_dir=OUTPUT_DIR,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQ_LENGTH,
)

print(f"\nDone! Pruned model saved to: {OUTPUT_DIR}")

# ── Quick smoke test ───────────────────────────────────────────────────────

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
