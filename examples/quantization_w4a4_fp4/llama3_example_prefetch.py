"""
Example: sequential pipeline with prefetch.

Runs the sequential pipeline (cache + subgraph passes) with sequential_prefetch=True
and no quantization (recipe=None), useful for benchmarking prefetch or testing
the pipeline in isolation.

Measurements:
  The block below times the oneshot() call (calibration pass). Run with:
    time python3 examples/quantization_w4a4_fp4/llama3_example_prefetch.py
  to get real/user/sys; the script also prints the elapsed time for the calibration.
"""

import time

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)
ds = ds.map(
    lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
)
ds = ds.map(
    lambda s: tokenizer(
        s["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    remove_columns=ds.column_names,
)

# Time the calibration pass (this is what you measure for prefetch benchmarks).
start = time.perf_counter()
oneshot(
    model=model,
    dataset=ds,
    recipe=None,
    pipeline="sequential",
    sequential_prefetch=True,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
elapsed = time.perf_counter() - start
print(f"Done. Calibration took {elapsed:.1f}s.")
