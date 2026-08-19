# NOTE: to use a custom dataset with your own data, see examples/custom_dataset_example.py
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

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Time the calibration pass (this is what you measure for prefetch benchmarks).
start = time.perf_counter()
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=None,
    pipeline="sequential",
    sequential_prefetch=True,
    max_seq_length=2048,
    num_calibration_samples=20,
)
elapsed = time.perf_counter() - start
print(f"Done. Calibration took {elapsed:.1f}s.")
