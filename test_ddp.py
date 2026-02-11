from contextlib import contextmanager
import argparse
import json
from datetime import datetime
import torch
from compressed_tensors.offload import offload_model
from compressed_tensors.offload.dispatch import remove_dispatch
from loguru import logger
import torch.distributed as dist
import inspect
import os
import psutil


import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.datasets import get_rank_partition

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run DDP quantization test")
parser.add_argument(
    "--model_id",
    type=str,
    default="meta-llama/Meta-Llama-3-8B-Instruct",
    help="Model ID to load from HuggingFace (default: meta-llama/Meta-Llama-3-8B-Instruct)"
)
parser.add_argument(
    "--device_map",
    type=str,
    default="cpu",
    help="Device map for model loading (default: cpu)"
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="Directory to save the quantized model (default: None, auto-generated from model_id)"
)
parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help="Path to save run metrics as JSON (default: None, no metrics file saved)"
)
args = parser.parse_args()

### USER API: torchrun --nproc_per_node=2 test_ddp.py --<args or just leave defaults>
args = parser.parse_args()

from compressed_tensors.offload import offload_model


MODEL_ID = args.model_id
with offload_model(): # <- context manager to wrap from_pretrained
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map=args.device_map)
# model.model.config.num_hidden_layers = 3
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512


ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
ds = ds.shuffle(seed=42)
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"],tokenize=False,)}
ds = ds.map(preprocess)
# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
import time
start = time.time()
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    # recipe=None,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    # pipeline="sequential"
)
elapsed_time = time.time() - start
print(f"\nPipeline took {elapsed_time} seconds, rank={dist.get_rank()}")
peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB, rank={dist.get_rank()}\n")

# Gather metrics from all ranks
rank = dist.get_rank()
world_size = dist.get_world_size()
metrics_list = [None] * world_size
local_metrics = {"rank": rank, "elapsed_time": elapsed_time, "peak_memory_gb": peak_memory_gb}
dist.all_gather_object(metrics_list, local_metrics)

max_time = max(m["elapsed_time"] for m in metrics_list)
max_memory = max(m["peak_memory_gb"] for m in metrics_list)

# Confirm generations of the quantized model look sane.
sample_generation = None
generation_time = None
gen_start = time.time()
print("dispatching for generation...")
dispatch_for_generation(model)
if  dist.get_rank() == 0:
    print("generating sample...", time.time() - gen_start)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    sample_generation = tokenizer.decode(output[0])
    generation_time = time.time() - gen_start
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    print(sample_generation)
    print("==========================================\n\n")

dist.barrier()
save_time = None
if args.save_dir:
    print("saving...")
    save_start = time.time()
    SAVE_DIR = args.save_dir
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)
    save_time = time.time() - save_start
    print("saved")
dist.barrier()

# Export metrics to data file
if dist.get_rank() == 0 and args.output_file:
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "device_map": args.device_map,
        "save_dir": args.save_dir,
        "dataset_id": DATASET_ID,
        "dataset_split": DATASET_SPLIT,
        "num_calibration_samples": NUM_CALIBRATION_SAMPLES,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "world_size": world_size,
        "quantization_scheme": "W4A16",
        "max_time": max_time,
        "max_memory": max_memory,
        "sample_generation": sample_generation,
        "generation_time": generation_time,
        "save_time": save_time,
        "metrics_list": metrics_list
    }

    # Read existing data if file exists
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            existing_data = json.load(f)
            # Handle both array and single object for backwards compatibility
            if isinstance(existing_data, list):
                all_runs = existing_data
            else:
                all_runs = [existing_data]
    else:
        all_runs = []

    # Append new run
    all_runs.append(output_data)

    # Write back
    with open(args.output_file, 'w') as f:
        json.dump(all_runs, f, indent=2)
    print(f"Metrics exported to {args.output_file}")
dist.barrier()

dist.destroy_process_group()