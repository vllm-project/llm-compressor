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

def is_ddp() -> bool:
    return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1

def init_dist():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    dist.barrier()

def convert_to_ct_offload(model, case, local_rank):

    if case=="cuda":
        remove_dispatch(model)

    if case == "cpu":
        if any(hasattr(m, '_hf_hook') for m in model.modules()):
            raise NotImplementedError(
                "Detected that model didn't fit entirely in ram and accelerate used "
                "disk offloading, need CT Distributed Disk Offloading to support this"
            )
        remove_dispatch(model)
        onload_device = torch.device(f"cuda:{local_rank}")
        offload_device = torch.device("cpu")
        offload_model(model, onload_device, offload_device)
    return model

def patch_from_pretrained(cls, local_rank):
    cls.from_pretrained_orig = cls.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        ### OVERWRITE DEVICE MAP TO HANDLE DISTRIBUTED CASE
        device_map = kwargs.get("device_map")
        case=None
        if device_map == "cuda":
            kwargs["device_map"]=local_rank
            case="cuda"
        elif device_map in ["cpu"]:
            # we only want to load into cpu once
            if local_rank != 0:
                kwargs["device_map"]="meta" 
        elif device_map in ["disk"]:    
            if local_rank == 0:
                kwargs["device_map"]="auto"
                kwargs["max_memory"] = {"cpu": 296049920}
            else:
                kwargs["device_map"]="meta" 


        elif device_map is None:
            logger.warning("No device_map given to from_pretrained, defaulting to cpu")
            kwargs["device_map"]="cpu" if local_rank == 0 else "meta" 
            case="cpu"
        elif device_map == "auto":
            raise NotImplementedError(f"device_map == {device_map} is not implemented, use cpu or cuda")
        else:
            raise NotImplementedError(f"device_map == {device_map} is not implemented, use cpu or cuda")
        
        ### LOAD WITH ACCELERATE + CORRECTED DEVICE MAP
        model = cls.from_pretrained_orig(*args, **kwargs)

        ### CONVERT FROM ACCELERATE TO OUR OFFLOADING TOOL
        model = convert_to_ct_offload(model, case, local_rank)

        ### PATCH SAVE_PRETRAINED SO IT WiLL WORK WITH CT OFFLOAD
        # model = patch_save_pretrained(model)

        return model

    cls.from_pretrained = patched_from_pretrained
    return cls


@contextmanager
def ct_offload():
    if not is_ddp():
        init_dist()
    ### Finds the correct frame with imports to patch
    frame = inspect.currentframe()
    while frame:
        # Skip frames from contextlib module
        if 'contextlib' not in frame.f_code.co_filename:
            caller_globals = frame.f_globals
            break
        frame = frame.f_back
    else:
        raise RuntimeError("Could not find caller frame")

    local_rank = dist.get_rank()
    # wrap from_pretrained
    # to swap accelerate offloading for CT offloading
    # wrap save_pretrained
    # to work with CT offloading
    patched = []
    for _, load_cls in caller_globals.items():
        if (
            hasattr(load_cls, 'from_pretrained') and
            hasattr(load_cls, '__module__') and
            'transformers' in load_cls.__module__
        ):
            patched.append(load_cls)
            patch_from_pretrained(load_cls, local_rank)

    yield

    ### CLEANUP #####
    for load_cls in patched:
        load_cls.from_pretrained = load_cls.from_pretrained_orig
        del load_cls.from_pretrained_orig
        

### USER API: torchrun --nproc_per_node=2 test_ddp.py
### START OF TEST
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

MODEL_ID = args.model_id
with ct_offload(): # <- context manager to wrap from_pretrained
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






"""
            for module in module_list:
                src_rank = module_to_rank[module]

                # Get parameters from module
                weight = getattr(module, "weight")
                weight_scale = getattr(module, "weight_scale")
                weight_zero_point = getattr(module, "weight_zero_point")
                weight_g_idx = getattr(module, "weight_g_idx", None)

                # Store for later update
                module_params.append((module, weight, weight_scale, weight_zero_point, weight_g_idx))

                # Broadcast each tensor asynchronously
                weight_comm = dist.broadcast(weight, src=src_rank, async_op=True)
                scale_comm = dist.broadcast(weight_scale, src=src_rank, async_op=True)
                zp_comm = dist.broadcast(weight_zero_point, src=src_rank, async_op=True)
                pending_comms.extend([weight_comm, scale_comm, zp_comm])

                if weight_g_idx is not None:
                    gidx_comm = dist.broadcast(weight_g_idx, src=src_rank, async_op=True)
                    pending_comms.append(gidx_comm)

            # Wait for all broadcasts to complete
            self._wait_for_comms(pending_comms)

            # Update all parameters
            for module, weight, weight_scale, weight_zero_point, weight_g_idx in module_params:
                update_offload_parameter(module, "weight", weight)
                update_offload_parameter(module, "weight_scale", weight_scale)
                update_offload_parameter(module, "weight_zero_point", weight_zero_point)
                if weight_g_idx is not None:
                    update_offload_parameter(module, "weight_g_idx", weight_g_idx)
"""