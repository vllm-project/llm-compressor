from contextlib import contextmanager
import torch
from compressed_tensors.offload import offload_model
from compressed_tensors.offload.dispatch import remove_dispatch
from loguru import logger
import torch.distributed as dist
import inspect
import os

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
        elif device_map == "cpu":
            # we only want to load into cpu once
            kwargs["device_map"]="cpu" if local_rank == 0 else "meta" 
            case="cpu"
        elif device_map is None:
            logger.warning("No device_map given to from_pretrained, defaulting to cpu")
            kwargs["device_map"]="cpu" if local_rank == 0 else "meta" 
            case="cpu"
        elif device_map == "disk":
            raise NotImplementedError(f"device_map == {device_map} is not implemented, use cpu or cuda")
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

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
with ct_offload(): # <- context manager to wrap from_pretrained
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="cpu")

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
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)
print(f"\nPipeline took {time.time()-start} seconds, rank={dist.get_rank()}")
peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB, rank={dist.get_rank()}\n")
# Confirm generations of the quantized model look sane.

if  dist.get_rank() == 0:
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
        model.device
    )
    output = model.generate(input_ids, max_new_tokens=100)
    print("\n\n")
    print("========== SAMPLE GENERATION ==============")
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

dist.barrier()
# if dist.get_rank() == 0:
#     SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-GPTQ-ddp"
#     model.save_pretrained(SAVE_DIR, save_compressed=True)
#     tokenizer.save_pretrained(SAVE_DIR)
# dist.barrier()

dist.destroy_process_group()

# CASE 1
# DEVICE_MAP = CPU -> Load only on rank 0 then use DistributedCPUCache
# SEQUENTIAL PIPELINE

# CASE 2
# DEVICE_MAP = CUDA -> Load whole model for each rank
# BASIC PIPELINE

# CASE 3
# DEVICE_MAP = DISK -> Load only on rank 0 then use ... TODO once dist disk cache is done
# SEQUENTIAL PIPELINE

# ---- OUT OF SCOPE?

# CASE 4 # TODO
#  ... -> Load Model into multiple GPUs for each rank
# BASIC PIPELINE

# CASE 5
# ... -> offload to rank0 gpus
# SEQUENTIAL PIPELINE