import time

import torch
from compressed_tensors.offload import dispatch_model, init_dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B"

# DDP: initialize distributed and load with auto_offload
init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto_offload")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {"text": "\n".join(m["content"] for m in example["messages"])}


ds = ds.map(preprocess)

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4 (fp4 with per-group-16 scales)
#   * use imatrix_mse observer to weight quantization error by channel importance
recipe = [
    QuantizationModifier(
        scheme="NVFP4A16",
        targets="Linear",
        ignore=["lm_head"],
        weight_observer="imatrix_mse",
    ),
]

torch.accelerator.reset_peak_memory_stats()
start_time = time.time()

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    preprocessing_num_workers=32,
    dataloader_num_workers=32,
)

elapsed_time = time.time() - start_time
peak_memory_gb = torch.accelerator.max_memory_allocated() / (1024**3)
print("Quantization Complete")
print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
device = torch.device("cuda", 0)
mem = torch.accelerator.get_memory_info(0)[1]
dispatch_model(model, device_memory={device: mem})
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
world_size = torch.distributed.get_world_size()
SAVE_DIR = (
    MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16-imatrix-DDP" + str(world_size)
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
