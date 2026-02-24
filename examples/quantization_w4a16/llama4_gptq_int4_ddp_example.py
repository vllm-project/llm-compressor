###############################################################################
# This script quantizes Llama4 with GPTQ + INT4 using DDP.
# run this with `torchrun --nproc_per_node=4 llama4_gptq_int4_ddp_example.py`
# or change nproc_per_node to your desired configuration
###############################################################################

import time

import torch
from compressed_tensors.offload import init_dist, load_offloaded_model
from datasets import load_dataset
from transformers import Llama4ForConditionalGeneration, Llama4Processor

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import GPTQModifier

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

###### DDP MODEL LOAD CHANGE #####
init_dist()
with load_offloaded_model():
    model = Llama4ForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto_offload"
    )
##################################

processor = Llama4Processor.from_pretrained(model_id)

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 8192

###### DDP DATA LOAD CHANGE #####
ds = load_dataset(
    DATASET_ID, name="LLM", split=get_rank_partition("train", NUM_CALIBRATION_SAMPLES)
)
##################################


def preprocess_function(example):
    messgages = []
    for message in example["messages"]:
        messgages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messgages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


# Recipe: GPTQ + INT4
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=[
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        "re:.*vision_model.*",
        "re:.*multi_modal_projector.*",
        "Llama4TextAttention",
    ],
)

torch.cuda.reset_peak_memory_stats()
start_time = time.time()

# Apply algorithms.
# due to the large size of Llama4, we specify sequential targets such that
# only one MLP is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    sequential_targets=["Llama4TextMLP"],
)

elapsed_time = time.time() - start_time
peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
print("Quantization Complete")
print(f"Time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")
print(f"Peak GPU Memory: {peak_memory_gb:.2f} GB")

print("Saving...")
# Save to disk compressed.
SAVE_DIR = (
    model_id.rstrip("/").split("/")[-1]
    + "-GPTQ-W4A16-G128-DDP"
    + str(torch.distributed.get_world_size())
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
