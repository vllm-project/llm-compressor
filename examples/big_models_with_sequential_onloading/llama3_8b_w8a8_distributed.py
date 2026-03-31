#############################################################################
# Distributed W8A8 quantization example with activation observer sync.
# run this with `torchrun --nproc_per_node=2 llama3_8b_w8a8_distributed.py`
# or change nproc_per_node to your desired configuration
#############################################################################

import torch
from compressed_tensors.offload import dispatch_model, init_dist, load_offloaded_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

###### DDP MODEL LOAD CHANGE #####
init_dist()
with load_offloaded_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto_offload"
    )
##################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

###### DDP DATA LOAD CHANGE #####
ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
##################################

ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# QuantizationModifier automatically detects torch.distributed and
# all-reduces activation observer statistics at layer boundaries
recipe = [
    QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
sample = tokenizer("Hello my name is", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

print("Saving...")
SAVE_DIR = (
    MODEL_ID.rstrip("/").split("/")[-1]
    + "-W8A8-DDP"
    + str(torch.distributed.get_world_size())
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
