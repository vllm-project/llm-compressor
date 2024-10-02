import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"

# adjust based off number of desired GPUs
# reserve_for_hessians=True reserves memory which is required by
# GPTQModifier and SparseGPTModifier
device_map = calculate_offload_device_map(
    MODEL_ID, num_gpus=2, reserve_for_hessians=True, torch_dtype=torch.bfloat16
)

model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


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


ds = ds.map(tokenize, remove_columns=ds.column_names)

# define a llmcompressor recipe for W8A8 quantization
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(
        targets="Linear", scheme="W8A8", ignore=["lm_head"],
    ),
]

SAVE_DIR = MODEL_ID.split("/")[1] + "-INT8"

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
    output_dir=SAVE_DIR,
)
