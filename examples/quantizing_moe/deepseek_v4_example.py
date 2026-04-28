from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
MODEL_ID = "mlx-community/DeepSeek-V4-Flash-bf16"

config = AutoConfig.from_pretrained(MODEL_ID)
del config.quantization_config
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    config=config,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
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

# Configure the quantization algorithm to run.
#   * quantize mlp/expert weights to NVFP4
#   * quantize attention projection weights to FP8_BLOCK
recipe = QuantizationModifier(
    config_groups={
        "group_moe": QuantizationScheme(
            targets=[
                r"re:model.*mlp.*(gate|up|down)_proj$",
            ],
            **NVFP4,
        ),
        "group_attn": QuantizationScheme(
            targets=[
                r"re:model.*self_attn\.(wq_a|wq_b|wkv|wo_a|wo_b)$",
                r"re:model.*self_attn\.compressor\.(wkv|wgate)$",
                r"re:model.*self_attn\.compressor\.indexer\.(wkv|wq_b|wgate)$",
            ],
            **FP8_BLOCK,
        ),
    },
    ignore=["lm_head"],
)

# Apply algorithms.
# due to the large size of DeepSeek-V4, we specify sequential targets such that
# only one block is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["DeepseekV4Attention", "DeepseekV4MLP"],
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-BLOCK"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
