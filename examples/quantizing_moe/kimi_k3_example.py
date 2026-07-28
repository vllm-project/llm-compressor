import torch
from compressed_tensors.quantization.quant_scheme import (
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoTokenizer

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modeling.kimi_k3 import KimiK3ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Select model and load it.
MODEL_ID = "inference-optimization/Kimi-K3-0.40B-MXFP4"
#MODEL_ID = "moonshotai/Kimi-K3"

# mlp_res_proj
# self_attention_res_proj
# block_sparse_moe.routed_expert_down_proj

# init_dist()
with load_context(KimiK3ForConditionalGeneration):
    model = KimiK3ForConditionalGeneration.from_pretrained(  # KimiK3ForConditionalGeneration
        MODEL_ID,
        device_map="auto",
        max_memory={},
        offload_folder="/mnt/nvme-data/engine/kylesayrs/offload_folder",
        trust_remote_code=True,
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

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

recipe = QuantizationModifier(
    config_groups={
        "experts": QuantizationScheme(
            targets=[r"re:.*block_sparse_moe.*"],
            **NVFP4,
        ),
    },
    ignore=["lm_head", r"re:.*block_sparse_moe\.gate.*"],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    recipe=recipe,
    batch_size=4,
    shuffle_calibration_samples=False,
    sequential_targets=["KimiMLAAttention", "KimiBlockSparseMLP"],
    #sequential_targets_per_subgraph=(896 // 3),
)

# Save to disk compressed.
SAVE_DIR = "/mnt/nvme-data/engine/kylesayrs/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True, save_original_format=False)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
