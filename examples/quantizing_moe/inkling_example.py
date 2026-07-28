import torch
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoTokenizer, InklingForConditionalGeneration

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Load the model
init_dist()
# if torch.distributed.get_rank() == 0:
#     torch.cuda.memory._record_memory_history(max_entries=10000000)
# model_id = "thinkingmachines/Inkling"
model_id = "inference-optimization/Inkling-0.6B-A0.6B"
with load_context(InklingForConditionalGeneration):
    model = InklingForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={},
        offload_folder="/mnt/nvme-data/engine/kylesayrs/offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 4  # 512
MAX_SEQUENCE_LENGTH = 1024  # 2048

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

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[r"re:.*attn\..*"],
            **FP8_BLOCK,
        ),
        "mlp": QuantizationScheme(
            targets=[r"re:.*mlp\..*"],
            **NVFP4,
        ),
    },
    ignore=[
        r"re:.*sconv$",
        r"re:.*mlp\.gate$",  # technically not necessary `InklingTopkRouter`
        r"re:.*shared_experts.*",
        r"re:audio_tower.*",
        r"re:vision_tower.*",
    ],
)

try:
    # Apply algorithms.
    num_experts = getattr(model.config, "n_routed_experts", 256)
    oneshot(
        model=model,
        dataset=ds,
        batch_size=1,
        recipe=recipe,
        shuffle_calibration_samples=False,
        # sequential_targets=["InklingAttention", "ExpertMLP"],
        # sequential_targets_per_subgraph=(num_experts // 4 + 10),
    )
finally:
    # if torch.distributed.get_rank() == 0:
    #     torch.cuda.memory._dump_snapshot("inkling_memory.pickle")
    pass

# Save to disk compressed.
# Note: base checkpoint generation_config needs fixing for newer transformers versions
model.generation_config.top_p = None
SAVE_DIR = (
    "/mnt/nvme-data/engine/kylesayrs/"
    + model_id.rstrip("/").split("/")[-1]
    + "-NVFP4-FP8"
)
model.save_pretrained(SAVE_DIR, save_compressed=True, save_original_format=False)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
