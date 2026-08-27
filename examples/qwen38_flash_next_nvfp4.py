#############################################################################
# Mixed-precision (NVFP4 + FP8_BLOCK) quantization of Qwen3.8-Flash-Next.
#
# Run across 4 GPUs (one rank per GPU) with:
#     torchrun --nproc-per-node 4 examples/qwen38_flash_next_nvfp4-fp8.py
#
# Scheme:
#   * routed MoE experts             -> NVFP4
#   * shared experts                 -> FP8_BLOCK
#   * attention (GatedDeltaNet + QSA) -> FP8_BLOCK
#   * vision tower / lm_head / MTP   -> left in full precision (bf16)
#
# The QSA `indexer` is sensitive to quantization and is left in full precision.
# The MoE router (`mlp.gate`) and `shared_expert_gate` are never quantized.
# Per-Layer-Embedding (PLE) and hyper-connection layers are left untouched.
#############################################################################

import torch
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen4ExpForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Load the model.
# Qwen3.8-Flash-Next is a vision-language MoE model, so it must be loaded with
# its `Qwen4ExpForConditionalGeneration` class (not `AutoModelForCausalLM`) in
# order to keep the vision tower. `load_context` linearizes the packed 3D MoE
# experts into per-expert `nn.Linear` layers so they can be quantized.
# init_dist()
MODEL_ID = "Qwen/Qwen3.8-Flash-Next"
with load_context(Qwen4ExpForConditionalGeneration):
    model = Qwen4ExpForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        # max_memory={},
        # offload_folder="/data/kylesayrs/hub/offload_folder-qwen38-flash-next-nvfp4-fp8",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess. Each rank calibrates on a disjoint partition;
# QuantizationModifier all-reduces observer statistics at layer boundaries.
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
#   * routed experts (post-linearization `mlp.experts.<i>.{gate,up,down}_proj`)
#     -> NVFP4
#   * attention and shared experts -> FP8_BLOCK
recipe = QuantizationModifier(
    config_groups={
        "experts": QuantizationScheme(
            targets=[r"re:.*mlp\.experts\..*(gate|up|down)_proj$"],
            **NVFP4,
        ),
    },
    ignore=[
        r"re:.*visual.*",  # vision tower stays full precision
        "lm_head",
        r"re:.*mlp\.gate$",  # MoE router
        r"re:.*shared_expert_gate$",  # shared-expert routing gate
        r"re:.*self_attn\.indexer\..*",  # sensitive to quantization
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    recipe=recipe,
    batch_size=8,
    shuffle_calibration_samples=False,
    propagate_error=False,
)

# Save to disk compressed. MTP tensors (not built by transformers) are copied
# over automatically by the save utility.
model.generation_config.top_p = None
SAVE_DIR = "/data/kylesayrs/hub/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
