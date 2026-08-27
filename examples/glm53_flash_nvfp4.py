#############################################################################
# Mixed-precision (NVFP4 + FP8_BLOCK) quantization of GLM-5.3-Flash.
#
# Run across 4 GPUs (one rank per GPU) with:
#     torchrun --nproc-per-node 4 examples/glm53_flash_nvfp4-fp8.py
#
# Scheme:
#   * routed MoE experts           -> NVFP4
#   * shared experts / dense MLPs  -> FP8_BLOCK
#   * attention (KDA + MLA)        -> FP8_BLOCK
#   * vision tower / lm_head / MTP -> left in full precision (bf16)
#
# The DeepSeek-style token `indexer` is sensitive to quantization and is left
# in full precision. The MoE router (`mlp.gate`) is never quantized.
#############################################################################

import torch
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Load the model.
# GLM-5.3-Flash is a vision-language MoE model, so it must be loaded with its
# `Glm5NextForConditionalGeneration` class (not `AutoModelForCausalLM`) in order
# to keep the vision tower. `load_context` linearizes the packed 3D MoE experts
# into per-expert `nn.Linear` layers so they can be quantized.
init_dist()
MODEL_ID = "GLM-5.3-Flash-BF16"  #"zai-org/GLM-5.3-Flash"
with load_context(Glm5NextForConditionalGeneration):
    model = Glm5NextForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="/data/kylesayrs/hub/offload_folder-glm53-nvfp4-fp8",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# `load_context` linearizes the packed 3D experts into a `LinearExperts2D`
# module, whose bound `_apply_gate` reads GLM's `swiglu_limit` off `self`. The
# generic linearization does not carry this attribute over, so re-attach it to
# each linearized experts module so calibration forward passes succeed.
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D  # noqa: E402

swiglu_limit = model.config.get_text_config().swiglu_limit
for module in model.modules():
    if isinstance(module, LinearExperts2D):
        module.swiglu_limit = swiglu_limit

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
#   * attention, shared experts, and dense MLPs -> FP8_BLOCK
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
        r"re:.*self_attn\.indexer\..*",  # sensitive to quantization
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    recipe=recipe,
    batch_size=1,
    shuffle_calibration_samples=False,
)

# Save to disk compressed. MTP tensors (not built by transformers) are copied
# over automatically by the save utility.
model.generation_config.top_p = None
SAVE_DIR = "/data/kylesayrs/hub/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
