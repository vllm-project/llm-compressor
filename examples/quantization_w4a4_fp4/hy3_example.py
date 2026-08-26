# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
import torch
from compressed_tensors.distributed import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Select model and load it.
MODEL_ID = "tencent/Hy3"

# NOTE: `transformers==5.14` breaks saving for disk-offloaded models.
# Please install `transformers>=5.15` or install from source
init_dist()
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    config_groups={
        "attention": QuantizationScheme(
            targets=[r"re:.*self_attn\..*"],
            **FP8_BLOCK,
        ),
        "experts": QuantizationScheme(
            targets=[r"re:.*mlp.*"],
            **NVFP4,
        ),
    },
    ignore=["lm_head"],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=tokenizer,
    dataset="perfectblend",
    recipe=recipe,
    batch_size=4,
    num_calibration_samples=512,
    shuffle_calibration_samples=False,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
