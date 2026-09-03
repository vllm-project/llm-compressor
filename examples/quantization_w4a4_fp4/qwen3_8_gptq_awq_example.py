# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.quantization.quant_scheme import (
    FP8_DYNAMIC,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"

# Load model.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)


recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\.(q|k|v|o)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                    r"re:.*lm_head",
                    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\..*(gate|up|down)_proj$",
                ],
                **FP8_DYNAMIC,
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp\..*(gate|up|down)_proj$"],
                **NVFP4,
            ),
        },
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
        ],
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": False,
            "observer": "static_minmax",
        },
    ),
]

# Apply quantization.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset="perfectblend",
    max_seq_length=4096,
    num_calibration_samples=512,
    moe_calibrate_all_experts=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-GPTQ-AWQ"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
