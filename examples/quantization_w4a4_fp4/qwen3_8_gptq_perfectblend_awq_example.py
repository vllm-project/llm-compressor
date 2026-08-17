import torch
from compressed_tensors.quantization.quant_scheme import (
    FP8_DYNAMIC,
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
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
    )
]

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(
    "mlabonne/open-perfectblend",
    split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.shuffle(seed=42)

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def preprocess_function(example):
    messages = [
        {
            "role": ROLE_MAP.get(msg["from"], msg["from"]),
            "content": [{"type": "text", "text": msg["value"]}],
        }
        for msg in example["conversations"]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": MAX_SEQUENCE_LENGTH,
            "add_special_tokens": False,
        },
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-GPTQ-perfectblend-AWQ"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
