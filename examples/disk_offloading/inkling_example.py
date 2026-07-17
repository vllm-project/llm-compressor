from transformers import AutoProcessor, InklingForConditionalGeneration
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "thinkingmachines/Inkling"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

# Select model and load it in the `load_context` context
with load_context(InklingForConditionalGeneration):
    model = InklingForConditionalGeneration.from_pretrained(
        MODEL_ID,
        max_memory={"cpu": 500e9},
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        offload_folder="./offload_folder",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4
recipe = QuantizationModifier(
    config_groups={
        "config_group_0": QuantizationScheme(
            targets=[
                r"re:model.*mlp.*(gate|up|down|gate_up)_proj$",
            ],
            **NVFP4,
        ),
        "config_group_1": QuantizationScheme(
            targets=[
                # NOTE: leaving weights_proj in bf16
                r"re:model.*self_attn.indexer.(wk|wq_b)$",
                r"re:model.*self_attn.kv_a_proj_with_mqa$",
                r"re:model.*self_attn.(kv_b|o|q_a|q_b)_proj$",
            ],
            **FP8_BLOCK,
        ),
    },
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "lm_head",
        "model.llm.unembed",
        "model.llm.embed",
        "re:.*sconv$",
        "re:.*norm.*",
        "re:.*bias$",
        "re:.*gate$",
        "re:.*global_scale$",
        "re:.*shared_experts.*",
        "re:.*visual.*",
        "re:.*vision.*",
        "re:.*audio.*",
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True, save_original_format=False)
processor.save_pretrained(SAVE_DIR)
