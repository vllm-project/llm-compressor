from transformers import (
    AutoProcessor,
    InklingForConditionalGeneration,
    InklingForCausalLM,
    AutoModelForCausalLM,
)
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MODEL_ID = "thinkingmachines/Inkling"
# MODEL_ID = "inference-optimization/Inkling-0.6B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-FP8-BLOCK"

# Select model and load it in the `load_context` context
with load_context(InklingForConditionalGeneration):
    model = InklingForConditionalGeneration.from_pretrained(
        MODEL_ID,
        max_memory={"cpu": 500e9},
        device_map="auto_offload",  # fit as much as possible on cpu, rest goes on disk
        offload_folder="./offload_folder",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.eos_token = "<|endoftext|>"
    processor.tokenizer.pad_token = "<|endoftext|>"

# Configure the quantization algorithm to run.
#   * quantize the weights to NVFP4
recipe = QuantizationModifier(
    config_groups={
        "config_group_0": QuantizationScheme(
            targets=[
                # r"re:.*layers.\d+.*attn.\w+",
                r"re:.*self_attn\.(q|k|v|o|r)_proj$",
            ],
            **FP8_BLOCK,
        ),
        "config_group_1": QuantizationScheme(
            targets=[
                # r"re:.*layers.\d+.*mlp\.\w+",
                r"re:.*mlp.*(gate|up|down)_proj$",
            ],
            **NVFP4,
        ),
    },
    ignore=[
        "lm_head",
        "model.llm.unembed",
        "model.llm.embed",
        "re:.*sconv.*",
        "re:.*norm.*",
        "re:.*bias$",
        "re:.*gate$",
        "re:.*global_scale$",
        "re:.*shared_experts.*",
        "re:.*visual.*",
        "re:.*vision.*",
        "re:.*audio.*",
        "re:model.mtp.*",
    ],
)

# Select calibration dataset.
DATASET_ID = "ultrachat-200k"
DATASET_SPLIT = "train_sft"

NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Apply algorithms.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset=DATASET_ID,
    splits={"calibration": f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"},
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True, save_original_format=False)
processor.save_pretrained(SAVE_DIR)
