"""
Quantize DiffusionGemma4 model experts to NVFP4.

This script quantizes ONLY the expert layers (90.44% of the model parameters)
to NVFP4 (4-bit weights, 4-bit activations), while keeping all other layers
(attention, router, embeddings, etc.) at full precision.

Model: gg-hf-st/test-checkpoint-26B-v2
- Total parameters: ~25.3B
- Expert parameters: 22.8B (90.44%)
- Non-expert parameters: 2.4B (9.56%)
"""

import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoProcessor
from transformers.models.diffusion_gemma4 import DiffusionGemma4ModelForBlockDiffusion

from llmcompressor import oneshot
from llmcompressor.modeling.diffusion_gemma4 import (  # noqa: F401
    CalibrationDiffusionGemma4TextExperts,
)
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model
MODEL_ID = "gg-hf-st/test-checkpoint-26B-v2"
model = DiffusionGemma4ModelForBlockDiffusion.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# The import of CalibrationDiffusionGemma4TextExperts is crucial for proper MoE
# calibration. This custom module automatically replaces the original
# DiffusionGemma4TextExperts class during calibration to:
# 1. Linearize the 3D expert tensors into individual nn.Linear modules
# 2. Ensure all experts are properly calibrated, even those not activated
#    for certain tokens during calibration

# Configure the quantization scheme
# NVFP4 (4-bit weights, 4-bit activations) for both encoder and decoder experts
recipe = QuantizationModifier(
    targets=["re:model\\.encoder\\..*\\.experts\\..*\\.gate_proj$",
             "re:model\\.encoder\\..*\\.experts\\..*\\.up_proj$",
             "re:model\\.encoder\\..*\\.experts\\..*\\.down_proj$",
             "re:model\\.decoder\\.layers\\..*\\.experts\\..*\\.gate_proj$",
             "re:model\\.decoder\\.layers\\..*\\.experts\\..*\\.up_proj$",
             "re:model\\.decoder\\.layers\\..*\\.experts\\..*\\.down_proj$"],
    scheme="NVFP4",
)
# Load calibration dataset
DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }

# Apply quantization with calibration data
# CRITICAL: Must specify sequential_targets to include BOTH encoder and decoder
# Otherwise sequential pipeline only processes decoder layers by default
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    sequential_targets=[
        "DiffusionGemma4DecoderTextLayer",  # Decoder layers (30 layers) - CORRECT name
        "DiffusionGemma4EncoderTextLayer",  # Encoder layers (30 layers) - CORRECT name
    ],
)

# Test sample generation
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)

# "The reason the sky is blue is because" + chat template
input_ids = torch.tensor(
    [[
        2, 105, 2364, 107, 818, 3282, 506, 7217, 563, 3730, 563,
        1547, 106, 107, 105, 4368, 107
    ]]
).to(model.device)

output = model.generate(
    input_ids,
    max_new_tokens=100,
    max_denoising_steps=48,
)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format
SAVE_DIR = "/raid/engine/dsikka/" + MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-Debug"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)