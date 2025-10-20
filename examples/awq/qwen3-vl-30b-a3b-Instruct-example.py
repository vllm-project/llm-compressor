import torch
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration
from transformers.models.qwen3_vl_moe import Qwen3VLMoeVisionModel

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modeling.patches import fast_pos_embed_interpolate
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

# NOTE: Requires a minimum of transformers 4.57.0

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Load model.
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# patch required for transformers<=4.57.1 to support offloading
Qwen3VLMoeVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate
model = replace_modules_for_calibration(model)

DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


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


# Configure AWQ quantization with smoothing and balancing
recipe = AWQModifier(
    ignore=[
        "re:.*embed_tokens",
        "re:.*input_layernorm$",
        "re:.*mlp[.]gate$",
        "re:.*post_attention_layernorm$",
        "re:.*norm$",
        "re:model[.]visual.*",
        "re:visual.*",
        "lm_head",
    ],
    mappings=[
        {
            "smooth_layer": "re:.*input_layernorm$",
            "balance_layers": ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
        },
        {"smooth_layer": "re:.*v_proj$", "balance_layers": ["re:.*o_proj$"]},
        {
            "smooth_layer": "re:.*post_attention_layernorm$",
            "balance_layers": ["re:.*gate_proj$", "re:.*up_proj$"],
        },
        {"smooth_layer": "re:.*up_proj$", "balance_layers": ["re:.*down_proj$"]},
    ],
    duo_scaling=True,
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "group_size": 32,
                "strategy": "group",
                "block_structure": None,
                "dynamic": False,
                "actorder": None,
                "observer": "mse",
                "observer_kwargs": {},
            },
            "input_activations": None,
            "output_activations": None,
            "format": None,
        }
    },
)

# Apply AWQ quantization.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset=DATASET_ID,
    splits=DATASET_SPLIT,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-AWQ-W8A16-mse-seq"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
