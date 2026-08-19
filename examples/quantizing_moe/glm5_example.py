# NOTE: to use a custom dataset with your own data, see examples/custom_dataset_example.py
import torch
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Load the model
# NOTE: `transformers==5.14` breaks saving for disk-offloaded models.
# Please install `transformers>=5.15` or install from source
init_dist()
model_id = "zai-org/GLM-5.2"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={"cpu": "500GiB"},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    config_groups={
        "attention_shared_experts": QuantizationScheme(
            targets=[r"re:.*self_attn\..*"],
            **FP8_BLOCK,
        ),
        "mlp": QuantizationScheme(
            targets=[r"re:.*mlp\..*"],
            **NVFP4,
        ),
    },
    ignore=[
        r"re:^model\.layers\.[0-2]\..*",
        r"re:.*mlp\.gate.*",
        r"re:.*indexer\.weights_proj$",  # sensitive to quantization
        r"lm_head",
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    batch_size=4,
    recipe=recipe,
    num_calibration_samples=512,
    shuffle_calibration_samples=False,
)

# Save to disk compressed.
# Note: base checkpoint generation_config needs fixing for newer transformers versions
model.generation_config.top_p = None
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
