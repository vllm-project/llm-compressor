# NOTE: to use a custom dataset with your own data, see examples/custom_dataset_example.py
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.pruning import REAPPruningModifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Select model and load it.
init_dist()
model_id = "inference-optimization/Qwen3.8-1.0B-A0.6B"  # Qwen/Qwen3.8-2.4T-A95B
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={},
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prune 25% of the experts in each MoE layer, based on saliency.
# You can adjust this value to prune more or less aggressively.
recipe = [
    REAPPruningModifier(sparsity=0.25),
    QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\..*",
                    r"re:.*linear_attn.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp\..*"],
                **NVFP4,
            ),
        },
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",
            "re:.*shared_expert_gate.*",
        ],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    pipeline="sequential",
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4-FP8-REAP-25"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
