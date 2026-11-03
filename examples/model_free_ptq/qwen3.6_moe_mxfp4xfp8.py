from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, MXFP4xFP8

from llmcompressor import model_free_ptq

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4xFP8"

# Apply MXFP4xFP8 to MoE weights
# Use FP8_BLOCK dense layers
# Save results to SAVE_DIR

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    config=QuantizationConfig(
        config_groups={
            "dense": QuantizationScheme(
                **FP8_BLOCK,
                targets=[
                    r"re:.*self_attn\.(q|k|v|o)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                    r"re:.*mlp\.shared_expert\.(gate|up|down)_proj$",
                ],
            ),
            "experts": QuantizationScheme(
                **MXFP4xFP8,
                targets=[
                    # Matches the per-expert names produced after the split
                    r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",
                ],
            ),
        }
    ),
    device="cuda:0",
)
