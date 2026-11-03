from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, MXFP4xFP8

from llmcompressor import model_free_ptq

MODEL_ID = "meta-models/Muse-Spark-Open-816B-A42B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4xFP8"

# Apply MXFP4xFP8 to the routed experts
# Use FP8_BLOCK for attention (including the gate_proj output gate) and the
# dense projections around the expert block
# Keep the fp32 router and the multimodal towers unquantized
# Save results to SAVE_DIR

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    config=QuantizationConfig(
        config_groups={
            "dense": QuantizationScheme(
                **FP8_BLOCK,
                targets=[
                    # gate_proj here is the attention output gate
                    r"re:.*self_attn\.(q|k|v|o|gate)_proj$",
                    r"re:.*mlp\.(pre|post)_expert_proj$",
                ],
            ),
            "experts": QuantizationScheme(
                **MXFP4xFP8,
                # Matches the per-expert names produced after the split
                targets=[r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$"],
            ),
        },
        ignore=[
            "lm_head",
            r"re:.*embed_tokens$",
            r"re:.*mlp\.gate$",
            r"re:model\.(audio_tower|vision_tower|vision_adapter|vision_projection).*",
        ],
    ),
    device="cpu",
)
