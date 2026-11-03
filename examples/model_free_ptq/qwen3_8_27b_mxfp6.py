from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, MXFP6A8, MXFP8

from llmcompressor import model_free_ptq

MODEL_ID = "Qwen/Qwen3.8-27B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP6A8"

# Calibration-free mixed precision sized for a single 32 GB GPU:
#   * dense MLP: MXFP6 (e2m3) weights, dynamic MXFP8 activations
#   * attention, GDN and MTP projections: MXFP8
#   * lm_head: FP8 per-channel weights, dynamic per-token activations
# When groups overlap, the first matching group wins, so the MTP group comes
# before the MLP group.
config = QuantizationConfig(
    config_groups={
        "mtp": QuantizationScheme(targets=[r"re:.*mtp\.layers\..*"], **MXFP8),
        "mlp": QuantizationScheme(
            targets=[r"re:.*mlp\.(gate|up|down)_proj$"], **MXFP6A8
        ),
        "attn": QuantizationScheme(
            targets=[
                r"re:.*self_attn\.[qkvo]_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
            ],
            **MXFP8,
        ),
        "lm_head": QuantizationScheme(targets=[r"re:.*lm_head$"], **FP8_DYNAMIC),
    }
)

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    config=config,
    ignore=[
        r"re:.*visual\..*",
        r"re:.*in_proj_(a|b|ba)$",
        r"re:.*mtp\.fc$",
        r"re:.*embed_tokens$",
    ],
    device="cuda:0",
)
