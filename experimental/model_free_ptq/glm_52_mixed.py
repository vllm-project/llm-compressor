from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    MXFP4,
    QuantizationScheme,
)

from llmcompressor import model_free_ptq

MODEL_ID = "inference-optimization/GLM-5.2-0.8B-A0.8B"  # zai-org/GLM-5.2
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4xFP8_BLOCK"

# use `model_free_ptq` to apply quantization
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    config=QuantizationConfig(
        config_groups={
            "linears": QuantizationScheme(
                targets=[
                    r"re:^model\.layers\.[0-2]\..*",
                    r"re:.*self_attn.*",
                ],
                weights=FP8_BLOCK["weights"],
                input_activations=FP8_BLOCK["input_activations"],
            ),
            "moe": QuantizationScheme(
                targets=[r"re:^model\.layers\.(?:[3-9]|[1-6][0-9]|7[0-8])\..*mlp.*"],
                weights=MXFP4["weights"],
                input_activations=FP8_BLOCK["input_activations"],
            ),
        }
    ),
    ignore=[
        r"model.embed_tokens",
        r"re:.*weights_proj.*",  # sensitive to quantization
        r"re:.*mlp\.gate$",  # moe gate
        r"re:.*eh_proj.*",  # mtp sensitive to quantization
        r"lm_head",
    ],
    max_workers=75,
)
