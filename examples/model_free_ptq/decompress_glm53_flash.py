# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This checkpoint has been uploaded as `RedHatAI/GLM-5.3-Flash-BF16`
from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    convert_checkpoint,
)

MODEL_ID = "zai-org/GLM-5.3-Flash"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-BF16"

# 1. dequantize ffn layers
# 2. dequantize attention layers
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=FP8BlockDequantizer(
        targets=[
            r"re:model\.language_model\.layers\.\d+\.mlp\..*(gate|up|down)_proj$",
            r"re:model\.language_model\.layers\.(3|7|11|15|19|23|27|31|35|39|43|45)\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$",  # noqa: E501
        ],
    ),
    max_workers=8,
)
