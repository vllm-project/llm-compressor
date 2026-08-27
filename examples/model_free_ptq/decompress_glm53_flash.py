# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from compressed_tensors.entrypoints.convert import (
    convert_checkpoint,
    FP8BlockDequantizer,
)

# deepseek-ai/DeepSeek-V3.2 checkpoint has layers that are quantized in the FP8
# quant method's FP8_BLOCK scheme. This script will upconvert to bfloat16 so that
# the model can be compressed in another configuration.
MODEL_ID = "zai-org/GLM-5.3-Flash"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-BF16"

# Convert DeepSeek-V3.2 back to dense bfloat16 format
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=FP8BlockDequantizer.from_pretrained(MODEL_ID),
    max_workers=4,
)