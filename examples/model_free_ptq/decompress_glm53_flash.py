# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This checkpoint has been uploaded as `RedHatAI/GLM-5.3-Flash-BF16`
from compressed_tensors.entrypoints.convert import (
    FP8BlockDequantizer,
    convert_checkpoint,
)

MODEL_ID = "zai-org/GLM-5.3-Flash"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-BF16"

convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    converter=FP8BlockDequantizer.from_pretrained(MODEL_ID),
    max_workers=4,
)
