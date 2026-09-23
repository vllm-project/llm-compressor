"""Data-free fallback when Transformers cannot load a checkpoint's MTP layers.

Requires FP8Converter from compressed-tensors PR #902:
https://github.com/vllm-project/compressed-tensors/pull/902
FP8Converter normalizes the source FP8/packed-FP4 checkpoint to compressed-tensors.
Model-free PTQ then dequantizes and requantizes eligible 2D text-layer weights,
including MTP layer 45, to FP8_DYNAMIC. This processes the full checkpoint in
two passes; it is not an MTP-only or Transformers MtpModel path.
"""

import os
from tempfile import TemporaryDirectory

from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    convert_checkpoint,
)
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor import model_free_ptq

MODEL_ID = "zai-org/GLM-5.3-Flash"
SAVE_DIR = os.environ.get("MTP_OUTPUT_DIR", "GLM-5.3-Flash-FP8-Dynamic-MTP")

try:
    from compressed_tensors.entrypoints.convert import FP8Converter
except ImportError as error:
    raise ImportError("This example requires compressed-tensors PR #902") from error

with TemporaryDirectory(prefix="mtp-fp8-", dir=os.environ.get("MTP_TMPDIR")) as ct_dir:
    convert_checkpoint(
        model_stub=MODEL_ID,
        save_directory=ct_dir,
        converter=FP8Converter.from_pretrained(MODEL_ID),
        max_workers=8,
    )
    model_free_ptq(
        model_stub=ct_dir,
        save_directory=SAVE_DIR,
        scheme=preset_name_to_scheme(
            "FP8_DYNAMIC", targets=[r"re:^model\.language_model\.layers\."]
        ),
        converter=CompressedTensorsDequantizer(ct_dir),
        ignore=[r"re:.*_conv1d$"],
        max_workers=8,
    )
