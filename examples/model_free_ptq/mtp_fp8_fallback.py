"""Data-free fallback when Transformers cannot load a checkpoint's MTP layers.

Requires FP8Converter from compressed-tensors PR #902:
https://github.com/vllm-project/compressed-tensors/pull/902
This example converts the full FP8 checkpoint, then quantizes all 2D weights,
including MTP, to FP8_DYNAMIC. It does not require MTP architecture support.
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
        scheme=preset_name_to_scheme("FP8_DYNAMIC", targets=["Linear"]),
        converter=CompressedTensorsDequantizer(ct_dir),
        ignore=["lm_head"],
        max_workers=8,
    )
