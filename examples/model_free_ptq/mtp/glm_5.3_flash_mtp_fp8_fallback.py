"""Convert the checkpoint, then quantize text and MTP weights to FP8_DYNAMIC.

Requires https://github.com/vllm-project/compressed-tensors/pull/902.
"""

from compressed_tensors.entrypoints.convert import (
    CompressedTensorsDequantizer,
    FP8Converter,
    convert_checkpoint,
)
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor import model_free_ptq

MODEL_ID = "zai-org/GLM-5.3-Flash"
CONVERTED_DIR = "GLM-5.3-Flash-CT"
SAVE_DIR = "GLM-5.3-Flash-FP8-Dynamic-MTP"

fp8_converter = FP8Converter.from_pretrained(MODEL_ID)
convert_checkpoint(
    model_stub=MODEL_ID,
    save_directory=CONVERTED_DIR,
    converter=fp8_converter,
    max_workers=8,
)
# Checkpoint paths include language_model; source exclusion paths omit it.
ignore = [
    pattern.replace(r"model\.layers\.", r"model\.language_model\.layers\.")
    for pattern in fp8_converter.ignore
] + [r"re:.*_conv1d$"]
model_free_ptq(
    model_stub=CONVERTED_DIR,
    save_directory=SAVE_DIR,
    scheme=preset_name_to_scheme(
        "FP8_DYNAMIC", targets=[r"re:^model\.language_model\.layers\."]
    ),
    converter=CompressedTensorsDequantizer(CONVERTED_DIR),
    ignore=ignore,
    max_workers=8,
)
