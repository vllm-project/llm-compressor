from compressed_tensors.entrypoints.convert import FP8Converter
from compressed_tensors.entrypoints.convert import convert_checkpoint

convert_checkpoint(
    "zai-org/GLM-5.3-Flash",
    "/data/kylesayrs/hub/zai-org/GLM-5.3-Flash-ct",
    FP8Converter.from_pretrained("zai-org/GLM-5.3-Flash"),
    max_workers=8,
)