from compressed_tensors.entrypoints.convert import FP8Converter
from compressed_tensors.entrypoints.convert import convert_checkpoint

convert_checkpoint(
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "/data/kylesayrs/hub/DeepSeek-V4-Flash-0731-ct",
    FP8Converter.from_pretrained("deepseek-ai/DeepSeek-V4-Flash-0731"),
    max_workers=8,
)