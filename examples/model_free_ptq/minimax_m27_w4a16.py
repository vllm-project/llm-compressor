from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from compressed_tensors.quantization import (
    QuantizationScheme,
)
from compressed_tensors.quantization.quant_scheme import W4A16

from llmcompressor import model_free_ptq

MODEL_ID = "MiniMaxAI/MiniMax-M2.7"
SAVE_DIR = "MiniMax-M2.7-W4A16"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme=QuantizationScheme(
        **W4A16,
        targets=[
            r"re:model.layers.\d+.self_attn.(q|k|v|o)_proj$",
            r"re:model.layers.\d+.block_sparse_moe.experts.\d+.w[1-3]$",
            # NOTE: required when loading in vllm
            "re:.*(gate_up|gate|up|down)_proj$",
        ],
    ),
    # Pre-process: dequantize original checkpoint's FP8_BLOCK layers
    converter=FP8BlockDequantizer(
        targets=[
            r"re:model.layers.\d+.self_attn.(q|k|v|o)_proj$",
            r"re:model.layers.\d+.block_sparse_moe.experts.\d+.w[1-3]$",
        ]
    ),
    max_workers=8,
)
