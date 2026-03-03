from compressed_tensors.entrypoints.convert import (
    ModelOptNvfp4Converter,
)
from compressed_tensors.quantization import (
    QuantizationScheme,
)
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK

from llmcompressor import model_free_ptq

MODEL_ID = "nvidia/DeepSeek-R1-NVFP4"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"


# Convert modelopt NVFP4 format to compressed-tensors format and
# apply FP8-Block to the model's compatible self_attn Linear layers
# Once quantized, the model is saved to SAVE_DIR.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme=QuantizationScheme(
        **FP8_BLOCK,
        targets=[
            # Target fused layers, must have the same quant config
            # shape 576x7168 is compatible with block size 128x128
            #   - self_attn.kv_a_proj_with_mqa
            #   - self_attn.q_a_proj
            "re:.*self_attn.(kv_a_proj_with_mqa|q_a_proj)$",
            # Skip self_attn.kv_b_proj, already dequantized by MLA
            # Target remaining self_attn layers:
            #   - self_attn.o_proj
            #   - self_attn.q_b_proj
            "re:.*self_attn.(o_proj|q_b_proj).*",
        ],
    ),
    max_workers=8,
    device="cuda:0",
    converter=ModelOptNvfp4Converter(
        targets=[
            # nvidia/DeepSeek-R1-NVFP4's nvfp4-quantized layers, found by inspection
            # - model.layers.0.mlp.down_proj.weight
            # - model.layers.0.mlp.gate_proj.weight
            # - model.layers.0.mlp.up_proj.weight
            # - model.layers.3.mlp.shared_experts.down_proj.weight
            # - model.layers.3.mlp.shared_experts.gate_proj.weight
            # - model.layers.3.mlp.shared_experts.up_proj.weight
            # - model.layers.3.mlp.experts.0.down_proj.weight
            # - model.layers.3.mlp.experts.0.gate_proj.weight
            # - model.layers.3.mlp.experts.0.up_proj.weight
            # NOTE: gate_up_proj also needs to be targeted, gate/up are fused
            "re:.*mlp.*(gate_up|gate|up|down)_proj$"
        ]
    ),
)
