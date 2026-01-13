from llmcompressor import model_free_ptq
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

MODEL_ID = "nvidia/DeepSeek-R1-NVFP4"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

# Apply FP8-Block to the model's compatible self_attn Linear layers
# Once quantized, the model is saved
# using compressed-tensors to the SAVE_DIR.
model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme=QuantizationScheme(
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.BLOCK,
            symmetric=True,
            dynamic=False,
            block_structure=[128, 128],
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.GROUP,
            symmetric=True,
            dynamic=True,
            observer=None,
            group_size=128,
        ),
        # TODO cannot set targets here, must be ["Linear"]
        # targets=[
        #     "re:.*self_attn.(o_proj|q_a_proj|q_b_proj).*"
        # ],
    ),
    ignore=[
        # NOTE: self_attn.kv_a_proj_with_mqa has incompatible shape 576x7168 with block size 128x128
        # NOTE: self_attn.kv_b_proj is already dequantized by MLA
        # This regex matches all strings that don't contain one of the following substrings:
        #   - self_attn.o_proj
        #   - self_attn.q_a_proj
        #   - self_attn.q_b_proj
        "re:^(?!.*self_attn.(o_proj|q_a_proj|q_b_proj)).*$"
    ],
    max_workers=8,
    device="cuda:0",
)

# TODO reverse modelopt NVFP4 tensor packing order

# TODO merge hf_quant_config.json with CT quantization_config in config.json
