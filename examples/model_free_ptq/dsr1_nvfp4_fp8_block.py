from llmcompressor import model_free_ptq
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

MODEL_ID = "nvidia/DeepSeek-R1-NVFP4"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

# Apply FP8-Block to the model's compatible self_attn Linear layers
# Once quantized, the model is saved to SAVE_DIR.
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
        targets=[
            # NOTE: self_attn.kv_a_proj_with_mqa has shape 576x7168, incompatible with block size 128x128
            # NOTE: self_attn.kv_b_proj is already dequantized by MLA
            # Target the remaining self_attn layers:
            #   - self_attn.o_proj
            #   - self_attn.q_a_proj
            #   - self_attn.q_b_proj
            "re:.*self_attn.(o_proj|q_a_proj|q_b_proj).*"
        ],
    ),
    max_workers=8,
    device="cuda:0",
)


def merge_configs():
    """ "
    Merge modelopt config with CT quantization_config in saved config.json
    and remove hf_quant_config
    """
    import json
    import os

    from compressed_tensors.quantization.quant_scheme import NVFP4

    with open(os.path.join(SAVE_DIR, "config.json")) as f:
        config = json.load(f)

    quant_config = QuantizationConfig.model_validate(config["quantization_config"])

    num_groups = len(quant_config.config_groups)

    quant_config.config_groups[f"config_group_{num_groups}"] = QuantizationScheme(
        targets=["re:.*mlp.*\.(gate|up|down)_proj$"], **NVFP4
    )
    quant_config.format = "mixed-precision"

    config["quantization_config"] = quant_config.model_dump()

    # TODO overwrite config.json instead
    with open(os.path.join(SAVE_DIR, "new_config.json"), "w") as f2:
        json.dump(config, f2, indent=4)

    hf_quant_config_path = os.path.join(SAVE_DIR, "hf_quant_config.json")
    if os.path.exists(hf_quant_config_path):
        os.remove(hf_quant_config_path)


merge_configs()
