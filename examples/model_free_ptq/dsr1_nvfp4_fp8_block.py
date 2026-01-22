from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
)
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, NVFP4

from llmcompressor import model_free_ptq

MODEL_ID = "nvidia/DeepSeek-R1-NVFP4"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"


# Apply FP8-Block to the model's compatible self_attn Linear layers
# Once quantized, the model is saved to SAVE_DIR.
def run_model_free_ptq():
    model_free_ptq(
        model_stub=MODEL_ID,
        save_directory=SAVE_DIR,
        scheme=QuantizationScheme(
            **FP8_BLOCK,
            targets=[
                # NOTE: skipping self_attn.kv_a_proj_with_mqa
                #  shape 576x7168 is incompatible with block size 128x128
                # NOTE: skipping self_attn.q_a_proj
                #  fused with kv_a_proj_with_mqa, so much have the same quant config
                # NOTE: skipping self_attn.kv_b_proj
                #  already dequantized by MLA
                # Target the remaining self_attn layers:
                #   - self_attn.o_proj
                #   - self_attn.q_b_proj
                "re:.*self_attn.(o_proj|q_b_proj).*"
            ],
        ),
        ignore=["re:.*self_attn.(kv_a_proj_with_mqa|kv_b_proj|q_a_proj)$"],
        max_workers=32,
        device="cuda:0",
    )


def merge_configs():
    """ "
    Merge modelopt config with CT quantization_config in saved config.json
    and remove hf_quant_config
    """
    import json
    import os

    with open(os.path.join(SAVE_DIR, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(SAVE_DIR, "old_config.json"), "w") as f2:
        json.dump(config, f2, indent=4)

    quant_config = QuantizationConfig.model_validate(config["quantization_config"])

    num_groups = len(quant_config.config_groups)

    quant_config.config_groups[f"config_group_{num_groups}"] = QuantizationScheme(
        **NVFP4,
        # NOTE: gate_up_proj also needed, when gate/up are fused
        targets=["re:.*mlp.*\.(gate_up|gate|up|down)_proj$"],
        format=CompressionFormat.nvfp4_pack_quantized.value,
    )
    quant_config.format = CompressionFormat.mixed_precision.value

    config["quantization_config"] = quant_config.model_dump()

    with open(os.path.join(SAVE_DIR, "config.json"), "w") as f2:
        json.dump(config, f2, indent=4)

    hf_quant_config_path = os.path.join(SAVE_DIR, "hf_quant_config.json")
    if os.path.exists(hf_quant_config_path):
        os.remove(hf_quant_config_path)


if __name__ == "__main__":
    run_model_free_ptq()
    merge_configs()
