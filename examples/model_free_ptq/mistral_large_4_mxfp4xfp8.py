import json
from pathlib import Path

from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, MXFP4xFP8

from llmcompressor import model_free_ptq

MODEL_ID = "mistralai/Mistral-Large-4-1T-A52B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4xFP8"


def write_vllm_mistral_quantization_metadata(save_dir: str | Path) -> None:
    """Align params.json quantization targets with vLLM's internal module paths."""
    save_dir = Path(save_dir)
    config = json.loads((save_dir / "config.json").read_text())
    quantization = config["quantization_config"]

    fp8_targets = []
    mxfp4_targets = []
    for root in (r"language_model\.model", r"model"):
        fp8_targets.extend(
            [
                rf"re:{root}\.layers\.\d+\.self_attn\.q_proj$",
                rf"re:{root}\.layers\.\d+\.self_attn\.o_proj$",
                rf"re:{root}\.layers\.\d+\.self_attn\.indexer\.wq_b$",
                rf"re:{root}\.layers\.\d+\.mlp\.shared_experts\.gate_up_proj$",
                rf"re:{root}\.layers\.\d+\.mlp\.shared_experts\.down_proj$",
            ]
        )
        mxfp4_targets.extend(
            [
                rf"re:{root}\.layers\.\d+\.mlp\.gate_up_proj$",
                rf"re:{root}\.layers\.\d+\.mlp\.down_proj$",
            ]
        )
    fp8_targets.extend([r"re:model\.joint_proj$", r"re:model\.target_projection$"])
    mxfp4_targets.append(
        r"re:language_model\.model\.layers\.\d+\.mlp\.experts\.\d+\."
        r"(?:gate_proj|up_proj|down_proj)$"
    )

    quantization["config_groups"]["fp8_block"]["targets"] = fp8_targets
    quantization["config_groups"]["mxfp4x_fp8_mlp"]["targets"] = mxfp4_targets
    params_path = save_dir / "params.json"
    params = json.loads(params_path.read_text())
    params["quantization"] = quantization
    params_path.write_text(json.dumps(params, indent=2) + "\n")


# Preserve the router, multimodal components, and native unquantized attention
# projections. Keep the DSpark embedding and output head in their source dtype.
ignore = [
    "tok_embeddings",
    "output",
    "re:vision_encoder.*",
    "re:patch_merger.*",
    "re:vision_language_adapter.*",
    r"re:.*\.gate$",
    r"re:.*\.wkv_b$",
    r"re:.*\.wkv_a_with_mqa$",
    r"re:.*\.indexer\.wk$",
    r"re:.*\.indexer\.weights_proj$",
    r"re:mtp_dspark\.autoregressive_head\.(embedding|output)$",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    config=QuantizationConfig(
        config_groups={
            "fp8_block": QuantizationScheme(
                **FP8_BLOCK,
                targets=[
                    r"re:layers\.\d+\.attention\.(?:wq|wo)$",
                    r"re:layers\.\d+\.attention\.indexer\.wq_b$",
                    r"re:layers\.\d+\.shared_experts\.w[123]$",
                    r"re:mtp_dspark\.backbone\.layers\.\d+\.attention\.(?:wq|wo)$",
                    r"re:mtp_dspark\.autoregressive_head\.joint_proj$",
                    r"re:mtp_dspark\.backbone\.target_projection$",
                ],
            ),
            "mxfp4x_fp8_mlp": QuantizationScheme(
                **MXFP4xFP8,
                targets=[
                    r"re:layers\.\d+\.experts\.\d+\.w[123]$",
                    r"re:layers\.\d+\.feed_forward\.w[123]$",
                    r"re:mtp_dspark\.backbone\.layers\.\d+\.feed_forward\.w[123]$",
                ],
            ),
        },
        ignore=ignore,
    ),
    converter=FP8BlockDequantizer.from_pretrained(
        MODEL_ID, ignore=ignore, scale_name="qscale_weight"
    ),
    device="cuda:0",
    max_workers=1,
)
write_vllm_mistral_quantization_metadata(SAVE_DIR)
