import json
from pathlib import Path

import torch
from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_BLOCK, MXFP4xFP8

from llmcompressor import model_free_ptq

MODEL_ID = "mistralai/Mistral-Large-4-1T-A52B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-MXFP4xFP8"

ignore = [
    "tok_embeddings",
    "output",
    r"re:vision_encoder.*",
    r"re:patch_merger.*",
    r"re:vision_language_adapter.*",
    r"re:.*\.gate$",
    r"re:.*\.wkv_b$",
    r"re:.*\.wkv_a_with_mqa$",
    r"re:.*\.indexer\.wk$",
    r"re:.*\.indexer\.weights_proj$",
    r"re:mtp_dspark\.autoregressive_head\.(embedding|output)$",
]

devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
if not devices:
    raise RuntimeError("Mistral Large 4 quantization requires a CUDA device")

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    config=QuantizationConfig(
        config_groups={
            "dense": QuantizationScheme(
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
            # Routed experts plus dense and MTP feed-forward projections.
            "experts": QuantizationScheme(
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
    device=devices,
    max_workers=len(devices),
)




# Write vLLM's Mistral-format quantization metadata after compression.
def write_vllm_mistral_quantization_metadata(save_dir: str | Path) -> None:
    """Copy quantization metadata to params.json using vLLM module targets.

    model_free_ptq writes the config using source checkpoint names. vLLM's
    native Mistral loader reads params.json, and regex targets are not remapped
    to its runtime module names automatically.
    """
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

    quantization["config_groups"]["dense"]["targets"] = fp8_targets
    quantization["config_groups"]["experts"]["targets"] = mxfp4_targets

    params_path = save_dir / "params.json"
    params = json.loads(params_path.read_text())
    params["quantization"] = quantization
    params_path.write_text(json.dumps(params, indent=2) + "\n")

write_vllm_mistral_quantization_metadata(SAVE_DIR)
