"""
Smoke test for issue #2699: save linearized Qwen3-VL-MoE as native 3D packed experts.

Usage:
    python examples/quantizing_moe/qwen3_vl_moe_3d_save.py

Optional HF stub (when available):
    MODEL_ID=inference-optimization/Qwen3-VL-1.0B-A0.4B-Instruct \\
        python examples/quantizing_moe/qwen3_vl_moe_3d_save.py
"""

import os
from pathlib import Path

import torch
from compressed_tensors.quantization import (
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)
from compressed_tensors.quantization.quant_args import QuantizationArgs
from safetensors import safe_open
from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

from llmcompressor.modeling.moe.linearize import linearize_moe
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)
from llmcompressor.utils import load_context
from llmcompressor.utils.dev import skip_weights_initialize


def _build_tiny_model():
    config = Qwen3VLMoeConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 256,
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
        },
    )
    with skip_weights_initialize():
        return Qwen3VLMoeForConditionalGeneration(config)


def _load_hf_model(model_id: str):
    with load_context(Qwen3VLMoeForConditionalGeneration):
        return Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )


def _expert_keys(save_dir: Path) -> list[str]:
    keys = []
    for path in save_dir.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            keys.extend(k for k in handle.keys() if "mlp.experts" in k)
    return sorted(keys)


def main():
    model_id = os.environ.get("MODEL_ID")
    out_dir = Path(os.environ.get("OUT_DIR", "/tmp/qwen3_vl_moe_3d_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_id:
        print(f"Loading {model_id} with load_context (linearizes MoE)...")
        model = _load_hf_model(model_id)
    else:
        print("No MODEL_ID set; building a tiny random Qwen3-VL-MoE...")
        model = _build_tiny_model()
        linearize_moe(model)

    # Attach scales so save mappings also cover qparams (channel-wise for this demo)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=8, type="float", strategy="channel"),
    )
    qconfig = QuantizationConfig(
        config_groups={"group_0": scheme},
        quantization_status=QuantizationStatus.FROZEN,
    )
    apply_quantization_config(model, qconfig)
    print("Applied quantization config (channel FP8 scales on Linear)")

    modify_save_pretrained(model)
    print(f"Saving to {out_dir} ...")
    model.save_pretrained(out_dir, safe_serialization=True)

    keys = _expert_keys(out_dir)
    packed = [k for k in keys if k.endswith(("gate_up_proj", "down_proj"))]
    linearized = [k for k in keys if ".experts.0." in k]

    print(f"\nExpert keys ({len(keys)}):")
    for k in keys[:20]:
        print(f"  {k}")
    if len(keys) > 20:
        print(f"  ... ({len(keys) - 20} more)")

    print(f"\nPacked 3D weight keys: {len(packed)}")
    print(f"Linearized expert.0 keys remaining: {len(linearized)}")

    if packed and not linearized:
        print("\nSUCCESS: checkpoint experts are 3D packed")
    else:
        print("\nFAILURE: expected packed gate_up_proj/down_proj without experts.N.*")
        raise SystemExit(1)

    print("\nOptional: reload with Transformers (no llmcompressor linearize)...")
    reloaded = Qwen3VLMoeForConditionalGeneration.from_pretrained(out_dir)
    experts = reloaded.model.language_model.layers[0].mlp.experts
    assert hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")
    print(f"Reload OK: {type(experts).__name__} has fused 3D params")


if __name__ == "__main__":
    main()
