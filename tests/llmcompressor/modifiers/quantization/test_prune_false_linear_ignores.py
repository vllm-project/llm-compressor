"""Drop GatedDeltaNet / RMSNormGated parents from reconstructed ignore lists."""

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationConfig,
    QuantizationScheme,
    QuantizationStatus,
    apply_quantization_config,
)

from llmcompressor.transformers.compression.compressed_tensors_utils import (
    prune_false_linear_ignores,
)


class Qwen3NextRMSNormGated(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))


class Qwen3NextGatedDeltaNet(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.in_proj_a = torch.nn.Linear(dim, dim)
        self.in_proj_b = torch.nn.Linear(dim, dim)
        self.norm = Qwen3NextRMSNormGated(dim)
        self.out_proj = torch.nn.Linear(dim, dim)


class DeepseekV4TopKRouter(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, dim))


class DeepseekV4TopKGate(torch.nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, dim))


class _TinyHybrid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = Qwen3NextGatedDeltaNet()
        self.mlp_gate = DeepseekV4TopKRouter()
        self.moe_gate = DeepseekV4TopKGate()


def _quantize_out_proj(model: torch.nn.Module) -> None:
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(num_bits=4, strategy="channel"),
    )
    config = QuantizationConfig(
        config_groups={"group_0": scheme},
        quantization_status=QuantizationStatus.INITIALIZED,
        ignore=[
            "re:.*linear_attn\\.in_proj_a$",
            "re:.*linear_attn\\.in_proj_b$",
        ],
    )
    apply_quantization_config(model, config)


def test_prune_drops_gated_containers_keeps_linears_and_routers():
    model = _TinyHybrid()
    ignore = [
        "linear_attn",
        "linear_attn.norm",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "mlp_gate",
        "moe_gate",
    ]
    pruned = prune_false_linear_ignores(model, ignore)
    assert "linear_attn" not in pruned
    assert "linear_attn.norm" not in pruned
    assert "linear_attn.in_proj_a" in pruned
    assert "linear_attn.in_proj_b" in pruned
    assert "mlp_gate" in pruned
    assert "moe_gate" in pruned


def test_from_pretrained_ignore_does_not_list_linear_attn_parent():
    model = _TinyHybrid()
    _quantize_out_proj(model)
    reconstructed = QuantizationConfig.from_pretrained(model)
    assert reconstructed is not None
    pruned = prune_false_linear_ignores(model, reconstructed.ignore)
    assert "linear_attn" not in pruned
    assert "linear_attn.norm" not in pruned
    assert "linear_attn.in_proj_a" in pruned
    assert "linear_attn.in_proj_b" in pruned
    assert "mlp_gate" in pruned
    assert "moe_gate" in pruned
    assert hasattr(model.linear_attn.out_proj, "quantization_scheme")
