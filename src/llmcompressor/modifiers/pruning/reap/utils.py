"""
Utilities for REAP: MoE detection, saliency tracking, and expert pruning.
"""

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from loguru import logger

__all__ = [
    "MoEModelAttrs",
    "REAPSaliencyTracker",
    "detect_moe_attrs",
    "find_moe_layers",
    "get_num_experts",
    "prune_moe_layer",
    "update_model_config",
]


@dataclass
class MoEModelAttrs:
    router_attr: str
    experts_attr: str
    num_experts_config_key: str


MOE_ATTR_REGISTRY: dict[str, MoEModelAttrs] = {
    # Qwen3 family
    "Qwen3MoeSparseMoeBlock": MoEModelAttrs("gate", "experts", "num_experts"),
    "CalibrationQwen3MoeSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts"
    ),
    "Qwen3_5MoeSparseMoeBlock": MoEModelAttrs("gate", "experts", "num_experts"),
    "SequentialQwen3_5MoeSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts"
    ),
    "Qwen3NextSparseMoeBlock": MoEModelAttrs("gate", "experts", "num_experts"),
    "CalibrationQwen3NextSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts"
    ),
    "Qwen3VLMoeTextSparseMoeBlock": MoEModelAttrs("gate", "experts", "num_experts"),
    "SequentialQwen3VLMoeTextSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts"
    ),
    # Mixtral
    "MixtralSparseMoeBlock": MoEModelAttrs("gate", "experts", "num_local_experts"),
    # DeepSeek
    "DeepseekV3MoE": MoEModelAttrs("gate", "experts", "n_routed_experts"),
    "SequentialDeepseekV3MoE": MoEModelAttrs("gate", "experts", "n_routed_experts"),
    # GLM4
    "Glm4MoeMoE": MoEModelAttrs("gate", "experts", "n_routed_experts"),
    "CalibrationGlm4MoeMoE": MoEModelAttrs("gate", "experts", "n_routed_experts"),
    # Llama4
    "Llama4TextMoe": MoEModelAttrs(
        "router", "experts", "num_local_experts"    ),
    "SequentialLlama4TextMoe": MoEModelAttrs("router", "experts", "num_local_experts"),
    # Gemma4
    "Gemma4TextExperts": MoEModelAttrs(
        "router", "experts", "num_experts"    ),
    "SequentialGemma4TextExperts": MoEModelAttrs("router", "experts", "num_experts"),
    # AfMoE
    "AfmoeMoE": MoEModelAttrs("router", "experts", "num_experts"),
    "CalibrationAfmoeMoE": MoEModelAttrs("router", "experts", "num_experts"),
}


def detect_moe_attrs(model: nn.Module) -> MoEModelAttrs | None:
    for _, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name in MOE_ATTR_REGISTRY:
            attrs = MOE_ATTR_REGISTRY[class_name]
            logger.info(
                f"Detected MoE architecture: {class_name} "
                f"(router={attrs.router_attr}, experts={attrs.experts_attr})"
            )
            return attrs

    for _, module in model.named_modules():
        router = _find_router_attr(module)
        if router is not None and hasattr(module, "experts"):
            attrs = MoEModelAttrs(
                router_attr=router,
                experts_attr="experts",
                num_experts_config_key="num_experts",
            )
            logger.info(
                f"Auto-detected MoE block: {module.__class__.__name__} "
                f"(router={router})"
            )
            return attrs

    return None


def _find_router_attr(module: nn.Module) -> str | None:
    for attr_name in ("gate", "router"):
        val = getattr(module, attr_name, None)
        if val is not None and isinstance(val, nn.Linear):
            return attr_name
    return None


def find_moe_layers(
    model: nn.Module,
    attrs: MoEModelAttrs,
    ignore: list[str] | None = None,
) -> dict[str, nn.Module]:
    ignore = ignore or []
    moe_layers = {}

    for name, module in model.named_modules():
        has_router = hasattr(module, attrs.router_attr)
        has_experts = hasattr(module, attrs.experts_attr)
        if not (has_router and has_experts):
            continue
        if not isinstance(getattr(module, attrs.router_attr), nn.Module):
            continue
        if any(re.search(pattern, name) for pattern in ignore):
            continue
        moe_layers[name] = module

    logger.info(f"Found {len(moe_layers)} MoE layers")
    return moe_layers


def get_num_experts(module: nn.Module, attrs: MoEModelAttrs) -> int:
    experts = getattr(module, attrs.experts_attr)
    if isinstance(experts, nn.ModuleList):
        return len(experts)

    if hasattr(experts, "num_experts"):
        return experts.num_experts

    for _, param in experts.named_parameters(recurse=False):
        if param.dim() >= 2:
            return param.shape[0]

    raise ValueError(
        f"Cannot determine number of experts from {module.__class__.__name__}"
    )


@dataclass
class REAPSaliencyTracker:
    """Accumulates S_j = mean(g_j * ||f_j||_2) per expert across batches."""

    num_experts: int
    sum_saliency: torch.Tensor = field(init=False)
    count: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.sum_saliency = torch.zeros(self.num_experts, dtype=torch.float64)
        self.count = torch.zeros(self.num_experts, dtype=torch.int64)

    def update(
        self,
        expert_idx: int,
        gate_values: torch.Tensor,
        output_norms: torch.Tensor,
    ):
        saliency = (gate_values.float() * output_norms.float()).sum()
        self.sum_saliency[expert_idx] += saliency.cpu().to(torch.float64)
        self.count[expert_idx] += gate_values.shape[0]

    @property
    def mean_saliency(self) -> torch.Tensor:
        safe_count = self.count.clamp(min=1).to(torch.float64)
        return self.sum_saliency / safe_count


def prune_moe_layer(
    model: nn.Module,
    layer_name: str,
    saliency: torch.Tensor,
    n_experts_to_drop: int,
    attrs: MoEModelAttrs,
) -> list[int]:
    num_experts = len(saliency)
    if n_experts_to_drop >= num_experts:
        raise ValueError(
            f"Cannot drop {n_experts_to_drop} experts from {layer_name} "
            f"which only has {num_experts} experts"
        )

    _, drop_indices = torch.topk(saliency, n_experts_to_drop, largest=False)
    drop_set = set(drop_indices.tolist())
    retained = sorted(i for i in range(num_experts) if i not in drop_set)

    logger.info(
        f"Pruning {layer_name}: dropping experts {sorted(drop_set)}, "
        f"retaining {retained} ({len(retained)}/{num_experts})"
    )

    moe_block = model.get_submodule(layer_name)
    router = getattr(moe_block, attrs.router_attr)
    experts = getattr(moe_block, attrs.experts_attr)

    if isinstance(experts, nn.ModuleList):
        new_experts = nn.ModuleList([experts[i] for i in retained])
        setattr(moe_block, attrs.experts_attr, new_experts)
    else:
        _prune_fused_experts(experts, retained)

    _prune_router(router, retained)

    if hasattr(moe_block, "num_experts"):
        moe_block.num_experts = len(retained)

    return retained


def _prune_router(router: nn.Module, retained: list[int]):
    retained_t = torch.tensor(retained, dtype=torch.long)

    router.weight = nn.Parameter(
        router.weight.data[retained_t, :], requires_grad=False
    )
    if router.bias is not None:
        router.bias = nn.Parameter(
            router.bias.data[retained_t], requires_grad=False
        )
    router.out_features = len(retained)


def _prune_fused_experts(experts: nn.Module, retained: list[int]):
    retained_t = torch.tensor(retained, dtype=torch.long)

    for name, param in list(experts.named_parameters(recurse=False)):
        if param.dim() >= 2:
            new_param = param.data[retained_t]
            setattr(experts, name, nn.Parameter(new_param, requires_grad=False))

    if hasattr(experts, "num_experts"):
        experts.num_experts = len(retained)


def update_model_config(
    model: nn.Module,
    attrs: MoEModelAttrs,
    new_num_experts: int,
):
    config = model.config
    attr_name = attrs.num_experts_config_key

    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is not None and hasattr(cfg, attr_name):
            old_val = getattr(cfg, attr_name)
            setattr(cfg, attr_name, new_num_experts)
            logger.info(f"Updated config.{attr_name}: {old_val} -> {new_num_experts}")
            return

    logger.warning(f"Config attribute '{attr_name}' not found, skipping config update")
