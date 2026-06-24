"""
Utilities for REAP: MoE detection, routing extraction, saliency tracking, and
expert pruning.
"""

from collections import OrderedDict
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressed_tensors import align_module_device
from loguru import logger

from llmcompressor.modeling.moe.linear_experts import ExpertMLP, LinearExperts2D
from llmcompressor.modeling.moe.granitemoe import GraniteMoeLinearExperts
from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts

__all__ = [
    "MoEModelAttrs",
    "REAPSaliencyTracker",
    "assert_homogeneous_moe",
    "assert_routing_feasible",
    "compute_retained_experts",
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

ROUTER_ATTRS = ["router", "gate"]
EXPERTS_ATTRS = ["experts"]
NUM_EXPERTS_CONFIG_KEYS = ["num_experts", "num_local_experts"]

def detect_moe_attrs(model: nn.Module) -> MoEModelAttrs | None:
    config = model.config

    router_attr = None
    experts_attr = None
    num_experts_config_key = None

    for key in NUM_EXPERTS_CONFIG_KEYS:
        if hasattr(config, key):
            num_experts_config_key = key
            break
        if hasattr(config, "text_config") and hasattr(config.text_config, key):
            num_experts_config_key = key
            break
    
    for _, module in model.named_modules():
        for attr in EXPERTS_ATTRS:
            if hasattr(module, attr):
                experts_attr = attr
                break
        for attr in ROUTER_ATTRS:
            if hasattr(module, attr):
                router_attr = attr
                break
        if experts_attr is not None and router_attr is not None:
            break

    if experts_attr is not None and num_experts_config_key is not None and router_attr is not None:
        return MoEModelAttrs(
            experts_attr=experts_attr,
            num_experts_config_key=num_experts_config_key,
            router_attr=router_attr
        )

    return None

def find_moe_layers(
    model: nn.Module,
    attrs: MoEModelAttrs,
    ignore: list[str] | None = None,
) -> dict[str, nn.Module]:
    ignore = ignore or []
    moe_layers = {}

    for name, module in model.named_modules():
        has_experts = hasattr(module, attrs.experts_attr)
        if not has_experts:
            continue
        
        experts = getattr(module, attrs.experts_attr)
        if not isinstance(experts, LinearExperts2D):
            continue
        if isinstance(experts, GraniteMoeLinearExperts):
            continue
        if isinstance(experts, Llama4LinearExperts):
            continue
        if any(re.search(pattern, name) for pattern in ignore):
            continue
        moe_layers[name] = module

    logger.info(f"Found {len(moe_layers)} MoE layers with experts in LinearExperts2D format")
    return moe_layers


def get_num_experts(module: nn.Module, attrs: MoEModelAttrs) -> int:
    """"Return the number of experts in a MoE layer, given that the layer has experts in LinearExperts 2D format"""
    experts = getattr(module, attrs.experts_attr)
    return experts.num_experts


def assert_homogeneous_moe(moe_layers: dict[str, nn.Module], attrs: MoEModelAttrs):
    """Assert that all detected MoE layers share the same ``num_experts``"""
    num_experts = set()

    for module in moe_layers.values():
        num_experts.add(get_num_experts(module, attrs))

    if len(num_experts) > 1:
        raise ValueError(
            f"Detected heterogeneous num_experts across MoE layers: {num_experts}"
        )


# ---------------------------------------------------------------------------
# Saliency tracking
# ---------------------------------------------------------------------------


@dataclass
class REAPSaliencyTracker:
    """
    Accumulates the REAP saliency ``S_j = mean(g_j * ||f_j||_2)`` per expert,
    averaged over the tokens routed to expert ``j``, where ``g_j`` is the router
    gate weight and ``f_j`` is the expert output.

    Accumulators live on the device of the incoming data (allocated lazily) to
    avoid a host sync on every update; they are moved to the host only when
    ``mean_saliency`` is read.
    """

    num_experts: int
    sum_saliency: torch.Tensor | None = field(init=False, default=None)
    count: torch.Tensor | None = field(init=False, default=None)

    def _ensure(self, device: torch.device):
        if self.sum_saliency is None:
            self.sum_saliency = torch.zeros(
                self.num_experts, dtype=torch.float64, device=device
            )
            self.count = torch.zeros(
                self.num_experts, dtype=torch.float64, device=device
            )

    def update(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_norms_dict: dict[int, torch.Tensor],
    ):
        """
        Vectorized accumulation over one batch.

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids
        :param topk_weights: ``[num_tokens, top_k]`` gate weight per selection
        :param expert_norms_dict: dict mapping expert_idx to output norms
            ``[num_routed_tokens]`` for tokens routed to that expert (sparse
            routing: experts only see tokens the router sent to them)
        """
        if not expert_norms_dict:
            return

        self._ensure(next(iter(expert_norms_dict.values())).device)
        
        # For each expert, find which tokens/slots routed to it and compute contributions
        for expert_idx in range(self.num_experts):
            if expert_idx not in expert_norms_dict:
                continue

            # Find all (token, slot) positions where this expert was selected
            mask = (topk_indices == expert_idx)
            positions = mask.nonzero(as_tuple=True)  # (token_indices, slot_indices)
            token_indices, slot_indices = positions

            # Get corresponding norms and weights
            expert_norms = expert_norms_dict[expert_idx]
            if len(expert_norms) != len(token_indices):
                raise RuntimeError(
                    f"REAP saliency tracker: expert {expert_idx} has "
                    f"{len(expert_norms)} norms but router sent "
                    f"{len(token_indices)} tokens to it. This indicates a bug in "
                    f"the expert hook or routing extraction logic."
                )

            expert_weights = topk_weights[token_indices, slot_indices]

            # Compute contributions
            contrib = expert_weights.to(torch.float64) * expert_norms.to(torch.float64)

            # Sum contributions for this expert
            self.sum_saliency[expert_idx] += contrib.sum()
            self.count[expert_idx] += len(contrib)

    @property
    def total_count(self) -> float:
        if self.count is None:
            return 0.0
        return float(self.count.sum().item())

    @property
    def mean_saliency(self) -> torch.Tensor:
        if self.sum_saliency is None:
            return torch.zeros(self.num_experts, dtype=torch.float64)
        sum_saliency = self.sum_saliency.detach().to("cpu", torch.float64)
        count = self.count.detach().to("cpu", torch.float64)
        return sum_saliency / count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def compute_retained_experts(
    saliency: torch.Tensor,
    n_experts_to_drop: int,
) -> list[int]:
    """Select which experts to keep, dropping the lowest-saliency ones."""
    num_experts = len(saliency)
    if n_experts_to_drop >= num_experts:
        raise ValueError(
            f"Cannot drop {n_experts_to_drop} experts from a layer with only "
            f"{num_experts} experts"
        )

    _, drop_indices = torch.topk(saliency, n_experts_to_drop, largest=False)
    drop_set = set(int(i) for i in drop_indices.tolist())
    retained = [i for i in range(num_experts) if i not in drop_set]

    return retained


def assert_routing_feasible(
    num_experts: int,
    n_experts_to_drop: int,
    top_k: int,
):
    """
    Ensure pruning leaves enough experts for the router to still select ``top_k``
    per token. Raises ``ValueError`` (fail fast, before calibration) when the
    requested sparsity would make routing infeasible.
    """
    available = num_experts - n_experts_to_drop

    if top_k > available:
        raise ValueError(
            f"REAP sparsity is too aggressive: the router selects top_k={top_k} "
            f"experts per token, but only {available} experts would remain "
            f"reachable after pruning "
            f"(num_experts={num_experts}, dropping≈{n_experts_to_drop})."
            f" Reduce the sparsity"
        )


def prune_moe_layer(
    model: nn.Module,
    layer_name: str,
    retained: list[int],
    attrs: MoEModelAttrs,
) -> list[int]:
    """
    Structurally prune a MoE block to keep only ``retained`` experts: rebuild the
    expert ``ModuleList``, shrink the router, and update expert-count attributes.
    Offload-safe: experts are kept as existing module objects (offload state
    travels with them) and the small router is resized under
    ``align_module_device``.
    """
    moe_block = model.get_submodule(layer_name)
    router = getattr(moe_block, attrs.router_attr)
    experts = getattr(moe_block, attrs.experts_attr)

    if isinstance(experts, LinearExperts2D):
        # Preserve non-expert modules (e.g., act_fn in LinearExperts2D)
        # These are modules that are not instances of ExpertMLP subclasses
        non_expert_modules = {}
        for key, module in experts._modules.items():
            if not isinstance(module, ExpertMLP):
                non_expert_modules[key] = module

        # Rebuild with retained experts
        str_indices = [str(i) for i in range(len(retained))]
        new_modules = OrderedDict(zip(str_indices, [experts[i] for i in retained]))

        # Re-add non-expert modules
        new_modules.update(non_expert_modules)

        experts._modules = new_modules
        if hasattr(experts, "num_experts"):
            experts.num_experts = len(retained)
    else:
        raise ValueError(
            f"Cannot prune experts from layer {layer_name}:"
            " experts are not in 2D format as expected"
        )

    _prune_router(router, retained)

    for holder in (moe_block, router):
        if isinstance(getattr(holder, "num_experts", None), int):
            holder.num_experts = len(retained)
        if isinstance(getattr(holder, "n_routed_experts", None), int):
            holder.n_routed_experts = len(retained)

    return retained


def _prune_router(router: nn.Module, retained: list[int]):
    retained_t = torch.tensor(retained, dtype=torch.long)

    with align_module_device(router):
        retained_t = retained_t.to(router.weight.device)
        new_weight = router.weight.detach()[retained_t].contiguous()
        new_bias = None
        if getattr(router, "bias", None) is not None:
            new_bias = router.bias.detach()[retained_t].contiguous()

    # Direct attribute assignment replaces a parameter/buffer with a different
    # shape and is correct for both offloaded modules (routed through the
    # OffloadCache, which re-offloads the new shape) and ordinary modules.
    router.weight = nn.Parameter(new_weight, requires_grad=False)
    if new_bias is not None:
        router.bias = nn.Parameter(new_bias, requires_grad=False)

    if isinstance(getattr(router, "out_features", None), int):
        router.out_features = len(retained)


def update_model_config(
    model: nn.Module,
    attrs: MoEModelAttrs,
    new_num_experts: int,
):
    config = model.config
    attr_name = attrs.num_experts_config_key

    updated = False
    for cfg, cfg_name in [
        (config, "config"),
        (getattr(config, "text_config", None), "text_config"),
    ]:
        if cfg is not None and hasattr(cfg, attr_name):
            old_val = getattr(cfg, attr_name)
            setattr(cfg, attr_name, new_num_experts)
            logger.info(
                f"Updated {cfg_name}.{attr_name}: {old_val} -> {new_num_experts}"
            )
            updated = True

    if not updated:
        logger.warning(
            f"Config attribute '{attr_name}' not found, skipping config update"
        )
