"""
Utilities for REAP: MoE detection, saliency tracking, and
expert pruning.
"""

import re
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from compressed_tensors import align_module_device
from loguru import logger

from llmcompressor.modeling.moe.granitemoe import GraniteMoeLinearExperts
from llmcompressor.modeling.moe.linear_experts import ExpertMLP, LinearExperts2D
from llmcompressor.modeling.moe.llama4 import Llama4LinearExperts

__all__ = [
    "MoeModelAttrs",
    "REAPSaliencyTracker",
    "get_moe_attrs",
    "prune_moe_layer",
    "update_model_config",
]


@dataclass
class MoeModelAttrs:
    num_experts_config_key: str
    router_attr: str
    experts_attr: str
    moe_layer_names: list[str]
    num_experts: int
    top_k: int
    has_text_config: bool
    n_group: int | None
    top_k_group: int | None
    group_size: int | None


ROUTER_ATTRS = ["router", "gate"]
EXPERTS_ATTRS = ["experts"]
NUM_EXPERTS_CONFIG_KEYS = ["num_experts", "num_local_experts", "moe_num_experts"]
TOP_K_CONFIG_KEYS = ["num_experts_per_tok", "top_k", "moe_top_k"]
N_GROUP_CONFIG_KEYS = ["n_group"]
TOP_K_GROUP_CONFIG_KEYS = ["topk_group", "top_k_group"]
NUM_EXPERTS_MODULE_KEYS = ["num_experts", "n_experts", "n_routed_experts"]


def get_moe_attrs(model: nn.Module, ignore: list[str]) -> MoeModelAttrs | None:
    config = model.config

    num_experts_config_key = None
    router_attr = None
    experts_attr = None
    moe_layer_names = []

    has_text_config = hasattr(config, "text_config")
    config = config.text_config if has_text_config else config

    for key in NUM_EXPERTS_CONFIG_KEYS:
        if hasattr(config, key):
            num_experts_config_key = key
            num_experts = getattr(config, key)
            break

    if num_experts_config_key is None:
        raise ValueError(
            "Could not find a config attribute for the number of experts. "
            "Make sure the name of the model config's "
            "num_experts attribute is in NUM_EXPERTS_CONFIG_KEYS in reap/utils.py"
        )

    top_k = None

    for key in TOP_K_CONFIG_KEYS:
        if hasattr(config, key):
            top_k = getattr(config, key)
            break

    if top_k is None:
        raise ValueError(
            "Could not find a config attribute for the top_k. "
            "Make sure the name of the model config's "
            "top_k attribute is in TOP_K_CONFIG_KEYS in reap/utils.py"
        )

    n_group = None

    for key in N_GROUP_CONFIG_KEYS:
        if hasattr(config, key):
            n_group = getattr(config, key)
            break

    top_k_group = None

    for key in TOP_K_GROUP_CONFIG_KEYS:
        if hasattr(config, key):
            top_k_group = getattr(config, key)
            break

    if (n_group is None) != (top_k_group is None):
        attr_name = "n_group" if n_group is not None else "top_k_group"
        raise ValueError(
            f"Detected one group-limited router attribute ({attr_name} "
            "is set) but could not find a config attribute for the other. "
            "Make sure the name of the model config's n_group and "
            "top_k_group attributes are in N_GROUP_CONFIG_KEYS and "
            "TOP_K_GROUP_CONFIG_KEYS in reap/utils.py"
        )

    group_size = None

    # Group-limited router checks
    if n_group is not None:
        if num_experts % n_group != 0:
            raise ValueError(
                f"Group limited router detected, but {num_experts} experts "
                f"not divisible by n_group={n_group}"
            )

        group_size = num_experts // n_group

    for _, module in model.named_modules():
        for e_attr in EXPERTS_ATTRS:
            if hasattr(module, e_attr):
                for r_attr in ROUTER_ATTRS:
                    if hasattr(module, r_attr):
                        router_attr = r_attr
                        experts_attr = e_attr
                        break
                break
        if experts_attr is not None and router_attr is not None:
            break

    if experts_attr is None or router_attr is None:
        raise ValueError(
            "Could not find a layer with both an experts module and a router "
            "module in the model. Make sure the model has MoE layers, and "
            "that the name of its experts module is in EXPERTS_ATTRS and it "
            "the name of its router module is in ROUTER_ATTRS in reap/utils.py"
        )

    for name, module in model.named_modules():
        if hasattr(module, experts_attr) and hasattr(module, router_attr):
            if any(re.search(pattern, name) for pattern in ignore):
                continue
            experts = getattr(module, experts_attr)
            # REAP currently only supports LinearExperts2D experts, as they receive the
            # top_k indices and weights from the router in their forward pass.
            # Granite and Llama4 experts diverge from this behavior, so they are unsupported
            # for now.
            if not isinstance(experts, LinearExperts2D):
                logger.warning(
                    f"Skipping layer {name}: experts module is not LinearExperts2D"
                )
                continue
            if isinstance(experts, GraniteMoeLinearExperts):
                logger.warning(
                    f"Skipping unsupported GraniteMoeLinearExperts layer: {name}"
                )
                continue
            if isinstance(experts, Llama4LinearExperts):
                logger.warning(
                    f"Skipping unsupported Llama4LinearExperts layer: {name}"
                )
                continue
            moe_layer_names.append(name)

    if not moe_layer_names:
        raise ValueError(
            "Could not find any supported MoE layers with experts in "
            "LinearExperts2D format. Make sure the model has MoE layers "
            "(excluding GraniteMoeLinearExperts and Llama4LinearExperts), "
            "and that the name of its experts module is in EXPERTS_ATTRS "
            "and it the name of its router module is in ROUTER_ATTRS in "
            "reap/utils.py"
        )

    logger.info(
        f"Found {len(moe_layer_names)} MoE layers with experts in "
        "LinearExperts2D format"
    )

    return MoeModelAttrs(
        num_experts_config_key=num_experts_config_key,
        router_attr=router_attr,
        experts_attr=experts_attr,
        moe_layer_names=moe_layer_names,
        num_experts=num_experts,
        top_k=top_k,
        has_text_config=has_text_config,
        n_group=n_group,
        top_k_group=top_k_group,
        group_size=group_size,
    )


# ---------------------------------------------------------------------------
# Saliency tracking
# ---------------------------------------------------------------------------


class REAPSaliencyTracker:
    """
    Accumulates the REAP saliency ``S_j = mean(g_j * ||f_j||_2)`` per expert,
    averaged over the tokens routed to expert ``j``, where ``g_j`` is the router
    gate weight and ``f_j`` is the expert output.

    Accumulators live on the device of the incoming data (allocated lazily) to
    avoid a host sync on every update; they are moved to the host only when
    ``mean_saliency`` is read.
    """

    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        self.sum_saliency: torch.Tensor | None = None
        self.count: torch.Tensor | None = None

    def _ensure(self, device: torch.device):
        if self.sum_saliency is None:
            self.sum_saliency = torch.zeros(
                self.num_experts, dtype=torch.float64, device=device
            )
            self.count = torch.zeros(
                self.num_experts, dtype=torch.float64, device=device
            )

    @torch.no_grad()
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

        # For each expert, find which tokens/slots routed to it and
        # compute contributions
        for expert_idx in range(self.num_experts):
            # If the expert did not receive any tokens in this batch,
            # it won't be in the norms dict
            if expert_idx not in expert_norms_dict:
                continue

            # Find all (token, slot) positions where this expert was selected
            mask = topk_indices == expert_idx
            # (token_indices, slot_indices)
            positions = mask.nonzero(as_tuple=True)
            token_indices, slot_indices = positions

            # Get corresponding norms and weights
            expert_norms = expert_norms_dict[expert_idx]
            assert len(expert_norms) == len(token_indices), (
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

    def compute_retained_experts(
        self,
        n_experts_to_drop: int,
        n_experts_to_drop_per_group: int | None,
        moe_attrs: MoeModelAttrs,
    ) -> list[int]:
        """Select which experts to keep, dropping the lowest-saliency ones."""
        saliency = self.mean_saliency

        if n_experts_to_drop_per_group is None:
            _, drop_indices = torch.topk(saliency, n_experts_to_drop, largest=False)
            drop_set = set(int(i) for i in drop_indices.tolist())
            retained = [i for i in range(self.num_experts) if i not in drop_set]
        else:
            retained: list[int] = []
            for g in range(moe_attrs.n_group):
                lo = g * moe_attrs.group_size
                grp = saliency[lo : lo + moe_attrs.group_size]
                _, drop_local = torch.topk(
                    grp, n_experts_to_drop_per_group, largest=False
                )
                drop_set = {lo + int(i) for i in drop_local.tolist()}
                retained.extend(
                    i for i in range(lo, lo + moe_attrs.group_size) if i not in drop_set
                )

        return retained

    @property
    def total_count(self) -> float:
        if self.count is None:
            return 0.0
        return float(self.count.sum().item())

    @property
    def mean_saliency(self) -> torch.Tensor:
        if self.sum_saliency is None:
            return torch.zeros(self.num_experts, dtype=torch.float64)
        sum_saliency = self.sum_saliency.to("cpu", torch.float64)
        count = self.count.to("cpu", torch.float64)
        return sum_saliency / count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def prune_moe_layer(
    model: nn.Module,
    layer_name: str,
    retained: list[int],
    moe_attrs: MoeModelAttrs,
) -> list[int]:
    """
    Structurally prune a MoE block to keep only ``retained`` experts: slice the
    expert ``ModuleList``, shrink the router, and update expert-count attributes.
    Offload-safe: experts are kept as existing module objects (offload state
    travels with them) and the small router is resized under
    ``align_module_device``.
    """
    moe_block = model.get_submodule(layer_name)
    router = getattr(moe_block, moe_attrs.router_attr)
    experts = getattr(moe_block, moe_attrs.experts_attr)

    # Preserve non-expert modules (e.g., act_fn in LinearExperts2D)
    # These are modules that are not instances of ExpertMLP subclasses
    non_expert_modules = {}
    for key, module in experts._modules.items():
        if not isinstance(module, ExpertMLP):
            non_expert_modules[key] = module

    # Rebuild with retained experts
    new_modules = OrderedDict(
        ((str(i), experts[pos]) for i, pos in enumerate(retained))
    )

    # Re-add non-expert modules
    new_modules.update(non_expert_modules)

    experts._modules = new_modules
    experts.num_experts = len(retained)

    _prune_router(router, retained)

    # Update num_experts for any other modules in the layer that may track it
    for holder in (moe_block, router):
        for key in NUM_EXPERTS_MODULE_KEYS:
            if isinstance(getattr(holder, key, None), int):
                setattr(holder, key, len(retained))

    return retained


def _prune_router(router: nn.Module, retained: list[int]):
    retained_t = torch.tensor(retained, dtype=torch.long)

    with align_module_device(router):
        retained_t = retained_t.to(router.weight.device)
        new_weight = router.weight.detach()[retained_t].contiguous()
        new_bias = None
        if getattr(router, "bias", None) is not None:
            new_bias = router.bias.detach()[retained_t].contiguous()
        # group-limited routers (DeepSeek-V3 / GLM4 / GLM-DSA) carry a per-expert
        # score-correction bias buffer that must be shrunk in lockstep
        correction = getattr(router, "e_score_correction_bias", None)
        new_correction = (
            correction.detach()[retained_t].contiguous()
            if correction is not None
            else None
        )

    # Direct attribute assignment replaces a parameter/buffer with a different
    # shape and is correct for both offloaded modules (routed through the
    # OffloadCache, which re-offloads the new shape) and ordinary modules.
    router.weight = nn.Parameter(new_weight, requires_grad=router.weight.requires_grad)
    if new_bias is not None:
        router.bias = nn.Parameter(new_bias, requires_grad=router.bias.requires_grad)
    if new_correction is not None:
        router.e_score_correction_bias = new_correction

    if isinstance(getattr(router, "out_features", None), int):
        router.out_features = len(retained)


def update_model_config(
    model: nn.Module,
    moe_attrs: MoeModelAttrs,
    new_num_experts: int,
):
    config = model.config.text_config if moe_attrs.has_text_config else model.config

    old_val = getattr(config, moe_attrs.num_experts_config_key)
    setattr(config, moe_attrs.num_experts_config_key, new_num_experts)
    logger.info(
        f"Updated {config.__class__.__name__}."
        f"{moe_attrs.num_experts_config_key}: {old_val} -> {new_num_experts}"
    )
