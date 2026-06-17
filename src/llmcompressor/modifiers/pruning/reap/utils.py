"""
Utilities for REAP: MoE detection, routing extraction, saliency tracking, and
expert pruning.

Supported architectures all share the property that, under
``moe_calibration_context``, the MoE block is replaced by a calibration wrapper
that routes *every* token through *every* expert (``calibrate_all_experts``).
This lets REAP observe each expert's output for all tokens. Each architecture's
router, however, produces gate weights differently, so the routing math is
captured per-architecture via ``ROUTING_MODE`` (see ``extract_routing``).
"""

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressed_tensors import align_module_device
from loguru import logger

__all__ = [
    "MoEModelAttrs",
    "REAPSaliencyTracker",
    "assert_homogeneous_moe",
    "assert_routing_feasible",
    "compute_retained_experts",
    "detect_moe_attrs",
    "extract_routing",
    "find_moe_layers",
    "get_num_experts",
    "get_router_num_groups",
    "get_router_topk_group",
    "prune_moe_layer",
    "update_model_config",
]


# Routing modes describe how to turn a router module's raw forward output into
# ``(topk_indices, topk_weights)`` -- the experts selected per token and the
# gate weight applied to each. See ``extract_routing``.
SOFTMAX = "softmax"  # router returns logits [T, E]; softmax + top-k (+ optional norm)
INDICES_WEIGHTS = "indices_weights"  # router returns (topk_indices, topk_weights)
WEIGHTS_INDICES = "weights_indices"  # router returns (topk_weights, topk_indices)
SCORES_LOGITS = "scores_logits"  # router returns (scores [T, E], logits [T, E])
GLM_DSA = "glm_dsa"  # router returns logits; group-limited sigmoid routing in block


@dataclass
class MoEModelAttrs:
    router_attr: str
    experts_attr: str
    num_experts_config_key: str
    routing_mode: str = SOFTMAX


MOE_ATTR_REGISTRY: dict[str, MoEModelAttrs] = {
    # Qwen3 family (softmax router)
    "Qwen3MoeSparseMoeBlock": MoEModelAttrs("gate", "experts", "num_experts", SOFTMAX),
    "CalibrationQwen3MoeSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts", SOFTMAX
    ),
    "Qwen3_5MoeSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts", SOFTMAX
    ),
    "SequentialQwen3_5MoeSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts", SOFTMAX
    ),
    "Qwen3NextSparseMoeBlock": MoEModelAttrs("gate", "experts", "num_experts", SOFTMAX),
    "CalibrationQwen3NextSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts", SOFTMAX
    ),
    "Qwen3VLMoeTextSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts", SOFTMAX
    ),
    "CalibrateQwen3VLMoeTextSparseMoeBlock": MoEModelAttrs(
        "gate", "experts", "num_experts", SOFTMAX
    ),
    # DeepSeek-V3 (sigmoid + group-limited top-k router, returns indices+weights)
    "DeepseekV3MoE": MoEModelAttrs(
        "gate", "experts", "n_routed_experts", INDICES_WEIGHTS
    ),
    "CalibrationDeepseekV3MoE": MoEModelAttrs(
        "gate", "experts", "n_routed_experts", INDICES_WEIGHTS
    ),
    # GLM4 MoE (sigmoid + group-limited top-k router, returns indices+weights)
    "Glm4MoeMoE": MoEModelAttrs("gate", "experts", "n_routed_experts", INDICES_WEIGHTS),
    "CalibrationGlm4MoeMoE": MoEModelAttrs(
        "gate", "experts", "n_routed_experts", INDICES_WEIGHTS
    ),
    # GLM-MoE-DSA / GLM4-MoE-Lite (group-limited sigmoid routing inside the block)
    "GlmMoeDsaMoE": MoEModelAttrs("gate", "experts", "n_routed_experts", GLM_DSA),
    "CalibrationGlmMoeDsaMoE": MoEModelAttrs(
        "gate", "experts", "n_routed_experts", GLM_DSA
    ),
    "Glm4MoeLiteMoE": MoEModelAttrs("gate", "experts", "n_routed_experts", GLM_DSA),
    "CalibrationGlm4MoeLiteMoE": MoEModelAttrs(
        "gate", "experts", "n_routed_experts", GLM_DSA
    ),
    # Llama4 (router returns sigmoid scores + logits)
    "Llama4TextMoe": MoEModelAttrs(
        "router", "experts", "num_local_experts", SCORES_LOGITS
    ),
    "SequentialLlama4TextMoe": MoEModelAttrs(
        "router", "experts", "num_local_experts", SCORES_LOGITS
    ),
    # AfMoE (router returns scores + selected experts)
    "AfmoeMoE": MoEModelAttrs("router", "experts", "num_experts", WEIGHTS_INDICES),
    "CalibrationAfmoeMoE": MoEModelAttrs(
        "router", "experts", "num_experts", WEIGHTS_INDICES
    ),
}

# Architectures that are intentionally not supported, mapped to the reason.
# Detected here so REAP can raise a clear error instead of silently
# mis-calibrating or emitting a broken checkpoint.
UNSUPPORTED_MOE_BLOCKS: dict[str, str] = {
    # No calibration wrapper exists, so calibrate_all_experts cannot route every
    # token through every expert -- REAP's per-expert norms would be incomplete.
    "MixtralSparseMoeBlock": (
        "Mixtral has no MoE calibration wrapper, so REAP cannot observe every "
        "expert on every token."
    ),
    # The Gemma4 'experts' block has no router attribute (routing happens in the
    # parent module and is passed in as arguments), so REAP cannot locate or
    # prune the router alongside the experts.
    "Gemma4TextExperts": (
        "Gemma4 routes outside the experts block; REAP cannot pair a router with "
        "the experts module."
    ),
    "SequentialGemma4TextExperts": (
        "Gemma4 routes outside the experts block; REAP cannot pair a router with "
        "the experts module."
    ),
}


def detect_moe_attrs(model: nn.Module) -> MoEModelAttrs | None:
    for _, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name in UNSUPPORTED_MOE_BLOCKS:
            raise NotImplementedError(
                f"REAP does not support MoE architecture '{class_name}': "
                f"{UNSUPPORTED_MOE_BLOCKS[class_name]}"
            )
        if class_name in MOE_ATTR_REGISTRY:
            attrs = MOE_ATTR_REGISTRY[class_name]
            logger.info(
                f"Detected MoE architecture: {class_name} "
                f"(router={attrs.router_attr}, experts={attrs.experts_attr}, "
                f"routing={attrs.routing_mode})"
            )
            return attrs

    # validate that "num_experts" exists in config before hard coding
    # it for auto-detection
    config = model.config
    config_has_num_experts = hasattr(config, "num_experts") or (
        hasattr(config, "text_config") and hasattr(config.text_config, "num_experts")
    )
    if not config_has_num_experts:
        model_class = next(model.named_modules())[1].__class__.__name__
        raise ValueError(
            f"Attribute auto-detection failed for model '{model_class}': "
            "'num_experts' not found in model.config or model.config.text_config. "
            "Please add your model's details to MOE_ATTR_REGISTRY in "
            "src/llmcompressor/modifiers/pruning/reap/utils.py to avoid auto-detection."
        )

    for _, module in model.named_modules():
        router = _find_router_attr(module)
        if router is not None and hasattr(module, "experts"):
            attrs = MoEModelAttrs(
                router_attr=router,
                experts_attr="experts",
                num_experts_config_key="num_experts",
                routing_mode=SOFTMAX,
            )
            logger.info(
                f"Auto-detected MoE block: {module.__class__.__name__} "
                f"(router={router}, experts=experts, routing={SOFTMAX})"
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

    raise ValueError(
        f"Cannot determine number of experts from {module.__class__.__name__}:"
        " experts are not in 2D format as expected"
    )


def get_router_num_groups(module: nn.Module, attrs: MoEModelAttrs) -> int:
    """
    Number of expert groups used by group-limited routers (DeepSeek-V3, GLM4,
    GLM-MoE-DSA). Returns 1 for routers without grouping. ``n_group`` may live on
    the router module or on the (calibration) block.
    """
    router = getattr(module, attrs.router_attr)
    for holder in (router, module):
        n_group = getattr(holder, "n_group", None)
        if isinstance(n_group, int) and n_group > 0:
            return n_group
    logger.info(
        f"n_group attribute not found on {module.__class__.__name__};"
        " assuming n_group=1"
    )
    return 1


def get_router_topk_group(module: nn.Module, attrs: MoEModelAttrs) -> int:
    """
    Number of expert groups a token may route into (``topk_group``) for
    group-limited routers. Defaults to ``n_group`` (all groups) when not set.
    """
    router = getattr(module, attrs.router_attr)
    for holder in (router, module):
        topk_group = getattr(holder, "topk_group", None)
        if isinstance(topk_group, int) and topk_group > 0:
            return topk_group
    logger.info(
        f"topk_group attribute not found on {module.__class__.__name__};"
        " assuming topk_group=n_group"
    )
    return get_router_num_groups(module, attrs)


def assert_homogeneous_moe(moe_layers: dict[str, nn.Module], attrs: MoEModelAttrs):
    """
    Assert that all detected MoE layers share the same ``num_experts``,
    ``n_group``, and ``topk_group``.
    """
    num_experts = set()
    n_groups = set()
    topk_groups = set()

    for module in moe_layers.values():
        num_experts.add(get_num_experts(module, attrs))
        n_groups.add(get_router_num_groups(module, attrs))
        topk_groups.add(get_router_topk_group(module, attrs))

    if len(num_experts) > 1:
        raise ValueError(
            f"Detected heterogeneous num_experts across MoE layers: {num_experts}"
        )
    if len(n_groups) > 1:
        raise ValueError(
            f"Detected heterogeneous n_group across MoE layers: {n_groups}"
        )
    if len(topk_groups) > 1:
        raise ValueError(
            f"Detected heterogeneous topk_group across MoE layers: {topk_groups}"
        )


# ---------------------------------------------------------------------------
# Routing extraction
# ---------------------------------------------------------------------------


def extract_routing(
    router_output,
    block: nn.Module,
    attrs: MoEModelAttrs,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a router module's raw forward output into per-token routing
    decisions, matching how the architecture actually combines experts.

    :param router_output: the value returned by the router submodule's forward
    :param block: the MoE block (calibration wrapper) holding routing config
    :param attrs: architecture descriptor (selects the routing mode)
    :param top_k: number of experts selected per token
    :return: ``(topk_indices, topk_weights)`` each shaped ``[num_tokens, top_k]``;
        ``topk_weights[t, s]`` is the gate weight applied to expert
        ``topk_indices[t, s]`` for token ``t``
    """
    mode = attrs.routing_mode

    if mode == INDICES_WEIGHTS:
        topk_indices, topk_weights = router_output
        return topk_indices, topk_weights

    if mode == WEIGHTS_INDICES:
        topk_weights, topk_indices = router_output
        return topk_indices, topk_weights

    if mode == SCORES_LOGITS:
        scores, logits = router_output
        _, topk_indices = torch.topk(logits, top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_indices)
        return topk_indices, topk_weights

    if mode == SOFTMAX:
        logits = router_output
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
        if getattr(block, "norm_topk_prob", True):
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_indices, topk_weights

    if mode == GLM_DSA:
        return _route_glm_dsa(router_output, block, attrs, top_k)

    raise ValueError(f"Unknown routing mode '{mode}'")


def _route_glm_dsa(
    router_logits: torch.Tensor,
    block: nn.Module,
    attrs: MoEModelAttrs,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Group-limited sigmoid routing, mirroring
    ``CalibrationGlmMoeDsaMoE.route_tokens_to_experts``. The score-correction
    bias and group/scaling parameters live on the block (and its gate).
    """
    router = getattr(block, attrs.router_attr)
    n_group = int(getattr(block, "n_group", getattr(router, "n_group", 1)) or 1)
    topk_group = int(getattr(block, "topk_group", n_group) or n_group)
    n_routed = router_logits.shape[-1]
    norm_topk_prob = getattr(block, "norm_topk_prob", True)
    routed_scaling_factor = getattr(block, "routed_scaling_factor", 1.0)
    correction_bias = getattr(router, "e_score_correction_bias", 0.0)

    scores = router_logits.sigmoid()
    scores_for_choice = scores + correction_bias
    group_scores = (
        scores_for_choice.view(-1, n_group, n_routed // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, n_routed // n_group)
        .reshape(-1, n_routed)
    )
    masked = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    topk_indices = torch.topk(masked, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


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
        expert_norms: torch.Tensor,
    ):
        """
        Vectorized accumulation over one batch.

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids
        :param topk_weights: ``[num_tokens, top_k]`` gate weight per selection
        :param expert_norms: ``[num_tokens, num_experts]`` output norm of every
            expert for every token (available because all experts see all tokens)
        """
        self._ensure(expert_norms.device)

        flat_idx = topk_indices.reshape(-1).to(torch.long)
        gathered_norms = expert_norms.gather(1, topk_indices.to(torch.long))
        contrib = topk_weights.to(torch.float64) * gathered_norms.to(torch.float64)
        self.sum_saliency.index_add_(0, flat_idx, contrib.reshape(-1))
        self.count.index_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float64)
        )

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
    n_group: int = 1,
) -> list[int]:
    """
    Select which experts to keep, dropping the lowest-saliency ones.

    For group-limited routers (``n_group > 1``) the experts form ``n_group``
    contiguous groups and the router requires every group to keep the same
    number of experts (so ``n_routed_experts`` stays divisible by ``n_group``).
    We therefore drop an equal number from each group; the requested drop count
    is rounded to the nearest multiple of ``n_group``.
    """
    num_experts = len(saliency)
    if n_experts_to_drop >= num_experts:
        raise ValueError(
            f"Cannot drop {n_experts_to_drop} experts from a layer with only "
            f"{num_experts} experts"
        )

    if n_group <= 1:
        _, drop_idx = torch.topk(saliency, n_experts_to_drop, largest=False)
        drop_set = set(drop_idx.tolist())
        return [i for i in range(num_experts) if i not in drop_set]

    if num_experts % n_group != 0:
        raise ValueError(f"{num_experts} experts not divisible by n_group={n_group}")
    group_size = num_experts // n_group
    drop_per_group = _drop_per_group(n_experts_to_drop, n_group, group_size)
    if drop_per_group == 0:
        raise ValueError(
            f"Requested drop of {n_experts_to_drop} experts rounds to 0 per "
            f"group (n_group={n_group}, group_size={group_size})"
        )
    effective = drop_per_group * n_group
    if effective != n_experts_to_drop:
        logger.warning(
            f"REAP: group-limited routing requires an equal drop per group; "
            f"dropping {effective} experts instead of the requested "
            f"{n_experts_to_drop} (n_group={n_group})"
        )

    retained: list[int] = []
    for g in range(n_group):
        lo = g * group_size
        grp = saliency[lo : lo + group_size]
        _, drop_local = torch.topk(grp, drop_per_group, largest=False)
        drop_set = {lo + int(i) for i in drop_local.tolist()}
        retained.extend(i for i in range(lo, lo + group_size) if i not in drop_set)
    return retained


def _drop_per_group(n_experts_to_drop: int, n_group: int, group_size: int) -> int:
    """Per-group drop count for group-limited routing: equal across groups and
    leaving at least 2 experts per group (group routing does a top-2 per group)."""
    drop_per_group = round(n_experts_to_drop / n_group)
    return max(0, min(drop_per_group, group_size - 2))


def assert_routing_feasible(
    num_experts: int,
    n_experts_to_drop: int,
    n_group: int,
    topk_group: int,
    top_k: int,
):
    """
    Ensure pruning leaves enough experts for the router to still select ``top_k``
    per token. For group-limited routers a token may only route into
    ``topk_group`` groups, so the reachable experts after pruning are
    ``topk_group * retained_per_group``. Raises ``ValueError`` (fail fast, before
    calibration) when the requested sparsity would make routing infeasible.
    """
    if n_group > 1:
        group_size = num_experts // n_group
        retained_per_group = group_size - _drop_per_group(
            n_experts_to_drop, n_group, group_size
        )
        available = topk_group * retained_per_group
    else:
        available = num_experts - n_experts_to_drop

    if top_k > available:
        raise ValueError(
            f"REAP sparsity is too aggressive: the router selects top_k={top_k} "
            f"experts per token, but only {available} experts would remain "
            f"reachable after pruning "
            f"(num_experts={num_experts}, dropping≈{n_experts_to_drop}"
            + (f", n_group={n_group}, topk_group={topk_group}" if n_group > 1 else "")
            + "). Reduce the sparsity."
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

    if isinstance(experts, nn.ModuleList):
        old_experts = experts
        new_experts = nn.ModuleList([experts[i] for i in retained])
        setattr(moe_block, attrs.experts_attr, new_experts)
        del old_experts
    else:
        raise ValueError(
            f"Cannot prune experts from layer {layer_name}:"
            " experts are not in 2D format as expected"
        )

    _prune_router(router, retained)

    # n_group / topk_group are intentionally left unchanged: per-group-balanced
    # pruning keeps the same number of equally-sized groups, so n_routed_experts
    # stays divisible by n_group and the router's grouping stays valid.
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
    router.weight = nn.Parameter(new_weight, requires_grad=False)
    if new_bias is not None:
        router.bias = nn.Parameter(new_bias, requires_grad=False)
    if new_correction is not None:
        router.e_score_correction_bias = new_correction

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
