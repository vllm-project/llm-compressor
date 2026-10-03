# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import re
from typing import Literal

import torch
from compressed_tensors.config import CompressionFormat
from compressed_tensors.entrypoints.convert.converters import Converter
from compressed_tensors.quantization import QuantizationConfig
from compressed_tensors.utils.moe import (
    build_expert_to_router_map,
    extract_expert_index,
    get_num_experts_config_key,
    get_num_experts_per_tok,
    load_model_config,
    renumber_expert_name,
)
from compressed_tensors.utils.safetensors_load import (
    get_checkpoint_files,
    get_weight_map,
)
from loguru import logger
from safetensors import safe_open

__all__ = ["ExpertPruner"]

# metrics which are read from a saliency report written by ``REAPPruningModifier``
REPORT_METRICS = ("saliency", "layerwise_saliency", "count")


class ExpertPruner(Converter):
    """
    Prune MoE experts based on a per-expert score. Experts are scored either by
    router weight magnitude (``router.weight[i].abs().sum()``, L1 magnitude) or
    by a statistic read from a saliency report produced by
    ``REAPPruningModifier`` (see its ``report_path`` argument). The
    lowest-scoring ``sparsity`` fraction of experts is pruned, retaining the
    rest. With ``uniform``, the same fraction is pruned from every layer.
    Otherwise, the lowest-scoring experts across all layers are pruned, so layers
    may retain different numbers of experts.

    Supports both 2D expert weights (one tensor per expert, e.g.
    ``model.layers.0.mlp.experts.3.gate_proj.weight``) and 3D stacked expert
    weights (``model.layers.0.mlp.experts.gate_proj.weight`` with shape
    ``[num_experts, out_features, in_features]``).

    Pruning removes expert weight tensors (or slices stacked tensors), adjusts
    router weights, and updates the expert count. Uniform pruning writes the new
    count to the model config, while non-uniform pruning records per-layer
    counts in the quantization config's ``layer_overrides``.

    Besides the router weight, a layer may carry auxiliary router tensors whose
    shape or contents depend on the expert count. These are pruned alongside the
    router:

    - **per-expert vectors** (``aux_pattern``), e.g. a routing-bias such as
      DeepSeek's ``ffn.gate.bias`` (``e_score_correction_bias``) with shape
      ``[num_experts]``. Sliced along dim 0 by the retained expert indices,
      exactly like the router weight's rows.
    - **expert-index tables** (``index_pattern``), e.g. DeepSeek's hash-routing
      ``ffn.gate.tid2eid`` with shape ``[vocab, num_experts_per_tok]`` whose
      *values* are expert indices. The values are remapped from old to new
      expert indices after renumbering. A layer carrying such a table is a hash
      (statically routed) layer: see the hash-layer handling below.

    Hash layers route each token to a fixed set of experts via a lookup table
    rather than a learned top-k, so experts cannot be redistributed after
    pruning. With ``uniform=True`` any hash layer makes pruning ambiguous and
    raises. With ``uniform=False`` every expert in a hash layer is protected
    (never pruned), so its table stays valid and the global budget is spent on
    the remaining learned-routing layers.

    :param router_pattern: regex matching router weight tensor names
    :param expert_pattern: regex matching expert weight tensor names
    :param aux_pattern: regex matching per-expert router vectors (e.g. routing
        bias) that must be sliced along dim 0 by retained expert indices
    :param index_pattern: regex matching expert-index tables (e.g. a hash
        routing table) whose values are expert indices to be remapped
    :param sparsity: fraction of experts to prune, in [0, 1]
    :param uniform: whether the same number of experts is pruned from every layer
    :param num_experts_config_key: config.json attribute name for expert count
    :param retained_experts: pre-computed mapping of router_name -> retained
        expert indices (sorted). Built by :meth:`from_pretrained`.
    :param expert_to_router: mapping of expert tensor name -> associated router
        tensor name. Built by :meth:`from_pretrained`.
    :param aux_to_router: mapping of auxiliary tensor name (bias or index table)
        -> associated router tensor name. Built by :meth:`from_pretrained`.
    :param hash_routers: set of router names whose layer routes statically via an
        expert-index table (``index_pattern``). Built by :meth:`from_pretrained`.
    :param expert_indices: for 2D experts, mapping of tensor name -> expert
        index. Empty for 3D experts.
    :param is_3d: whether expert weights are 3D stacked tensors
    """

    def __init__(self, *args, **kwargs):
        raise ValueError("Please use `from_pretrained`")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        sparsity: float,
        metric: Literal["magnitude", "saliency", "layerwise_saliency", "count"],
        uniform: bool = True,
        router_pattern: str = r"mlp\.gate\.weight$",
        expert_pattern: str = r"mlp\.experts\.(gate_up_proj|down_proj)",
        aux_pattern: str | None = r"(?:mlp|ffn)\.gate\.bias$",
        index_pattern: str | None = r"(?:mlp|ffn)\.gate\.tid2eid$",
        num_experts_config_key: str | None = None,
        saliency_report_path: str | None = None,
    ) -> ExpertPruner:
        """
        Build the converter by scanning the checkpoint for router weights,
        scoring experts, and determining which to retain.

        :param model_name_or_path: HuggingFace stub or local checkpoint path
        :param sparsity: fraction of experts to prune, in [0, 1]
        :param metric: how experts are scored, lowest scores are pruned first:
            - ``"magnitude"``: L1 magnitude of each expert's router weight row
            - ``"saliency"``: raw REAP saliency from the saliency report
            - ``"layerwise_saliency"``: REAP saliency divided by the highest
              saliency of its layer. Normalizing makes saliency comparable
              across layers, which matters for non-uniform pruning
            - ``"count"``: number of tokens routed to each expert, from the
              saliency report
        :param uniform: if True, prune ``round(sparsity * num_experts)`` experts
            from every layer and write the new count to
            ``num_experts_config_key`` in the model config. If False, prune the
            ``round(sparsity * total_experts)`` lowest-scoring experts across all
            layers, never taking a layer below ``num_experts_per_tok``, and
            record the per-layer counts under ``num_experts_config_key`` in the
            quantization config's ``layer_overrides``
        :param router_pattern: regex matching router weight tensor names
        :param expert_pattern: regex matching expert weight tensor names
        :param aux_pattern: regex matching per-expert router vectors (e.g.
            ``ffn.gate.bias``) sliced along dim 0 by retained experts. ``None``
            disables auxiliary-vector pruning
        :param index_pattern: regex matching expert-index tables (e.g.
            ``ffn.gate.tid2eid``) whose values are remapped to new expert
            indices. A layer with such a table is treated as a hash layer.
            ``None`` disables index-table handling
        :param num_experts_config_key: config.json key holding the expert count.
            If None, auto-detected from config.json.
        :param saliency_report_path: path to the ``.json`` report written by
            ``REAPPruningModifier``. Required for every metric but
            ``"magnitude"``. Its layers must be ordered by layer index
        """
        if not 0 <= sparsity <= 1:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
        if metric != "magnitude" and metric not in REPORT_METRICS:
            raise ValueError(
                f"metric must be one of {('magnitude',) + REPORT_METRICS}, "
                f"got {metric!r}"
            )
        if metric in REPORT_METRICS and saliency_report_path is None:
            raise ValueError(f"metric={metric!r} requires a saliency_report_path")

        router_re = re.compile(router_pattern)
        expert_re = re.compile(expert_pattern)
        aux_re = re.compile(aux_pattern) if aux_pattern else None
        index_re = re.compile(index_pattern) if index_pattern else None

        model_files = get_checkpoint_files(model_name_or_path)
        weight_map = get_weight_map(model_files)
        config_data = load_model_config(model_files)

        # auto-detect num_experts_config_key from config.json
        if num_experts_config_key is None:
            num_experts_config_key = get_num_experts_config_key(config_data)

        # detect num_experts_per_tok for routing-floor validation
        num_experts_per_tok = get_num_experts_per_tok(config_data)

        # collect router and expert tensor names. Routers are ordered by natural
        # (numeric) layer order to match the saliency report
        router_names = sorted(
            (n for n in weight_map if router_re.search(n)), key=_natural_sort_key
        )
        expert_names = [n for n in weight_map if expert_re.search(n)]

        if not router_names:
            raise ValueError(f"No tensors matched router_pattern {router_pattern!r}")
        if not expert_names:
            raise ValueError(f"No tensors matched expert_pattern {expert_pattern!r}")

        # associate each expert tensor with a router tensor by longest common
        # dot-separated prefix
        expert_to_router = build_expert_to_router_map(expert_names, router_names)

        # collect auxiliary router tensors (per-expert vectors and expert-index
        # tables) and associate each with its router by common prefix
        aux_names = [n for n in weight_map if aux_re and aux_re.search(n)]
        index_names = [n for n in weight_map if index_re and index_re.search(n)]
        aux_to_router = build_expert_to_router_map(
            aux_names + index_names, router_names
        )

        # a router whose layer carries an expert-index table routes statically
        # (hash routing), so its experts cannot be redistributed after pruning
        hash_routers = {aux_to_router[n] for n in index_names}

        # detect 2D vs 3D and extract expert indices for 2D
        expert_indices, is_3d = _detect_moe_layout(
            expert_names, weight_map, model_files
        )

        # score experts and determine which to retain
        if metric == "magnitude":
            scores_by_router = _magnitude_scores(router_names, weight_map, model_files)
        else:
            scores_by_router = _report_scores(
                router_names, weight_map, model_files, saliency_report_path, metric
            )

        if uniform:
            if hash_routers:
                raise ValueError(
                    f"{len(hash_routers)} layer(s) route statically via an "
                    f"expert-index table ({index_pattern!r}); these experts are "
                    "selected discretely and cannot be redistributed after "
                    "pruning. Use uniform=False to prune only the learned-routing "
                    "layers while protecting the hash layers"
                )
            retained_experts = _compute_retained_experts(scores_by_router, sparsity)
        else:
            retained_experts = _compute_retained_experts_nonuniform(
                scores_by_router,
                sparsity,
                floor=num_experts_per_tok or 1,
                protected=hash_routers,
            )

        if num_experts_per_tok is not None:
            k = min(len(v) for v in retained_experts.values())
            if k < num_experts_per_tok:
                raise ValueError(
                    f"sparsity={sparsity} retains {k} expert(s) per layer but "
                    f"num_experts_per_tok={num_experts_per_tok}; cannot prune "
                    "below the per-token routing floor"
                )

        for rname, retained in retained_experts.items():
            logger.debug(
                f"{rname}: retaining {len(retained)}/{len(scores_by_router[rname])} "
                f"experts {retained}"
            )

        # bypass `__init__`, which is reserved to direct users to this method
        instance = cls.__new__(cls)
        instance.router_pattern = router_re
        instance.expert_pattern = expert_re
        instance.aux_pattern = aux_re
        instance.index_pattern = index_re
        instance.sparsity = sparsity
        instance.uniform = uniform
        instance.metric = metric
        instance.num_experts_config_key = num_experts_config_key
        instance.retained_experts = retained_experts
        instance.expert_to_router = expert_to_router
        instance.aux_to_router = aux_to_router
        instance.hash_routers = hash_routers
        instance.expert_indices = expert_indices
        instance.is_3d = is_3d
        instance.num_experts_per_tok = num_experts_per_tok
        return instance

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}

        for name in list(tensors):
            tensor = tensors[name]

            if self.router_pattern.search(name):
                retained = self.retained_experts[name]
                idx = torch.tensor(retained, dtype=torch.long, device=tensor.device)
                result[name] = tensor[idx].contiguous()

            elif self.expert_pattern.search(name):
                router_name = self.expert_to_router[name]
                retained = self.retained_experts[router_name]

                if self.is_3d:
                    idx = torch.tensor(retained, dtype=torch.long, device=tensor.device)
                    result[name] = tensor.index_select(0, idx).contiguous()
                else:
                    retained_set = set(retained)
                    expert_idx = self.expert_indices[name]
                    if expert_idx not in retained_set:
                        continue
                    new_idx = retained.index(expert_idx)
                    new_name = renumber_expert_name(name, expert_idx, new_idx)
                    result[new_name] = tensor

            elif self.index_pattern is not None and self.index_pattern.search(name):
                # expert-index table (e.g. tid2eid): values are expert indices,
                # remap each from its old index to its new (renumbered) index
                router_name = self.aux_to_router[name]
                retained = self.retained_experts[router_name]
                result[name] = _remap_expert_indices(tensor, retained).contiguous()

            elif self.aux_pattern is not None and self.aux_pattern.search(name):
                # per-expert vector (e.g. routing bias): slice dim 0 by retained
                router_name = self.aux_to_router[name]
                retained = self.retained_experts[router_name]
                idx = torch.tensor(retained, dtype=torch.long, device=tensor.device)
                result[name] = tensor[idx].contiguous()

            else:
                result[name] = tensor

        return result

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.process(tensors)

        for name, tensor in out.items():
            if self.index_pattern is not None and self.index_pattern.search(name):
                # all expert indices must point into the retained range (skip on
                # meta tensors, which carry no values to check)
                k = len(self.retained_experts[self.aux_to_router[name]])
                if not tensor.is_meta and tensor.numel() and int(tensor.max()) >= k:
                    raise ValueError(
                        f"{name}: expert-index table references expert "
                        f"{int(tensor.max())} but only {k} experts are retained"
                    )
                continue

            if self.aux_pattern is not None and self.aux_pattern.search(name):
                # per-expert vector must have one entry per retained expert
                expected = len(self.retained_experts[self.aux_to_router[name]])
                if tensor.shape[0] != expected:
                    raise ValueError(
                        f"{name}: expected {expected} entries after prune, "
                        f"got {tensor.shape[0]}"
                    )
                continue

            if not self.expert_pattern.search(name):
                continue

            if self.is_3d:
                router_name = self.expert_to_router[name]
                expected = len(self.retained_experts[router_name])
                if tensor.shape[0] != expected:
                    raise ValueError(
                        f"{name}: expected first dim {expected} after prune, "
                        f"got {tensor.shape[0]}"
                    )
            else:
                # renumbered names reuse lower indices of the same layer, so
                # they still map to the right router
                router_name = self.expert_to_router[name]
                k = len(self.retained_experts[router_name])
                expert_idx = extract_expert_index(name)
                if expert_idx is None or expert_idx >= k:
                    raise ValueError(
                        f"{name}: expert index {expert_idx} out of range for "
                        f"{k} retained experts, could be orphan"
                    )

        return out

    def update_config(
        self, config: QuantizationConfig | None
    ) -> QuantizationConfig | None:
        if self.uniform:
            return config

        # record how many experts each MoE layer retains, ordered by layer. The
        # config is created if the checkpoint is not otherwise quantized
        if config is None:
            config = QuantizationConfig(
                config_groups={}, format=CompressionFormat.dense.value
            )
        else:
            config = config.model_copy(deep=True)

        num_experts = [len(retained) for retained in self.retained_experts.values()]
        config.layer_overrides[self.num_experts_config_key] = num_experts
        logger.info(
            f"quantization_config.layer_overrides: added {self.num_experts_config_key} "
            f"for {len(num_experts)} layers"
        )
        return config

    def update_model_config(self, model_config: dict) -> dict:
        # non-uniform expert counts are recorded in the quantization config
        if not self.uniform:
            return model_config

        counts = {len(val) for val in self.retained_experts.values()}
        if len(counts) != 1:
            raise ValueError(
                f"non-uniform retained expert counts {counts}; "
                "cannot write a single num_experts"
            )
        num_experts = counts.pop()
        _set_nested_key(model_config, self.num_experts_config_key, num_experts)
        return model_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_nested_key(config_data: dict, key: str, value: int):
    if key in config_data:
        old = config_data[key]
        config_data[key] = value
        logger.info(f"config.json: {key}: {old} -> {value}")
    elif "text_config" in config_data and key in config_data["text_config"]:
        old = config_data["text_config"][key]
        config_data["text_config"][key] = value
        logger.info(f"config.json text_config.{key}: {old} -> {value}")
    else:
        config_data[key] = value
        logger.info(f"config.json: added {key} = {value}")


def _detect_moe_layout(
    expert_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> tuple[dict[str, int], bool]:
    """
    Detect whether experts are 2D (per-expert tensors) or 3D (stacked).
    For 2D, also extract the expert index from each tensor name.

    Returns (expert_indices, is_3d).
    """
    sample_name = expert_names[0]
    sample_shard = weight_map[sample_name]
    sample_path = model_files[sample_shard]

    with safe_open(sample_path, framework="pt") as f:
        sample_shape = f.get_slice(sample_name).get_shape()

    if len(sample_shape) >= 3:
        return {}, True

    # 2D: extract expert index from each tensor name
    expert_indices: dict[str, int] = {}
    for name in expert_names:
        idx = extract_expert_index(name)
        if idx is None:
            raise ValueError(
                f"Expert tensor {name} appears to be 2D but could not extract "
                "expert index from tensor name. Expected a numeric segment "
                "following an 'expert'-containing segment (e.g. experts.3.weight)"
            )
        expert_indices[name] = idx

    return expert_indices, False


def _natural_sort_key(name: str):
    """Sort key that orders embedded integers numerically (``layers.2`` before
    ``layers.10``)."""
    return [
        int(token) if token.isdigit() else token for token in re.split(r"(\d+)", name)
    ]


def _group_by_file(
    router_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> dict[str, list[str]]:
    """Group router names by source file to minimize I/O."""
    routers_by_file: dict[str, list[str]] = {}
    for rname in router_names:
        routers_by_file.setdefault(model_files[weight_map[rname]], []).append(rname)
    return routers_by_file


def _magnitude_scores(
    router_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> dict[str, torch.Tensor]:
    """
    Load each router weight and score experts by L1 magnitude
    (``weight.abs().sum(dim=-1)``). Absolute values are used so positive and
    negative router weights do not cancel.
    """
    scores: dict[str, torch.Tensor] = {}
    for path, names in _group_by_file(router_names, weight_map, model_files).items():
        with safe_open(path, framework="pt") as f:
            for rname in names:
                weight = f.get_tensor(rname)
                scores[rname] = weight.abs().sum(dim=-1).to(torch.float64)

    # preserve router (layer) order
    return {rname: scores[rname] for rname in router_names}


def _report_scores(
    router_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
    report_path: str | os.PathLike,
    metric: str,
) -> dict[str, torch.Tensor]:
    """
    Read per-expert scores for ``metric`` from a saliency report written by
    ``REAPPruningModifier``. The report holds one list of per-expert values per
    MoE layer, ordered by layer, which are matched to ``router_names`` in order.
    Only router shapes are read from the checkpoint, to validate the report.
    """
    with open(report_path, "r") as file:
        report = json.load(file)

    key = "saliency" if metric == "layerwise_saliency" else metric
    if key not in report:
        raise ValueError(
            f"Report {report_path} does not contain {key!r} values; "
            f"available keys: {sorted(report)}"
        )
    layer_values = report[key]

    if len(layer_values) != len(router_names):
        raise ValueError(
            f"Report {report_path} contains {len(layer_values)} layer(s), but "
            f"{len(router_names)} router weight(s) were found in the checkpoint. "
            "The report must be generated from the same model"
        )

    num_experts: dict[str, int] = {}
    for path, names in _group_by_file(router_names, weight_map, model_files).items():
        with safe_open(path, framework="pt") as f:
            for rname in names:
                num_experts[rname] = f.get_slice(rname).get_shape()[0]

    scores: dict[str, torch.Tensor] = {}
    for rname, values in zip(router_names, layer_values):
        if len(values) != num_experts[rname]:
            raise ValueError(
                f"Report layer for {rname} has {len(values)} experts, but the "
                f"checkpoint has {num_experts[rname]}. The report must be "
                "generated from the same model"
            )
        layer_scores = torch.tensor(values, dtype=torch.float64)
        if metric == "layerwise_saliency":
            max_score = layer_scores.max()
            if max_score > 0:
                layer_scores = layer_scores / max_score
        scores[rname] = layer_scores

    return scores


def _compute_retained_experts(
    scores_by_router: dict[str, torch.Tensor],
    sparsity: float,
) -> dict[str, list[int]]:
    """
    Return the retained expert indices (sorted) per router, keeping the highest
    scores. The number kept is derived per router as
    ``max(1, num_experts - round(sparsity * num_experts))``.
    """
    retained: dict[str, list[int]] = {}
    for rname, scores in scores_by_router.items():
        num_experts = scores.shape[0]
        num_prune = round(sparsity * num_experts)
        k = max(1, num_experts - num_prune)
        _, top_indices = torch.topk(scores, k)
        retained[rname] = sorted(top_indices.tolist())
    return retained


def _compute_retained_experts_nonuniform(
    scores_by_router: dict[str, torch.Tensor],
    sparsity: float,
    floor: int = 1,
    protected: set[str] | None = None,
) -> dict[str, list[int]]:
    """
    Return the retained expert indices (sorted) per router after pruning the
    ``round(sparsity * total_experts)`` lowest-scoring experts across all
    routers. No router is taken below ``floor`` retained experts (the router's
    per-token selection count), so the budget may not be fully spent.

    Routers in ``protected`` (e.g. statically-routed hash layers) keep all of
    their experts and are excluded from both the pruning budget and the
    candidate pool.
    """
    protected = protected or set()

    # protected routers do not contribute to the budget nor the candidate pool
    total_experts = sum(
        scores.shape[0]
        for rname, scores in scores_by_router.items()
        if rname not in protected
    )
    budget = round(sparsity * total_experts)

    # (score, router, expert), ascending, so the lowest scores are pruned first
    candidates = sorted(
        (score, rname, expert_idx)
        for rname, scores in scores_by_router.items()
        if rname not in protected
        for expert_idx, score in enumerate(scores.tolist())
    )

    pruned: dict[str, set[int]] = {rname: set() for rname in scores_by_router}
    num_pruned = 0
    for _, rname, expert_idx in candidates:
        if num_pruned >= budget:
            break
        num_experts = scores_by_router[rname].shape[0]
        if num_experts - len(pruned[rname]) <= floor:
            continue  # keep at least `floor` experts in this layer
        pruned[rname].add(expert_idx)
        num_pruned += 1

    if num_pruned < budget:
        logger.warning(
            f"Only pruned {num_pruned}/{budget} experts: every layer has reached "
            f"the minimum of {floor} retained experts"
        )

    return {
        rname: [i for i in range(scores.shape[0]) if i not in pruned[rname]]
        for rname, scores in scores_by_router.items()
    }


def _remap_expert_indices(
    table: torch.Tensor, retained: list[int]
) -> torch.Tensor:
    """
    Remap the values of an expert-index table (e.g. ``tid2eid``) from old expert
    indices to their new (renumbered) positions. ``retained[new] == old``, so
    the inverse ``old -> new`` map is built and applied element-wise.

    References to pruned experts have no valid new index. They are only reachable
    if a statically-routed (hash) layer had experts pruned; such layers are
    protected upstream, so this is an internal invariant. If one is found, it is
    an error rather than a silent reassignment.
    """
    # `validate` runs on meta tensors (no data); the remap needs real values, so
    # return a same-shape placeholder and defer correctness to `process`
    if table.is_meta:
        return torch.empty_like(table)

    max_old = max(int(table.max()) if table.numel() else 0, max(retained))
    old_to_new = torch.full(
        (max_old + 1,), -1, dtype=table.dtype, device=table.device
    )
    for new_idx, old_idx in enumerate(retained):
        old_to_new[old_idx] = new_idx

    remapped = old_to_new[table.long()].to(table.dtype)
    if bool((remapped < 0).any()):
        raise ValueError(
            "expert-index table references a pruned expert; statically-routed "
            "layers must retain all experts"
        )
    return remapped
