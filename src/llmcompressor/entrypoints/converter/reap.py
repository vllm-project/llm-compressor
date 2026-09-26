# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
REAP (Router-weighted Expert Activation Pruning) checkpoint converter.

Prunes experts from a MoE safetensors checkpoint using the per-expert saliency
report produced by ``REAPPruningModifier`` (see its ``report_path`` argument).
The checkpoint is never fully loaded into memory: the pruning decision is made
from the report and cheap tensor metadata, then expert/router tensors are sliced
shard by shard as the conversion pipeline streams them.

See: https://arxiv.org/abs/2510.13999
"""

from __future__ import annotations

import json
import os
import re

import torch
from compressed_tensors.base import QUANTIZATION_CONFIG_NAME
from compressed_tensors.entrypoints.convert.converters import Converter
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

__all__ = ["REAPExpertPruner"]

# metrics the user may request; ``saliency_norm`` is derived from ``saliency``
# rather than stored directly in the report
METRICS = ("saliency", "saliency_norm", "count")

# config.json key under which per-layer expert counts are recorded for
# non-uniform pruning
NUM_EXPERTS_PER_LAYER_KEY = "num_experts_per_layer"

# module segments that identify an MoE router (a.k.a. gate) module
_ROUTER_SEGMENTS = ("gate", "router", "wg")


class REAPExpertPruner(Converter):
    """
    Prune MoE experts from a checkpoint according to a REAP saliency report.

    Experts are ranked per layer by the chosen ``metric``. With ``uniform`` the
    same number of experts is dropped from every layer; otherwise a single global
    budget of experts is dropped wherever the (per-layer normalized) score is
    lowest. Pruning slices the router weight, removes/renumbers the pruned expert
    tensors, and records the new expert counts in the model config.

    Build instances with :meth:`from_pretrained`, which reads the report and
    checkpoint metadata and decides which experts to retain.

    :param model_stub: HuggingFace stub or local path of the checkpoint to prune
    :param report_path: path to the ``.json`` report written by
        ``REAPPruningModifier``
    :param metric: report metric used to rank experts (see :meth:`from_pretrained`)
    :param uniform: whether the same number of experts is dropped per layer
    :param sparsity: fraction of experts to prune, in [0, 1)
    :param num_experts_config_key: config.json key holding the expert count
    :param retained_experts: mapping of router weight name -> sorted list of
        retained expert indices. Built by :meth:`from_pretrained`
    :param expert_to_router: mapping of expert tensor name -> associated router
        weight name. Built by :meth:`from_pretrained`
    :param expert_indices: for 2D (per-expert tensor) checkpoints, mapping of
        tensor name -> expert index. Empty for 3D (stacked) checkpoints
    :param is_3d: whether expert weights are 3D stacked tensors
    """

    def __init__(
        self,
        model_stub: str | os.PathLike,
        report_path: str | os.PathLike,
        metric: str,
        uniform: bool,
        sparsity: float,
        num_experts_config_key: str,
        retained_experts: dict[str, list[int]],
        expert_to_router: dict[str, str],
        expert_indices: dict[str, int],
        is_3d: bool,
    ):
        self.model_stub = model_stub
        self.report_path = report_path
        self.metric = metric
        self.uniform = uniform
        self.sparsity = sparsity
        self.num_experts_config_key = num_experts_config_key
        self.retained_experts = retained_experts
        self.expert_to_router = expert_to_router
        self.expert_indices = expert_indices
        self.is_3d = is_3d

        # a router "family" is the router module (gate) and any sibling tensors
        # (e.g. bias / correction terms) that are indexed by expert. They all
        # share the module prefix and are sliced on dim 0 together.
        self._retained_by_prefix = {
            _module_prefix(name): retained
            for name, retained in retained_experts.items()
        }

    @classmethod
    def from_pretrained(
        cls,
        model_stub: str | os.PathLike,
        report_path: str | os.PathLike,
        metric: str = "saliency_norm",
        uniform: bool = False,
        sparsity: float = 0.0,
        num_experts_config_key: str | None = None,
    ) -> REAPExpertPruner:
        """
        Build the converter by reading the REAP report and checkpoint metadata
        and deciding which experts to retain.

        :param model_stub: HuggingFace stub or local checkpoint path
        :param report_path: path to the ``.json`` report written by
            ``REAPPruningModifier``
        :param metric: report metric used to rank experts:
            - ``"saliency_norm"``: saliency divided by the highest saliency value
              of each layer (default). Normalizing per layer makes saliency
              comparable across layers, which matters for non-uniform pruning.
            - ``"saliency"``: raw per-expert saliency
            - ``"count"``: per-expert token count
        :param uniform: if True, prune the same number of experts from every
            layer and record the new count in the ``num_experts`` config key. If
            False, prune a single global budget of experts wherever the score is
            lowest and, instead of touching ``num_experts``, record the per-layer
            expert counts in the compressed-tensors config
        :param sparsity: fraction of experts to prune. ``0.1`` removes 10% of all
            experts (uniformly or non-uniformly, per ``uniform``)
        :param num_experts_config_key: config.json key holding the expert count.
            Auto-detected from config.json when None
        """
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

        with open(report_path, "r") as file:
            report = json.load(file)
        metric_values = _metric_values(report, metric, report_path)

        model_files = get_checkpoint_files(model_stub)
        weight_map = get_weight_map(model_files)
        config_data = load_model_config(model_files)

        if num_experts_config_key is None:
            num_experts_config_key = get_num_experts_config_key(config_data)

        num_experts_per_tok = get_num_experts_per_tok(config_data)
        if num_experts_per_tok is None:
            raise ValueError(
                f"Could not determine num_experts_per_tok from the config of "
                f"{model_stub}. It is required to keep at least that many experts "
                "per layer so the router can still select its top-k experts"
            )

        # order routers by natural (numeric) layer order to match the report,
        # which is written in the model's module-registration order
        router_names = sorted(
            (name for name in weight_map if _is_router_weight(name)),
            key=_natural_sort_key,
        )
        if not router_names:
            raise ValueError(
                f"Could not find any router weights in {model_stub}. REAP expert "
                "pruning requires a MoE checkpoint with router (gate) weights"
            )
        if len(router_names) != len(metric_values):
            raise ValueError(
                f"Report at {report_path} contains saliency for "
                f"{len(metric_values)} layer(s), but {len(router_names)} router "
                "weight(s) were found in the checkpoint. The report must be "
                "generated from the same model"
            )

        num_experts_by_router = _num_experts_by_router(
            router_names, weight_map, model_files
        )
        scores_by_router: dict[str, torch.Tensor] = {}
        for router_name, values in zip(router_names, metric_values):
            n = num_experts_by_router[router_name]
            if len(values) != n:
                raise ValueError(
                    f"Report layer for {router_name} has {len(values)} experts, "
                    f"but the checkpoint has {n}. The report must be generated "
                    "from the same model"
                )
            scores_by_router[router_name] = torch.tensor(values, dtype=torch.float64)

        # Never take a layer below the router's per-token floor: the router must
        # still be able to select ``num_experts_per_tok`` experts. For uniform
        # pruning this clamps the per-layer count; for non-uniform pruning it
        # bounds where the global drop budget can be spent.
        retained = _compute_retained_experts(
            scores_by_router=scores_by_router,
            num_experts_by_router=num_experts_by_router,
            sparsity=sparsity,
            uniform=uniform,
            floor=num_experts_per_tok,
        )

        expert_names = [
            name for name in weight_map if extract_expert_index(name) is not None
        ]
        expert_names += [
            name
            for name in weight_map
            if _looks_like_expert(name) and extract_expert_index(name) is None
        ]
        if not expert_names:
            raise ValueError(
                f"Could not find any expert weights in {model_stub}. REAP expert "
                "pruning requires a MoE checkpoint with expert weights"
            )
        expert_to_router = build_expert_to_router_map(expert_names, router_names)
        expert_indices, is_3d = _detect_moe_layout(
            expert_names, weight_map, model_files
        )

        for router_name in router_names:
            n = num_experts_by_router[router_name]
            logger.debug(
                f"{router_name}: retaining {len(retained[router_name])}/{n} experts"
            )

        return cls(
            model_stub=model_stub,
            report_path=report_path,
            metric=metric,
            uniform=uniform,
            sparsity=sparsity,
            num_experts_config_key=num_experts_config_key,
            retained_experts=retained,
            expert_to_router=expert_to_router,
            expert_indices=expert_indices,
            is_3d=is_3d,
        )

    def process(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}

        for name in list(tensors):
            tensor = tensors[name]

            retained = self._retained_by_prefix.get(_module_prefix(name))
            if retained is not None and name not in self.expert_to_router:
                # router-family tensor (gate weight/bias): slice experts on dim 0
                idx = torch.tensor(retained, dtype=torch.long, device=tensor.device)
                result[name] = tensor.index_select(0, idx).contiguous()

            elif name in self.expert_to_router:  # expert weight
                router_name = self.expert_to_router[name]
                retained = self.retained_experts[router_name]

                if self.is_3d:
                    idx = torch.tensor(retained, dtype=torch.long, device=tensor.device)
                    result[name] = tensor.index_select(0, idx).contiguous()
                else:
                    expert_idx = self.expert_indices[name]
                    if expert_idx not in retained:
                        continue  # pruned expert: drop the tensor entirely
                    new_idx = retained.index(expert_idx)
                    new_name = renumber_expert_name(name, expert_idx, new_idx)
                    result[new_name] = tensor

            else:
                result[name] = tensor

        return result

    def validate(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.process(tensors)

        for name, tensor in out.items():
            if name not in self.expert_to_router:
                continue
            if self.is_3d:
                expected = len(self.retained_experts[self.expert_to_router[name]])
                if tensor.shape[0] != expected:
                    raise ValueError(
                        f"{name}: expected first dim {expected} after prune, "
                        f"got {tensor.shape[0]}"
                    )
            else:
                router_name = self.expert_to_router[name]
                k = len(self.retained_experts[router_name])
                expert_idx = extract_expert_index(name)
                if expert_idx is None or expert_idx >= k:
                    raise ValueError(
                        f"{name}: expert index {expert_idx} out of range for "
                        f"{k} retained experts, could be an orphan"
                    )
        return out

    def update_config(self, config):
        # pruning does not change quantization; pass any existing config through
        return config

    def update_model_config(self, model_config: dict) -> dict:
        if self.uniform:
            counts = {len(val) for val in self.retained_experts.values()}
            if len(counts) != 1:
                raise ValueError(
                    f"uniform pruning must retain the same number of experts in "
                    f"every layer, got counts {sorted(counts)}"
                )
            _set_nested_key(model_config, self.num_experts_config_key, counts.pop())
        else:
            # record how many experts each layer keeps in the compressed-tensors
            # config, creating it if the checkpoint is not otherwise quantized
            experts_per_layer = {
                _module_prefix(_module_prefix(name)): len(retained)
                for name, retained in self.retained_experts.items()
            }
            quant_config = model_config.setdefault(QUANTIZATION_CONFIG_NAME, {})
            quant_config[NUM_EXPERTS_PER_LAYER_KEY] = experts_per_layer
            logger.info(
                f"config.json: added {QUANTIZATION_CONFIG_NAME}."
                f"{NUM_EXPERTS_PER_LAYER_KEY} for {len(experts_per_layer)} layers"
            )
        return model_config

    def get_dependencies(self, weight_name: str) -> set[str]:
        return set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metric_values(report: dict, metric: str, report_path) -> list[list[float]]:
    """Return the per-layer, per-expert scores for ``metric`` from the report."""
    if metric == "saliency_norm":
        if "saliency" not in report:
            raise ValueError(
                f"Report {report_path} has no 'saliency' values, which are "
                "required to compute 'saliency_norm'"
            )
        normalized: list[list[float]] = []
        for layer in report["saliency"]:
            scores = torch.tensor(layer, dtype=torch.float64)
            max_score = scores.max()
            if max_score > 0:
                scores = scores / max_score
            normalized.append(scores.tolist())
        return normalized

    if metric not in report:
        raise ValueError(
            f"Report {report_path} does not contain metric {metric!r}; "
            f"available metrics: {tuple(k for k in METRICS if k in report)}"
        )
    return report[metric]


def _module_prefix(name: str) -> str:
    """Drop the final dot-segment: ``a.b.weight`` -> ``a.b``."""
    return name.rpartition(".")[0]


def _natural_sort_key(name: str):
    """Sort key that orders embedded integers numerically (``layers.2`` before
    ``layers.10``)."""
    return [
        int(token) if token.isdigit() else token for token in re.split(r"(\d+)", name)
    ]


def _is_router_weight(name: str) -> bool:
    parts = name.split(".")
    return len(parts) >= 2 and parts[-1] == "weight" and parts[-2] in _ROUTER_SEGMENTS


def _looks_like_expert(name: str) -> bool:
    return any("expert" in part.lower() for part in name.split("."))


def _num_experts_by_router(
    router_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> dict[str, int]:
    """Number of experts per router, read from the router weight's first dim."""
    routers_by_file: dict[str, list[str]] = {}
    for name in router_names:
        routers_by_file.setdefault(model_files[weight_map[name]], []).append(name)

    num_experts: dict[str, int] = {}
    for path, names in routers_by_file.items():
        with safe_open(path, framework="pt") as f:
            for name in names:
                num_experts[name] = f.get_slice(name).get_shape()[0]
    return num_experts


def _detect_moe_layout(
    expert_names: list[str],
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> tuple[dict[str, int], bool]:
    """
    Detect whether experts are 2D (per-expert tensors) or 3D (stacked). For 2D,
    also extract the expert index from each tensor name.

    Returns ``(expert_indices, is_3d)``.
    """
    indices = {name: extract_expert_index(name) for name in expert_names}

    if all(idx is None for idx in indices.values()):
        return {}, True
    if any(idx is None for idx in indices.values()):
        raise ValueError(
            "Checkpoint mixes stacked (3D) and per-expert (2D) expert tensors; "
            "this is not supported"
        )
    return indices, False


def _compute_retained_experts(
    scores_by_router: dict[str, torch.Tensor],
    num_experts_by_router: dict[str, int],
    sparsity: float,
    uniform: bool,
    floor: int,
) -> dict[str, list[int]]:
    """
    Select which experts to retain per router.

    With ``uniform``, ``round(sparsity * num_experts)`` experts are dropped from
    every layer. Otherwise a single global budget of ``round(sparsity *
    total_experts)`` experts is dropped wherever the score is lowest. In both
    cases no layer is taken below ``floor`` retained experts (the router's
    per-token selection count).
    """
    retained: dict[str, list[int]] = {}

    if uniform:
        for router_name, scores in scores_by_router.items():
            n = num_experts_by_router[router_name]
            num_prune = round(sparsity * n)
            k = max(floor, n - num_prune)
            _, top_indices = torch.topk(scores, k)
            retained[router_name] = sorted(top_indices.tolist())
        return retained

    total_experts = sum(num_experts_by_router.values())
    budget = round(sparsity * total_experts)

    # (score, router, expert), ascending, so the least salient are dropped first
    candidates = sorted(
        (score, router_name, expert_idx)
        for router_name, scores in scores_by_router.items()
        for expert_idx, score in enumerate(scores.tolist())
    )

    dropped: dict[str, set[int]] = {name: set() for name in scores_by_router}
    num_dropped = 0
    for _, router_name, expert_idx in candidates:
        if num_dropped >= budget:
            break
        n = num_experts_by_router[router_name]
        if n - len(dropped[router_name]) <= floor:
            continue  # keep at least `floor` experts in this layer
        dropped[router_name].add(expert_idx)
        num_dropped += 1

    for router_name in scores_by_router:
        n = num_experts_by_router[router_name]
        drop = dropped[router_name]
        retained[router_name] = [i for i in range(n) if i not in drop]
    return retained


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
