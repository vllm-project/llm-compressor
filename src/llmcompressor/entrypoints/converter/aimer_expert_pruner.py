# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
AIMER (Absolute mean over root mean square IMportance for Expert Ranking) expert
pruning for MoE checkpoints.

See: https://arxiv.org/abs/2603.18492
"""

from __future__ import annotations

import os

import torch
from safetensors import safe_open

from llmcompressor.entrypoints.converter.expert_pruner import (
    ExpertPruner,
    _group_by_file,
)

__all__ = ["AIMERExpertPruner"]

# safetensors dtypes of floating-point weights, which AIMER is defined over
FLOAT_DTYPES = ("F16", "BF16", "F32", "F64")

# default expert_pattern: per-expert projections (``...experts.3.gate_proj``),
# stacked split projections (``...experts.gate_proj``) and stacked fused
# projections (``...experts.gate_up_proj``)
DEFAULT_EXPERT_PATTERN = (
    r"mlp\.experts\.(?:\d+\.)?(?:gate_up_proj|gate_proj|up_proj|down_proj)"
)


class AIMERExpertPruner(ExpertPruner):
    r"""
    Prune MoE experts by their AIMER score, which is computed from the expert
    weights alone: no calibration data is needed and the model is never loaded.
    For each expert, over all of its projection weights ``w`` (the gate, up
    and down projections)

        ``AIMER = sum(|w|) / sqrt(N * sum(w^2))``

    the mean absolute weight divided by the root mean square weight, where ``N``
    is the expert's total number of weights. Following the paper, the experts
    with the highest scores are pruned.

    Of the tensors matched by ``expert_pattern``, only projection weights are
    scored: ``<name>_proj.weight`` tensors (one per expert) and ``<name>_proj``
    parameters (stacked), e.g. ``gate_proj``, ``up_proj``, ``down_proj`` or
    fused ``gate_up_proj``, as in the Hugging Face MoE implementations. Every
    expert must match a complete set of projections, either ``gate_proj``,
    ``up_proj`` and ``down_proj`` or fused ``gate_up_proj`` and ``down_proj``:
    a partial set raises, rather than silently scoring a different statistic.
    Biases, scales, weights of modules nested inside a projection and other
    auxiliary tensors are not scored, but are pruned with their expert when
    matched. Every tensor matched by ``expert_pattern`` must have dtype F16,
    BF16, F32 or F64. This rejects matched packed-integer and FP8
    representations, but does not detect all forms of quantization: the
    intended workflow is to prune experts before quantizing the checkpoint.

    Pruning itself is done by :class:`ExpertPruner`; see it for uniform and
    non-uniform pruning, group-limited routers and the supported checkpoint
    layouts. The paper evaluates uniform pruning only.

    The default ``expert_pattern`` matches per-expert, stacked split and
    stacked fused gate/up/down projections; pass a custom one for other
    layouts.

    Example::

        pruner = AIMERExpertPruner.from_pretrained(
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            sparsity=0.25,
            expert_pattern=r"mlp\.experts\.\d+\.(gate|up|down)_proj",
        )
        convert_checkpoint(
            model_stub="Qwen/Qwen3-30B-A3B-Instruct-2507",
            save_directory="Qwen3-30B-A3B-Instruct-2507-AIMER-25",
            converter=pruner,
        )
    """

    metrics = ("aimer",)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        sparsity: float,
        **kwargs,
    ) -> AIMERExpertPruner:
        """
        Build the converter by scoring the checkpoint's experts and determining
        which to retain.

        :param model_name_or_path: HuggingFace stub or local checkpoint path
        :param sparsity: fraction of experts to prune, in [0, 1]
        :param kwargs: any other argument of :meth:`ExpertPruner.from_pretrained`
            except ``metric`` and ``saliency_report_path``, e.g. ``uniform`` or
            ``expert_pattern``. ``expert_pattern`` defaults to a pattern matching
            per-expert, stacked split and stacked fused gate/up/down projections
        """
        for unsupported in ("metric", "saliency_report_path"):
            if unsupported in kwargs:
                raise TypeError(
                    f"AIMERExpertPruner.from_pretrained does not accept "
                    f"{unsupported!r}: experts are scored by AIMER"
                )
        kwargs.setdefault("expert_pattern", DEFAULT_EXPERT_PATTERN)
        return super().from_pretrained(
            model_name_or_path, sparsity, metric="aimer", **kwargs
        )

    @classmethod
    def _score_experts(
        cls,
        metric: str,
        router_names: list[str],
        expert_to_router: dict[str, str],
        expert_indices: dict[str, int],
        is_3d: bool,
        weight_map: dict[str, str],
        model_files: dict[str, str],
        saliency_report_path: str | os.PathLike | None,
    ) -> dict[str, torch.Tensor]:
        # `metric` and `saliency_report_path` are unused: experts are always
        # scored by AIMER. They are kept in the signature to match the base
        # class, which passes both positionally
        return _aimer_scores(
            router_names,
            expert_to_router,
            expert_indices,
            is_3d,
            weight_map,
            model_files,
        )


def _aimer_scores(
    router_names: list[str],
    expert_to_router: dict[str, str],
    expert_indices: dict[str, int],
    is_3d: bool,
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> dict[str, torch.Tensor]:
    """
    Score each expert by its negated AIMER score, so that the highest-AIMER
    experts are the lowest-scoring ones, which ``ExpertPruner`` prunes first.

    The absolute and squared sums of each expert's projection weights are
    accumulated in float64, and their element counts as integers, across all of
    its tensors, reading one expert at a time so that stacked (3D) tensors are
    never loaded whole. The sums are kept relative to a running per-expert
    scale (the largest absolute weight seen so far), which preserves the
    scale-invariant score for experts whose weights are uniformly tiny (whose
    squared weights would underflow to zero) or uniformly huge (whose squared
    weights would overflow to infinity).
    """
    num_experts: dict[str, int] = {}
    for path, names in _group_by_file(router_names, weight_map, model_files).items():
        with safe_open(path, framework="pt") as f:
            for rname in names:
                num_experts[rname] = f.get_slice(rname).get_shape()[0]

    abs_sum = {r: torch.zeros(n, dtype=torch.float64) for r, n in num_experts.items()}
    sq_sum = {r: torch.zeros(n, dtype=torch.float64) for r, n in num_experts.items()}
    scale = {r: torch.zeros(n, dtype=torch.float64) for r, n in num_experts.items()}
    numel = {r: torch.zeros(n, dtype=torch.int64) for r, n in num_experts.items()}
    # projection names scored for each expert, which must form a complete set
    projections = {r: [set() for _ in range(n)] for r, n in num_experts.items()}
    non_finite = {r: torch.zeros(n, dtype=torch.bool) for r, n in num_experts.items()}

    expert_names = list(expert_to_router)
    for path, names in _group_by_file(expert_names, weight_map, model_files).items():
        with safe_open(path, framework="pt") as f:
            for name in names:
                tensor = f.get_slice(name)
                # every matched expert tensor is checked, not only the scored
                # ones, so that e.g. a packed `.weight_packed` projection is
                # rejected instead of being silently left out of the score
                if tensor.get_dtype() not in FLOAT_DTYPES:
                    raise ValueError(
                        f"{name} has dtype {tensor.get_dtype()}, but AIMER requires "
                        "every matched expert tensor to have one of these dtypes: "
                        f"{FLOAT_DTYPES}. Prune experts before quantizing the "
                        "checkpoint"
                    )
                # the paper scores projection weights only: biases, scales and
                # other auxiliary tensors are pruned with their expert, but
                # not scored
                proj = _projection_name(name)
                if proj is None:
                    continue

                rname = expert_to_router[name]
                if is_3d:
                    experts = range(tensor.get_shape()[0])
                else:
                    experts = [expert_indices[name]]

                for expert in experts:
                    projections[rname][expert].add(proj)
                    weight = tensor[expert] if is_3d else tensor[:]
                    weight = weight.to(torch.float64)
                    numel[rname][expert] += weight.numel()
                    if weight.numel() == 0:
                        continue

                    max_abs = weight.abs().max()
                    if not bool(max_abs.isfinite()):
                        non_finite[rname][expert] = True
                    elif max_abs > 0:
                        # rescale the running sums when a larger weight
                        # appears, so that the squared sum neither underflows
                        # nor overflows
                        if max_abs > scale[rname][expert]:
                            factor = scale[rname][expert] / max_abs
                            abs_sum[rname][expert] *= factor
                            sq_sum[rname][expert] *= factor * factor
                            scale[rname][expert] = max_abs
                        scaled = weight / scale[rname][expert]
                        abs_sum[rname][expert] += scaled.abs().sum()
                        sq_sum[rname][expert] += scaled.square().sum()

    scores: dict[str, torch.Tensor] = {}
    for rname in router_names:
        missing = (numel[rname] == 0).nonzero().flatten().tolist()
        if missing:
            raise ValueError(
                f"{rname}: no projection weights matched expert_pattern for "
                f"experts {missing}"
            )
        bad = non_finite[rname].nonzero().flatten().tolist()
        if bad:
            raise ValueError(f"{rname}: experts {bad} have non-finite weights")
        incomplete = [
            expert
            for expert, projs in enumerate(projections[rname])
            if not _is_complete(projs)
        ]
        if incomplete:
            found = sorted(set().union(*projections[rname]))
            raise ValueError(
                f"{rname}: experts {incomplete} matched only the projections "
                f"{found}. AIMER scores the gate, up and down projections of "
                "each expert, so expert_pattern must match all of them, e.g. "
                r"mlp\.experts\.\d+\.(gate|up|down)_proj or "
                r"mlp\.experts\.(gate_up_proj|down_proj)"
            )

        aimer = abs_sum[rname] / (numel[rname].double().sqrt() * sq_sum[rname].sqrt())
        # an all-zero expert has an undefined (0/0) score, which the reference
        # implementation scores as 0
        scores[rname] = -torch.where(sq_sum[rname] > 0, aimer, 0.0)

    return scores


def _projection_name(name: str) -> str | None:
    """Base name of the projection a matched expert tensor belongs to, e.g.
    ``gate_proj`` for ``...experts.3.gate_proj.weight`` or
    ``...experts.gate_up_proj``, or ``None`` if the tensor is not a projection
    weight: a bias or scale (``.bias``, ``_proj_bias``, ``.weight_scale``) or
    the weight of a module nested inside a projection
    (``...down_proj.input_norm.weight``)."""
    projection = name.removesuffix(".weight").rsplit(".", 1)[-1]
    return projection if projection.endswith("_proj") else None


def _is_complete(projections: set[str]) -> bool:
    """Whether a set of projection names covers the projections AIMER scores:
    the split ``gate_proj``, ``up_proj`` and ``down_proj``, or the fused
    ``gate_up_proj`` and ``down_proj``."""
    return "down_proj" in projections and (
        "gate_up_proj" in projections or {"gate_proj", "up_proj"} <= projections
    )
