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


class AIMERExpertPruner(ExpertPruner):
    """
    Prune MoE experts by their AIMER score, which is computed from the expert
    weights alone: no calibration data is needed and the model is never loaded.
    For each expert, over all of its projection weights ``w`` (e.g. the gate, up
    and down projections)

        ``AIMER = sum(|w|) / sqrt(N * sum(w^2))``

    the mean absolute weight divided by the root mean square weight, where ``N``
    is the expert's total number of weights. Following the paper, the experts
    with the highest scores are pruned.

    Of the tensors matched by ``expert_pattern``, only projection weights are
    scored: per-expert ``<proj>.weight`` tensors and stacked ``<name>_proj``
    parameters, as in the Hugging Face MoE implementations. Projections named
    otherwise are not scored. Biases and auxiliary tensors are not scored, but
    are pruned with their expert. Every tensor matched by ``expert_pattern`` must
    have dtype F16, BF16, F32 or F64. This rejects matched packed-integer and FP8
    representations, but does not detect all forms of quantization: the intended
    workflow is to prune experts before quantizing.

    Pruning itself is done by :class:`ExpertPruner`; see it for uniform and
    non-uniform pruning, group-limited routers and the supported checkpoint
    layouts. The paper evaluates uniform pruning only.

    Example::

        pruner = AIMERExpertPruner.from_pretrained(
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            sparsity=0.25,
            expert_pattern=r"mlp\\.experts\\.\\d+\\.(gate|up|down)_proj",
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
            ``expert_pattern``
        """
        for unsupported in ("metric", "saliency_report_path"):
            if unsupported in kwargs:
                raise TypeError(
                    f"AIMERExpertPruner.from_pretrained does not accept "
                    f"{unsupported!r}: experts are scored by AIMER"
                )
        return super().from_pretrained(
            model_name_or_path, sparsity, metric="aimer", **kwargs
        )

    @classmethod
    def _score_experts(
        cls,
        router_names: list[str],
        expert_to_router: dict[str, str],
        expert_indices: dict[str, int],
        is_3d: bool,
        weight_map: dict[str, str],
        model_files: dict[str, str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
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
    never loaded whole.
    """
    num_experts: dict[str, int] = {}
    for path, names in _group_by_file(router_names, weight_map, model_files).items():
        with safe_open(path, framework="pt") as f:
            for rname in names:
                num_experts[rname] = f.get_slice(rname).get_shape()[0]

    abs_sum = {r: torch.zeros(n, dtype=torch.float64) for r, n in num_experts.items()}
    sq_sum = {r: torch.zeros(n, dtype=torch.float64) for r, n in num_experts.items()}
    numel = {r: torch.zeros(n, dtype=torch.int64) for r, n in num_experts.items()}

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
                # the paper scores projection weights only: biases and auxiliary
                # tensors are pruned with their expert, but not scored
                if not _is_projection_weight(name):
                    continue

                rname = expert_to_router[name]
                if is_3d:
                    experts = range(tensor.get_shape()[0])
                else:
                    experts = [expert_indices[name]]

                for expert in experts:
                    weight = tensor[expert] if is_3d else tensor[:]
                    weight = weight.to(torch.float64)
                    abs_sum[rname][expert] += weight.abs().sum()
                    sq_sum[rname][expert] += weight.square().sum()
                    numel[rname][expert] += weight.numel()

    scores: dict[str, torch.Tensor] = {}
    for rname in router_names:
        missing = (numel[rname] == 0).nonzero().flatten().tolist()
        if missing:
            raise ValueError(
                f"{rname}: no weights matched expert_pattern for experts {missing}"
            )
        non_finite = (
            (~(abs_sum[rname].isfinite() & sq_sum[rname].isfinite()))
            .nonzero()
            .flatten()
            .tolist()
        )
        if non_finite:
            raise ValueError(f"{rname}: experts {non_finite} have non-finite weights")

        aimer = abs_sum[rname] / (numel[rname].double().sqrt() * sq_sum[rname].sqrt())
        # an all-zero expert has an undefined (0/0) score, which the reference
        # implementation scores as 0
        scores[rname] = -torch.where(sq_sum[rname] > 0, aimer, 0.0)

    return scores


def _is_projection_weight(name: str) -> bool:
    """Whether an expert tensor is a projection weight: a per-expert
    ``<proj>.weight`` tensor, or a stacked (3D) ``<proj>`` parameter such as
    ``mlp.experts.gate_up_proj``. Excludes e.g. ``.bias``, ``_proj_bias`` and
    ``.weight_scale``."""
    return name.endswith(".weight") or name.endswith("_proj")
