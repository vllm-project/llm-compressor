"""
Utilities for MoNe (Mixture of Novices): saliency + novice-output tracking and
expert collapse.

MoNe shares REAP's structure. It selects low-saliency experts using the same
``S_j = mean(g_j * ||f_j||_2)`` metric, but instead of pruning those experts it
*collapses* them into constant "novice" vectors (each expert's mean output, an
unbiased estimator of its contribution). Retained experts are left-shifted while
the router keeps all ``R + N`` rows, so novices continue to receive their
original router weights at inference time.
"""

import copy
import re
from collections import OrderedDict

import torch
import torch.nn as nn
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    wait_for_comms,
)
from compressed_tensors.offload import get_execution_device
from torch import distributed as dist

from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.reap.utils import (
    NUM_EXPERTS_CONFIG_KEYS,
    MoeModelAttrs,
    REAPSaliencyTracker,
    _prune_router,
)

__all__ = [
    "MoNeSaliencyTracker",
    "collapse_moe_layer",
    "build_moe_num_experts",
]


class MoNeSaliencyTracker(REAPSaliencyTracker):
    """
    Extends :class:`REAPSaliencyTracker` with per-expert mean-output accumulation.

    The saliency statistics are identical to REAP and decide which experts to
    collapse. Additionally, the running sum of each expert's output activations
    (over the tokens routed to it) is accumulated so that the mean output -- the
    constant "novice" vector for a collapsed expert -- can be recovered.
    """

    def __init__(self, num_experts: int):
        super().__init__(num_experts)
        self.sum_output: torch.Tensor | None = None
        self.output_count: torch.Tensor | None = None

    def _ensure_outputs(self, device: torch.device, hidden_dim: int):
        if self.sum_output is None:
            self.sum_output = torch.zeros(
                self.num_experts, hidden_dim, dtype=torch.float64, device=device
            )
            self.output_count = torch.zeros(
                self.num_experts, dtype=torch.float64, device=device
            )

    @torch.no_grad()
    def update_outputs(
        self,
        topk_indices: torch.Tensor,
        expert_outputs_dict: dict[int, torch.Tensor],
    ):
        """
        Accumulate the sum of each expert's output over the tokens routed to it.

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids, used to
            mask routed tokens when ``calibrate_all_experts`` is enabled
        :param expert_outputs_dict: dict mapping expert_idx to that expert's output
            activations. In sparse mode these are already the routed-token outputs
            ``[num_routed, hidden_dim]``; in calibrate-all mode they are the
            dense outputs ``[num_tokens, hidden_dim]`` and are masked here.
        """
        if not expert_outputs_dict:
            return

        sample = next(iter(expert_outputs_dict.values()))
        self._ensure_outputs(sample.device, sample.shape[-1])

        calibrate_all = get_calibrate_all_experts_flag()
        for expert_idx, output in expert_outputs_dict.items():
            if calibrate_all:
                mask = (topk_indices == expert_idx).any(dim=-1)
                output = output[mask]
            if output.shape[0] == 0:
                continue
            self.sum_output[expert_idx] += output.to(torch.float64).sum(dim=0)
            self.output_count[expert_idx] += output.shape[0]

    def reduce_output_stats(
        self,
        device: torch.device,
        hidden_dim: int,
        group: "dist.ProcessGroup | None" = None,
    ):
        """Reduce novice-output accumulators to the source rank (rank 0)."""
        self._ensure_outputs(device, hidden_dim)
        if is_distributed():
            self.sum_output = self.sum_output.cpu()
            self.output_count = self.output_count.cpu()
            pending = [
                dist.reduce(
                    self.sum_output,
                    dst=get_source_rank(),
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                    group=group,
                ),
                dist.reduce(
                    self.output_count,
                    dst=get_source_rank(),
                    op=dist.ReduceOp.SUM,
                    async_op=True,
                    group=group,
                ),
            ]
            wait_for_comms(pending)

    def compute_novices(self, dropped: list[int]) -> torch.Tensor:
        """Return the ``(len(dropped), hidden_dim)`` matrix of novice vectors --
        the mean output of each collapsed expert, ordered to match ``dropped``.
        """
        if self.sum_output is None:
            raise RuntimeError(
                "MoNeSaliencyTracker.compute_novices called before any outputs "
                "were accumulated."
            )
        sum_output = self.sum_output.to("cpu", torch.float64)
        count = self.output_count.to("cpu", torch.float64).clamp(min=1.0)
        mean_output = sum_output / count.unsqueeze(-1)
        drop_idx = torch.tensor(dropped, dtype=torch.long)
        return mean_output[drop_idx]


def collapse_moe_layer(
    model: nn.Module,
    layer_name: str,
    retained: list[int],
    novices: torch.Tensor,
    moe_attrs: MoeModelAttrs,
) -> list[int]:
    """
    Collapse a MoE block into retained experts + novices (MoNe).

    The retained experts are left-shifted into a contiguous ``ModuleList`` exactly
    as in REAP pruning. Unlike REAP, the router is *not* shrunk: its rows are
    reordered so the retained experts occupy rows ``0..R`` and the collapsed
    experts occupy rows ``R..R+N`` (matching the order of ``novices``). The
    collapsed experts' mean outputs are installed as ``experts.novices``.

    :param novices: ``(N, hidden_dim)`` tensor of novice vectors, ordered to match
        the dropped experts (i.e. the router rows that follow the retained block)
    """
    moe_block = model.get_submodule(layer_name)
    router = getattr(moe_block, moe_attrs.router_attr)
    experts = getattr(moe_block, moe_attrs.experts_attr)

    num_experts = moe_attrs.num_experts
    retained_set = set(retained)
    dropped = [i for i in range(num_experts) if i not in retained_set]
    assert novices.shape[0] == len(dropped), (
        f"collapse_moe_layer: got {novices.shape[0]} novices but "
        f"{len(dropped)} experts were collapsed"
    )

    # Preserve non-expert modules (e.g. act_fn in LinearExperts2D)
    non_expert_modules = {
        key: module
        for key, module in experts._modules.items()
        if not isinstance(module, ExpertMLP)
    }

    # Rebuild with retained experts (left-shifted), re-adding non-expert modules
    new_modules = OrderedDict(
        (str(i), experts[pos]) for i, pos in enumerate(retained)
    )
    new_modules.update(non_expert_modules)
    experts._modules = new_modules
    experts.num_experts = len(retained)

    # Narrow the repack source config so to_experts_module() rebuilds only the
    # retained experts; the router and novices keep the full R + N width.
    _narrow_source_config(experts, len(retained))

    # Install novice vectors on the experts module.
    _set_novices(experts, novices)

    # Reorder router rows to [retained..., dropped...] while keeping all R + N rows.
    _prune_router(router, retained + dropped)

    # Reorder a MoE-block-level score-correction bias if present (e.g. Hy3).
    e_bias = getattr(moe_block, "e_score_correction_bias", None)
    if e_bias is not None:
        from compressed_tensors.offload.utils import to_tensor

        order = torch.tensor(retained + dropped, dtype=torch.long, device=e_bias.device)
        moe_block.e_score_correction_bias = to_tensor(
            e_bias[order].contiguous(), e_bias
        )

    return retained


def _set_novices(experts: nn.Module, novices: torch.Tensor) -> None:
    exec_device = get_execution_device(experts)
    target_dtype = novices.dtype
    for param in experts.parameters():
        if param.is_floating_point():
            target_dtype = param.dtype
            break
    novices = novices.to(device=exec_device, dtype=target_dtype).contiguous()
    experts.register_parameter(
        "novices", nn.Parameter(novices, requires_grad=False)
    )


def _narrow_source_config(experts: nn.Module, new_num_experts: int) -> None:
    config = getattr(experts, "_source_config", None)
    if config is None:
        return
    new_config = copy.copy(config)
    for key in NUM_EXPERTS_CONFIG_KEYS:
        if hasattr(new_config, key):
            setattr(new_config, key, new_num_experts)
    experts._source_config = new_config


def build_moe_num_experts(
    model: nn.Module,
    moe_attrs: MoeModelAttrs,
    retained_experts: dict[str, list[int]],
) -> list[int | None]:
    """
    Build the per-decoder-layer retained-expert-count list for
    :class:`~compressed_tensors.config.ExpertSparsityConfig`.

    The list length equals the number of decoder layers. Dense layers (and MoE
    layers that were not collapsed) are ``None``; collapsed MoE layers hold their
    retained expert count.
    """
    config = model.config.text_config if moe_attrs.has_text_config else model.config
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(
            "Could not determine num_hidden_layers to build moe_sparsity_config"
        )

    per_layer: list[int | None] = [None] * num_layers
    for layer_name, retained in retained_experts.items():
        match = re.search(r"layers\.(\d+)", layer_name)
        if match is None:
            raise ValueError(
                f"Could not extract a decoder layer index from '{layer_name}'"
            )
        per_layer[int(match.group(1))] = len(retained)

    return per_layer
