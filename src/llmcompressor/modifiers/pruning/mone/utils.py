"""
Utilities for MoNe (Mixture of Novices): saliency tracking, novice-output
tracking and expert collapse.

MoNe shares REAP's *collapse* structure but uses its own saliency metric. The
lowest-saliency experts are *collapsed* into constant "novice" vectors (each
expert's mean output, an unbiased estimator of its contribution). Retained
experts are left-shifted while the router keeps all ``R + N`` rows, so novices
continue to receive their original router weights at inference time.

Expert saliency is the MoNe "fusion" metric

    ``S_j = fluctuation_j * routing_score_j``

where ``fluctuation_j = || std_tokens(f_j) ||_2`` is the L2 norm (over hidden
dims) of the per-dimension standard deviation of expert ``j``'s output ``f_j``,
and ``routing_score_j = E_tokens[ g_j * 1(j in top-k) ]`` is the
frequency-weighted mean router gate weight (averaged over *all* tokens, zero when
``j`` is not selected). Experts whose output varies little (low fluctuation) are
well approximated by their constant mean, so they are collapsed first.
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
    _prune_router,
)

__all__ = [
    "MoNeSaliencyTracker",
    "collapse_moe_layer",
    "build_moe_num_experts",
]


class MoNeSaliencyTracker:
    """
    Accumulates the MoNe "fusion" saliency metric and the per-expert mean
    output used as the novice vector.

    Two groups of statistics are accumulated per expert, over the tokens routed
    to it:

    - Output statistics (``sum_output``, ``sum_sq_output``, ``output_count``):
      the mean output is the novice vector, and the per-dimension variance gives
      the ``fluctuation`` term of the saliency metric.
    - Routing statistics (``sum_gate``, ``token_count``): the frequency-weighted
      mean gate weight gives the ``routing_score`` term.

    The accumulators are zero-initialized on ``device`` at construction and
    never move devices: statistics are accumulated and reduced in float32
    (gpu-backed process groups do not support float64) where the calibration
    data flows, and a layer which has not seen tokens yet contributes its
    zeros to every reduction.

    Saliency is ``fluctuation_j * routing_score_j``; experts with the lowest
    saliency are collapsed into novices.
    """

    def __init__(self, num_experts: int, hidden_dim: int, device: torch.device):
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.device = device
        # output statistics: mean -> novice vector, variance -> fluctuation
        self.sum_output = torch.zeros(
            num_experts, hidden_dim, dtype=torch.float32, device=device
        )
        self.sum_sq_output = torch.zeros_like(self.sum_output)
        self.output_count = torch.zeros(num_experts, dtype=torch.float32, device=device)
        # routing statistics: frequency-weighted mean gate weight
        self.sum_gate = torch.zeros(num_experts, dtype=torch.float32, device=device)
        self.token_count: int = 0

    @torch.no_grad()
    def update_outputs(
        self,
        topk_indices: torch.Tensor,
        expert_outputs_dict: dict[int, torch.Tensor],
    ):
        """
        Accumulate the sum and sum-of-squares of each expert's output over the
        tokens routed to it (for the novice mean and the fluctuation term).

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids, used to
            mask routed tokens when ``calibrate_all_experts`` is enabled
        :param expert_outputs_dict: dict mapping expert_idx to that expert's output
            activations. In sparse mode these are already the routed-token outputs
            ``[num_routed, hidden_dim]``; in calibrate-all mode they are the
            dense outputs ``[num_tokens, hidden_dim]`` and are masked here.
        """
        if not expert_outputs_dict:
            return

        calibrate_all = get_calibrate_all_experts_flag()
        for expert_idx, output in expert_outputs_dict.items():
            if calibrate_all:
                mask = (topk_indices == expert_idx).any(dim=-1)
                output = output[mask]
            if output.shape[0] == 0:
                continue
            output = output.to(torch.float32)
            self.sum_output[expert_idx] += output.sum(dim=0)
            self.sum_sq_output[expert_idx] += output.pow(2).sum(dim=0)
            self.output_count[expert_idx] += output.shape[0]

    @torch.no_grad()
    def update_routing(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """
        Accumulate the frequency-weighted router gate weight per expert.

        ``sum_gate[j]`` collects ``g_j`` summed over the tokens for which ``j`` is
        in the top-k, and ``token_count`` counts all tokens, so that
        ``routing_score_j = sum_gate[j] / token_count`` equals
        ``E_tokens[ g_j * 1(j in top-k) ]``.

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids
        :param topk_weights: ``[num_tokens, top_k]`` gate weight per selection
        """
        flat_idx = topk_indices.reshape(-1).to(torch.long)
        flat_weights = topk_weights.reshape(-1).to(torch.float32)
        self.sum_gate.index_add_(0, flat_idx, flat_weights)
        self.token_count += topk_indices.shape[0]

    def reduce_stats(self):
        """Reduce all accumulators to the source rank (rank 0), in float32 on
        the device they were accumulated on, over the default process group.
        The token count is all-reduced so that every rank holds the count
        summed across ranks.

        The accumulators are reduced in place: ``dist.reduce`` only writes
        its result at the destination, so non-source ranks keep their local
        statistics, while the source rank's accumulators become the global
        sums, which their consumers (report, retained-expert selection,
        novices) read immediately after. A rank whose layer has not seen
        tokens contributes its zero-initialized accumulators, keeping the
        collectives aligned across ranks.
        """
        if not is_distributed():
            return

        token_count = torch.tensor(
            float(self.token_count), dtype=torch.float32, device=self.device
        )
        pending = [
            dist.reduce(
                tensor,
                dst=get_source_rank(),
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            for tensor in (
                self.sum_output,
                self.sum_sq_output,
                self.output_count,
                self.sum_gate,
            )
        ]
        pending.append(
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM, async_op=True)
        )
        wait_for_comms(pending)
        self.token_count = int(token_count.item())

    @property
    def total_count(self) -> int:
        """Total number of tokens seen by the routing hook; after
        :meth:`reduce_stats` this is the count summed across ranks, used to
        detect which trackers actually collected calibration samples."""
        return self.token_count

    @property
    def output_fluctuation(self) -> torch.Tensor:
        """``[num_experts]`` L2 norm (over hidden dims) of the per-dimension
        standard deviation of each expert's output."""
        count = self.output_count.clamp(min=1.0).unsqueeze(-1)
        mean = self.sum_output / count
        variance = (self.sum_sq_output / count - mean.pow(2)).clamp_min(0.0)
        std = variance.sqrt()
        return torch.linalg.norm(std, dim=1)

    @property
    def routing_score(self) -> torch.Tensor:
        """``[num_experts]`` frequency-weighted mean gate weight."""
        if self.token_count <= 0:
            return torch.zeros(
                self.num_experts, dtype=torch.float32, device=self.device
            )
        return self.sum_gate / float(self.token_count)

    @property
    def saliency(self) -> torch.Tensor:
        """The MoNe fusion saliency ``fluctuation * routing_score`` per expert."""
        return self.output_fluctuation * self.routing_score

    def compute_retained_experts(self, n_experts_to_drop: int) -> torch.Tensor:
        """Select which experts to keep, dropping the lowest-saliency ones."""
        saliency = torch.nan_to_num(self.saliency, nan=-float("inf"))
        _, drop_indices = torch.topk(saliency, n_experts_to_drop, largest=False)
        drop_set = {int(i) for i in drop_indices.tolist()}
        retained = [i for i in range(self.num_experts) if i not in drop_set]
        return torch.tensor(retained, dtype=torch.int, device=saliency.device)

    def compute_novices(self, dropped: list[int]) -> torch.Tensor:
        """Return the ``(len(dropped), hidden_dim)`` matrix of novice vectors --
        the mean output of each collapsed expert, ordered to match ``dropped``.
        """
        count = self.output_count.clamp(min=1.0).unsqueeze(-1)
        mean_output = self.sum_output / count
        drop_idx = torch.tensor(dropped, dtype=torch.long, device=mean_output.device)
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
    new_modules = OrderedDict((str(i), experts[pos]) for i, pos in enumerate(retained))
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
    experts.register_parameter("novices", nn.Parameter(novices, requires_grad=False))


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
