"""
MoNe (Mixture of Novices) pruning modifier for MoE models.

MoNe mirrors REAP's *collapse* structure (see
:class:`~llmcompressor.modifiers.pruning.reap.REAPPruningModifier`) but ranks
experts with its own "fusion" saliency metric
``S_j = ||std_tokens(f_j)||_2 * E_tokens[g_j * 1(j in top-k)]`` accumulated via
calibration hooks. It selects the lowest-saliency experts per layer and,
instead of pruning them, *collapses* them into constant "novice" vectors -- each
expert's mean output, an unbiased estimator of its contribution -- leaving the
router untouched so the novices keep receiving their original routing weights.

See: https://arxiv.org/abs/2507.00390
"""

from functools import partial
from typing import Any

import torch
from compressed_tensors import align_module_device
from compressed_tensors.config import MoESparsityMethod
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
)
from compressed_tensors.offload import get_execution_device
from loguru import logger
from pydantic import PrivateAttr
from torch import distributed as dist

from llmcompressor.core import Event, State
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modifiers.pruning.base import ExpertPruningModifierBase
from llmcompressor.modifiers.pruning.mone.utils import collapse_moe_layer

__all__ = ["MoNePruningModifier"]


class MoNePruningModifier(ExpertPruningModifierBase):
    """
    Collapses low-saliency experts from MoE layers into constant "novice" vectors
    using the MoNe saliency metric. For each expert ``j`` the saliency is

        ``S_j = ||std_tokens(f_j)||_2 * E_tokens[g_j * 1(j in top-k)]``

    the product of the expert's output fluctuation (the L2 norm of the
    per-dimension standard deviation of expert ``j``'s output ``f_j``) and its
    routing score (the frequency-weighted mean router gate weight ``g_j``,
    averaged over all tokens and zero when ``j`` is not in the top-k). The
    lowest-saliency experts per layer are collapsed: their mean
    output over calibration tokens becomes a constant novice vector, retained
    experts are left-shifted, and the router is kept at its original width so
    novices continue to receive their routing weights. The per-layer retained
    counts are recorded in the checkpoint's
    ``quantization_config.moe_sparsity_config`` with ``method="novice"``.

    MoNe runs during the sequential calibration pipeline: routing and expert
    output statistics are accumulated via hooks on the MoE experts, and the
    structural collapse for a layer is executed when it completes
    (``SEQUENTIAL_EPOCH_END``).

    :param sparsity: fraction of experts to collapse per layer, in ``[0, 1)``.
        When ``0.0``, no experts are collapsed; saliency is still computed so
        that a ``report_path`` can be written, but the model weights and
        config are never modified.
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional path to a ``.json`` file where the per-expert
        statistics are written after calibration completes. The report is a JSON
        object mapping ``count`` and ``saliency`` to one inner list per MoE layer
        (ordered by layer). Required when ``sparsity`` is ``0.0``, since the
        report is then the modifier's only output.

    Example recipe::

        MoNePruningModifier:
          sparsity: 0.25
    """

    sparsity_method = MoESparsityMethod.NOVICE

    # per-layer MoNe accumulators, in float32 on each layer's execution
    # device. Output statistics (sum_output, sum_sq_output, output_counts):
    # the mean output is the novice vector, and the per-dimension variance
    # gives the fluctuation term of the saliency metric. Routing statistics
    # (sum_gates, token_counts): the frequency-weighted mean gate weight
    # gives the routing score term. They never move devices (they are
    # reduced in place over the default process group) and a layer which has
    # not seen tokens yet still contributes zeros
    _sum_output: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _sum_sq_output: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _output_counts: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _sum_gates: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _token_counts: dict[str, int] = PrivateAttr(default_factory=dict)
    # per-batch expert outputs captured by _expert_hook, as
    # {layer_name: {expert_idx: [num_routed_tokens, hidden_dim]}}
    _output_buffers: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )

    # -- lifecycle ------------------------------------------------------------

    def on_initialize(self, state: State, **kwargs) -> bool:
        initialized = super().on_initialize(state, **kwargs)

        # n_group == 1 is a degenerate "group-limited" router: a single group
        # holds every expert, so routing is equivalent to ungrouped top-k and
        # MoNe's uniform collapse is correct. Only genuine group-limited
        # routers (n_group > 1), which would need group-balanced collapse,
        # are unsupported.
        if self._moe_attrs.n_group is not None and self._moe_attrs.n_group > 1:
            if self._n_experts_to_drop > 0:
                raise NotImplementedError(
                    "MoNe does not support group-limited routers "
                    f"(n_group={self._moe_attrs.n_group}). Group-balanced expert "
                    "collapse is not implemented."
                )
            # report-only mode: no collapsing happens, but the recorded
            # saliency could not be used to collapse this model's experts
            logger.warning(
                "MoNe does not support group-limited routers "
                f"(n_group={self._moe_attrs.n_group}); saliency will be "
                "recorded for the report, but this model's experts cannot "
                "be collapsed"
            )

        return initialized

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        for layer_name in self._moe_attrs.moe_layer_names:
            module = state.model.get_submodule(layer_name)
            experts = getattr(module, self._moe_attrs.experts_attr)
            router = getattr(module, self._moe_attrs.router_attr)

            with align_module_device(router):
                hidden_dim = router.weight.shape[1]

            self._init_layer_stats(
                layer_name,
                self._moe_attrs.num_experts,
                hidden_dim,
                get_execution_device(experts),
            )
            self._output_buffers[layer_name] = {}

        super().on_calibration_start(state, event, **kwargs)

    def on_finalize(self, state: State, **kwargs) -> bool:
        finalized = super().on_finalize(state, **kwargs)
        self._sum_output.clear()
        self._sum_sq_output.clear()
        self._output_counts.clear()
        self._sum_gates.clear()
        self._token_counts.clear()
        self._output_buffers.clear()
        return finalized

    # -- saliency accumulation -----------------------------------------------

    def _init_layer_stats(
        self, layer_name: str, num_experts: int, hidden_dim: int, device: torch.device
    ):
        self._sum_output[layer_name] = torch.zeros(
            num_experts, hidden_dim, dtype=torch.float32, device=device
        )
        self._sum_sq_output[layer_name] = torch.zeros_like(self._sum_output[layer_name])
        self._output_counts[layer_name] = torch.zeros(
            num_experts, dtype=torch.float32, device=device
        )
        self._sum_gates[layer_name] = torch.zeros(
            num_experts, dtype=torch.float32, device=device
        )
        self._token_counts[layer_name] = 0

    def _reduce_layer_stats(self, layer_name: str):
        """Reduce this layer's accumulators to the source rank, leaving the
        token count summed across ranks on every rank."""
        sum_output = self._sum_output[layer_name]
        self._token_counts[layer_name] = int(
            self._reduce_stats(
                [
                    sum_output,
                    self._sum_sq_output[layer_name],
                    self._output_counts[layer_name],
                    self._sum_gates[layer_name],
                ],
                self._token_counts[layer_name],
                sum_output.device,
            )
        )

    @torch.no_grad()
    def _update_routing(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """
        Accumulate the frequency-weighted router gate weight per expert.

        ``sum_gates[j]`` collects ``g_j`` summed over the tokens for which ``j`` is
        in the top-k, and ``token_counts`` counts all tokens, so that
        ``routing_score_j = sum_gates[j] / token_counts`` equals
        ``E_tokens[ g_j * 1(j in top-k) ]``.

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids
        :param topk_weights: ``[num_tokens, top_k]`` gate weight per selection
        """
        flat_idx = topk_indices.reshape(-1).to(torch.long)
        flat_weights = topk_weights.reshape(-1).to(torch.float32)
        self._sum_gates[layer_name].index_add_(0, flat_idx, flat_weights)
        self._token_counts[layer_name] += topk_indices.shape[0]

    @torch.no_grad()
    def _update_outputs(
        self,
        layer_name: str,
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
            self._sum_output[layer_name][expert_idx] += output.sum(dim=0)
            self._sum_sq_output[layer_name][expert_idx] += output.pow(2).sum(dim=0)
            self._output_counts[layer_name][expert_idx] += output.shape[0]

    def _routing_score(self, layer_name: str) -> torch.Tensor:
        """``[num_experts]`` frequency-weighted mean gate weight."""
        token_count = self._token_counts[layer_name]
        if token_count <= 0:
            return torch.zeros_like(self._sum_gates[layer_name])
        return self._sum_gates[layer_name] / float(token_count)

    def _output_fluctuation(self, layer_name: str) -> torch.Tensor:
        """``[num_experts]`` L2 norm (over hidden dims) of the per-dimension
        standard deviation of each expert's output."""
        count = self._output_counts[layer_name].clamp(min=1.0).unsqueeze(-1)
        mean = self._sum_output[layer_name] / count
        variance = (self._sum_sq_output[layer_name] / count - mean.pow(2)).clamp_min(
            0.0
        )
        std = variance.sqrt()
        return torch.linalg.norm(std, dim=1)

    def _saliency(self, layer_name: str) -> torch.Tensor:
        """The MoNe fusion saliency ``fluctuation * routing_score`` per expert."""
        return self._output_fluctuation(layer_name) * self._routing_score(layer_name)

    def _compute_retained_experts(self, layer_name: str) -> torch.Tensor:
        """Select which experts to keep, dropping the lowest-saliency ones."""
        saliency = torch.nan_to_num(self._saliency(layer_name), nan=-float("inf"))
        _, drop_indices = torch.topk(saliency, self._n_experts_to_drop, largest=False)
        drop_set = {int(i) for i in drop_indices.tolist()}
        retained = [i for i in range(saliency.shape[0]) if i not in drop_set]
        return torch.tensor(retained, dtype=torch.int, device=saliency.device)

    def _compute_novices(self, layer_name: str, dropped: list[int]) -> torch.Tensor:
        """Return the ``(len(dropped), hidden_dim)`` matrix of novice vectors --
        the mean output of each collapsed expert, ordered to match ``dropped``.
        """
        count = self._output_counts[layer_name].clamp(min=1.0).unsqueeze(-1)
        mean_output = self._sum_output[layer_name] / count
        drop_idx = torch.tensor(dropped, dtype=torch.long, device=mean_output.device)
        return mean_output[drop_idx]

    # -- pruning --------------------------------------------------------------

    def _prune_moe_layer(self, model: torch.nn.Module, layer_name: str) -> bool:
        """Reduce this layer's statistics, record them for the report, and
        collapse the lowest-saliency experts of the layer into novices."""

        if layer_name not in self._sum_output:
            # Layer has already been finalized
            return False

        self._reduce_layer_stats(layer_name)

        # every rank holds the token count summed across ranks after the
        # reduction, so all decide identically whether the layer collected
        # any samples; layers with no samples anywhere keep their statistics
        # in case data arrives in a later epoch
        if self._token_counts[layer_name] == 0:
            return False

        # the source rank holds the globally reduced statistics
        if is_source_process():
            self._report_values[layer_name] = {
                "count": self._output_counts[layer_name].tolist(),
                "saliency": self._saliency(layer_name).tolist(),
            }

        if self._n_experts_to_drop > 0:
            n_dropped = self._n_experts_to_drop
            sum_output = self._sum_output[layer_name]
            retained = self._decide_retained_experts(
                sum_output.device,
                partial(self._compute_retained_experts, layer_name),
            )
            retained_set = set(retained)
            num_experts = sum_output.shape[0]
            dropped = [i for i in range(num_experts) if i not in retained_set]

            if is_source_process():
                novices = self._compute_novices(layer_name, dropped)
            else:
                novices = torch.empty(
                    n_dropped,
                    sum_output.shape[1],
                    dtype=torch.float32,
                    device=sum_output.device,
                )

            if is_distributed():
                dist.broadcast(novices, src=get_source_rank())

            self._retained_experts[layer_name] = retained
            collapse_moe_layer(model, layer_name, retained, novices, self._moe_attrs)

        # free this layer's accumulators / buffers now
        del self._sum_output[layer_name]
        del self._sum_sq_output[layer_name]
        del self._output_counts[layer_name]
        del self._sum_gates[layer_name]
        del self._token_counts[layer_name]
        self._output_buffers.pop(layer_name, None)

        return True

    # -- calibration hooks ---------------------------------------------------

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Record expert ``f_j`` full outputs (for the novice mean and the
        output-fluctuation term) for every token this batch."""
        if layer_name not in self._output_buffers:
            # Layer has already been collapsed
            return
        if isinstance(output, tuple):
            output = output[0]
        self._output_buffers[layer_name][expert_idx] = output.detach()

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Combine router decisions with expert outputs to update the routing and
        novice accumulators."""
        top_k_indices = args[1]
        top_k_weights = args[2]

        if layer_name not in self._output_buffers:
            # Layer has already been collapsed
            return

        self._update_routing(layer_name, top_k_indices, top_k_weights)
        self._update_outputs(
            layer_name, top_k_indices, self._output_buffers[layer_name]
        )

        self._output_buffers[layer_name] = {}
