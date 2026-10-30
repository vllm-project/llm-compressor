"""
REAP (Router-weighted Expert Activation Pruning) modifier for MoE models.

See: https://arxiv.org/abs/2510.13999
"""

from functools import partial
from typing import Any

import torch
from compressed_tensors.config import MoESparsityMethod
from compressed_tensors.distributed import is_source_process
from compressed_tensors.offload import get_execution_device
from loguru import logger
from pydantic import PrivateAttr

from llmcompressor.core import Event, State
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modifiers.pruning.base import ExpertPruningModifierBase
from llmcompressor.modifiers.pruning.reap.utils import REAP_PRECISION, prune_moe_layer

__all__ = ["REAPPruningModifier"]


class REAPPruningModifier(ExpertPruningModifierBase):
    """
    Prunes experts from MoE layers using the REAP saliency metric. For each
    expert ``j`` the saliency is

        ``S_j = mean(g_j * ||f_j||_2)``

    averaged over the tokens routed to expert ``j``, where:

    - ``g_j`` is the router gate weight assigned to expert ``j`` (the coefficient
      that multiplies the expert's output when combining experts), and
    - ``f_j`` is expert ``j``'s output activation for that token, so
      ``||f_j||_2`` is its L2 norm.

    The lowest-saliency experts are removed per layer. REAP runs during the
    sequential calibration pipeline: saliency is accumulated via hooks on the MoE
    experts, the structural pruning for a layer is executed when it
    completes (``SEQUENTIAL_EPOCH_END``). The per-layer retained counts are
    recorded in the checkpoint's ``quantization_config.moe_sparsity_config``
    with ``method="prune"``.

    :param sparsity: fraction of experts to remove per layer, in ``[0, 1)``.
        When ``0.0``, no experts are pruned; saliency is still computed so
        that a ``report_path`` can be written, but the model weights and
        config are never modified.
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional path to a ``.json`` file where the per-expert
        statistics are written after calibration completes. The report is a JSON
        object mapping ``count`` and ``saliency`` to one inner list per MoE layer
        (ordered by layer). Use ``tools/plot_reap_report.py`` to generate a
        saliency visualization. Required when ``sparsity`` is ``0.0``, since the
        report is then the modifier's only output.

    Example recipe::

        REAPPruningModifier:
          sparsity: 0.25
    """

    sparsity_method = MoESparsityMethod.PRUNE

    _n_experts_to_drop_per_group: int | None = PrivateAttr(default=None)

    # per-layer REAP saliency accumulators, in float32 on each layer's
    # execution device: sum_saliency[j] accumulates g_j * ||f_j||_2 over the
    # tokens routed to expert j, counts[j] the number of routed tokens, and
    # total_counts the total number of routing decisions. They never move
    # devices (they are reduced in place over the default process group) and
    # a layer which has not seen tokens yet still contributes zeros
    _sum_saliency: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _counts: dict[str, torch.Tensor] = PrivateAttr(default_factory=dict)
    _total_counts: dict[str, float] = PrivateAttr(default_factory=dict)
    # per-batch expert output norms captured by _expert_hook, as
    # {layer_name: {expert_idx: [num_routed_tokens]}}
    _norm_buffers: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )

    # -- lifecycle ------------------------------------------------------------

    def on_initialize(self, state: State, **kwargs) -> bool:
        initialized = super().on_initialize(state, **kwargs)

        # group-limited routers require an equal number of drops per group
        if self._n_experts_to_drop > 0 and self._moe_attrs.n_group is not None:
            self._n_experts_to_drop_per_group = round(
                self._n_experts_to_drop / self._moe_attrs.n_group
            )

            if self._n_experts_to_drop_per_group == 0:
                raise ValueError(
                    f"Group-limited router detected: sparsity={self.sparsity} "
                    f"results in 0 experts to drop per group "
                    f"(out of {self._moe_attrs.group_size}). "
                    "No pruning will be performed."
                )

            logger.info(
                f"Group-limited router detected: will drop "
                f"{self._n_experts_to_drop_per_group} experts/group "
                f"(out of {self._moe_attrs.group_size} in each of "
                f"{self._moe_attrs.n_group} groups)"
            )
            if (
                self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                != self._n_experts_to_drop
            ):
                logger.warning(
                    f"REAP: group-limited routing requires an equal drop "
                    f"per group; dropping "
                    f"{self._n_experts_to_drop_per_group * self._moe_attrs.n_group} "
                    f"experts instead of the requested "
                    f"{self._n_experts_to_drop} "
                    f"(n_group={self._moe_attrs.n_group})"
                )
                self._n_experts_to_drop = (
                    self._n_experts_to_drop_per_group * self._moe_attrs.n_group
                )

        return initialized

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        for layer_name in self._moe_attrs.moe_layer_names:
            experts = getattr(
                state.model.get_submodule(layer_name), self._moe_attrs.experts_attr
            )
            self._init_layer_stats(
                layer_name,
                self._moe_attrs.num_experts,
                get_execution_device(experts),
            )
            self._norm_buffers[layer_name] = {}

        super().on_calibration_start(state, event, **kwargs)

    def on_finalize(self, state: State, **kwargs) -> bool:
        finalized = super().on_finalize(state, **kwargs)
        self._sum_saliency.clear()
        self._counts.clear()
        self._total_counts.clear()
        self._norm_buffers.clear()
        return finalized

    # -- saliency accumulation -----------------------------------------------

    def _init_layer_stats(
        self, layer_name: str, num_experts: int, device: torch.device
    ):
        self._sum_saliency[layer_name] = torch.zeros(
            num_experts, dtype=REAP_PRECISION, device=device
        )
        self._counts[layer_name] = torch.zeros(
            num_experts, dtype=REAP_PRECISION, device=device
        )
        self._total_counts[layer_name] = 0.0

    def _reduce_layer_stats(self, layer_name: str):
        """Reduce this layer's saliency accumulators to the source rank,
        leaving the total count summed across ranks on every rank."""
        sum_saliency = self._sum_saliency[layer_name]
        self._total_counts[layer_name] = self._reduce_stats(
            [sum_saliency, self._counts[layer_name]],
            self._total_counts[layer_name],
            sum_saliency.device,
        )

    def _mean_saliency(self, layer_name: str) -> torch.Tensor:
        return self._sum_saliency[layer_name] / self._counts[layer_name].clamp(min=1.0)

    @torch.no_grad()
    def _update_saliency(
        self,
        layer_name: str,
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

        sum_saliency = self._sum_saliency[layer_name]
        counts = self._counts[layer_name]
        self._total_counts[layer_name] += topk_indices.numel()

        if get_calibrate_all_experts_flag():
            stacked_norms = torch.stack(
                [expert_norms_dict[i] for i in range(self._moe_attrs.num_experts)],
                dim=1,
            )
            flat_idx = topk_indices.reshape(-1).to(torch.long)
            gathered_norms = stacked_norms.gather(1, topk_indices.to(torch.long))
            contrib = topk_weights.to(torch.float32) * gathered_norms.to(torch.float32)
            counts.index_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
            )
            sum_saliency.index_add_(0, flat_idx, contrib.reshape(-1))
        else:
            # Flatten in (slot, token) order to match torch.where order
            # in LinearExperts2D torch.where scans row-by-row (slot 0 all
            # tokens, then slot 1 all tokens, etc.)
            flat_idx = topk_indices.T.reshape(-1).to(torch.long)
            flat_weights = topk_weights.T.reshape(-1).to(torch.float32)

            # Build flat norms tensor aligned with flat_idx
            flat_norms = torch.zeros_like(flat_weights)
            for expert_idx, expert_norms in expert_norms_dict.items():
                mask = flat_idx == expert_idx

                # Assertion check: number of norms must match number of routed tokens
                assert len(expert_norms) == mask.sum().item(), (
                    f"REAP saliency accumulation: expert {expert_idx} has "
                    f"{len(expert_norms)} norms but router sent "
                    f"{mask.sum().item()} tokens to it. This indicates a bug in "
                    f"the expert hook or routing extraction logic."
                )

                flat_norms[mask] = expert_norms.to(torch.float32)

            # Vectorized computation using index_add_
            contrib = flat_weights * flat_norms
            sum_saliency.index_add_(0, flat_idx, contrib)
            counts.index_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
            )

    def _compute_retained_experts(self, layer_name: str) -> torch.Tensor:
        """Select which experts to keep, dropping the lowest-saliency ones."""
        saliency = self._mean_saliency(layer_name)

        if self._n_experts_to_drop_per_group is None:
            _, drop_indices = torch.topk(
                saliency, self._n_experts_to_drop, largest=False
            )
            drop_set = set(int(i) for i in drop_indices.tolist())
            retained = [i for i in range(saliency.shape[0]) if i not in drop_set]
        else:
            retained: list[int] = []
            for group in range(self._moe_attrs.n_group):
                lo = group * self._moe_attrs.group_size
                grp = saliency[lo : lo + self._moe_attrs.group_size]
                _, drop_local = torch.topk(
                    grp, self._n_experts_to_drop_per_group, largest=False
                )
                drop_set = {lo + int(i) for i in drop_local.tolist()}
                retained.extend(
                    i
                    for i in range(lo, lo + self._moe_attrs.group_size)
                    if i not in drop_set
                )

        return torch.tensor(retained, dtype=torch.int, device=saliency.device)

    # -- pruning --------------------------------------------------------------

    def _prune_moe_layer(self, model: torch.nn.Module, layer_name: str) -> bool:
        """Reduce this layer's saliency statistics, record them for the
        report, and prune the lowest-saliency experts from the layer."""

        if layer_name not in self._sum_saliency:
            # Layer has already been finalized
            return False

        self._reduce_layer_stats(layer_name)

        # every rank holds the total count summed across ranks after the
        # reduction, so all decide identically whether the layer collected
        # any samples; layers with no samples anywhere keep their statistics
        # in case data arrives in a later epoch
        if self._total_counts[layer_name] == 0:
            return False

        # the source rank holds the globally reduced statistics
        if is_source_process():
            self._report_values[layer_name] = {
                "count": self._counts[layer_name].tolist(),
                "saliency": self._mean_saliency(layer_name).tolist(),
            }

        if self._n_experts_to_drop > 0:
            retained = self._decide_retained_experts(
                self._sum_saliency[layer_name].device,
                partial(self._compute_retained_experts, layer_name),
            )
            self._retained_experts[layer_name] = retained
            prune_moe_layer(model, layer_name, retained, self._moe_attrs)

        # free this layer's accumulators / buffers now
        del self._sum_saliency[layer_name]
        del self._counts[layer_name]
        del self._total_counts[layer_name]
        self._norm_buffers.pop(layer_name, None)

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
        """Record expert ``f_j`` output norms for every token this batch."""
        if layer_name not in self._norm_buffers:
            # Layer has already been pruned
            return
        if isinstance(output, tuple):
            output = output[0]
        with torch.no_grad():
            norms = torch.linalg.norm(output.float(), dim=-1).reshape(-1)
        self._norm_buffers[layer_name][expert_idx] = norms

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: Any,
    ):
        """Combine router decisions with expert norms to update saliency."""
        top_k_indices, top_k_weights = args[1:3]

        if layer_name not in self._norm_buffers:
            # Layer has already been pruned
            return

        self._update_saliency(
            layer_name, top_k_indices, top_k_weights, self._norm_buffers[layer_name]
        )

        self._norm_buffers[layer_name] = {}
