"""
REAP (Router-weighted Expert Activation Pruning) modifier for MoE models.

See: https://arxiv.org/abs/2510.13999
"""

import torch
from compressed_tensors.config import MoESparsityMethod
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    wait_for_comms,
)
from loguru import logger
from pydantic import PrivateAttr
from torch import distributed as dist
from transformers.modeling_utils import local_torch_dtype

from llmcompressor.core import State
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.expert_sparsity.base import (
    ExpertPruningModifierBase,
)

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

    :param sparsity: fraction of experts to remove per layer, in ``[0, 1)``.
        When ``0.0``, no experts are pruned; saliency is still computed so
        that a ``report_path`` can be written, but the model weights and
        config are never modified.
    :param ignore: module name patterns to skip during MoE layer pruning.
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

    # subclass configuration
    sparsity_method = MoESparsityMethod.PRUNE

    # per-batch expert output norms, keyed by layer name then expert index.
    # One slot per expert (None until that expert's hook fires this batch),
    # populated by the expert hooks and discarded by the experts block hook
    # at the end of each batch
    _expert_norms: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )

    def init_moe_stats(
        self,
        moe_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        """
        Allocate the REAP accumulators for a layer.

        Creates the per-expert saliency sums and routed-token counts, plus
        the per-batch expert-norm buffers filled by :meth:`expert_hook`.

        :param moe_name: Name of the MoE layer
        :param num_experts: Number of experts in the layer
        :param hidden_dim: Unused by REAP
        :param device: The layer's execution device
        """
        with local_torch_dtype(self.precision), device:
            self._moe_stats[moe_name] = {
                "sum_saliency": torch.zeros(num_experts),
                "count": torch.zeros(num_experts, dtype=torch.int),
            }
            self._expert_norms[moe_name] = {}

    @torch.no_grad()
    def expert_hook(
        self,
        moe_name: str,
        expert_idx: int,
        module: ExpertMLP,
        args: tuple,
        output: torch.Tensor,
    ):
        """
        Record the per-token L2 norm of the expert's output.

        Upcasts to the modifier's precision and stores into the layer's
        per-batch norm buffer, for :meth:`experts_block_hook` to consume at
        the end of the batch.

        :param moe_name: Name of the MoE layer the expert belongs to
        :param expert_idx: Index of the expert within its layer
        :param module: The expert module the hook is attached to
        :param args: The expert's forward arguments
        :param output: The expert's output for the current batch
        """
        if moe_name not in self._moe_stats:
            logger.warning(
                f"Attempted to calibrate layer {moe_name} which was never "
                "initialized or has already been finalized"
            )
            return

        norm = torch.linalg.norm(output.to(self.precision), dim=-1)
        self._expert_norms[moe_name][expert_idx] = norm

    @torch.no_grad()
    def experts_block_hook(
        self,
        moe_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: torch.Tensor,
    ):
        """
        Add each routed token's contribution to the layer's statistics.

        Accumulates ``topk_weight * ||f_j||`` into the per-expert saliency
        sums and bumps the per-expert routed-token counts. Two token layouts
        are handled: under the calibrate-all-experts flag every expert is
        evaluated for every token, so norms are gathered by top-k index;
        otherwise indices are flattened in (slot, token) order to match the
        ``torch.where`` scatter order of ``LinearExperts2D``. Resets the
        per-batch norm buffers afterwards.

        :param moe_name: Name of the MoE layer the block belongs to
        :param module: The experts block the hook is attached to
        :param args: The block's forward arguments, where ``args[1]`` and
            ``args[2]`` are the router's top-k indices and weights
        :param output: The block's output for the current batch
        """
        if moe_name not in self._moe_stats:
            logger.warning(
                f"Attempted to calibrate layer {moe_name} which was never "
                "initialized or has already been finalized"
            )
            return

        topk_indices: torch.Tensor = args[1].to(torch.long)
        topk_weights: torch.Tensor = args[2].to(self.precision)

        sum_saliency = self._moe_stats[moe_name]["sum_saliency"]
        counts = self._moe_stats[moe_name]["count"]

        with local_torch_dtype(self.precision):
            if get_calibrate_all_experts_flag():
                stacked_norms = torch.stack(self._expert_norms[moe_name], dim=1)
                flat_idx = topk_indices.reshape(-1)
                gathered_norms = stacked_norms.gather(1, topk_indices)
                contrib = topk_weights * gathered_norms
                counts.index_add_(
                    0, flat_idx, torch.ones_like(flat_idx, dtype=counts.dtype)
                )
                sum_saliency.index_add_(0, flat_idx, contrib.reshape(-1))
            else:
                # Flatten in (slot, token) order to match torch.where order
                # in LinearExperts2D torch.where scans row-by-row (slot 0 all
                # tokens, then slot 1 all tokens, etc.)
                flat_idx = topk_indices.T.reshape(-1)
                flat_weights = topk_weights.T.reshape(-1)

                # Build flat norms tensor aligned with flat_idx
                flat_norms = torch.zeros_like(flat_weights)
                for expert_idx, expert_norms in self._expert_norms[moe_name].items():
                    mask = flat_idx == expert_idx
                    flat_norms[mask] = expert_norms

                # Vectorized computation using index_add_
                contrib = flat_weights * flat_norms
                sum_saliency.index_add_(0, flat_idx, contrib)
                counts.index_add_(
                    0, flat_idx, torch.ones_like(flat_idx, dtype=counts.dtype)
                )

        # reset expert norms for next batch
        num_experts = self._moe_config.num_experts
        self._expert_norms[moe_name] = [None for _ in range(num_experts)]

    def reduce_moe_stats(self, moe_name: str):
        """
        Reduce a layer's statistics across ranks, in place.

        Reduces the saliency sums to the source rank, which alone computes
        the pruning decision and report from them, and all-reduces the
        per-expert counts so every rank agrees on whether the layer saw
        calibration data.

        :param moe_name: Name of the MoE layer whose stats to reduce
        """
        if not is_distributed():
            return

        stats = self._moe_stats[moe_name]
        pending = [
            dist.reduce(
                stats["sum_saliency"],
                dst=get_source_rank(),
                op=dist.ReduceOp.SUM,
                async_op=True,
            ),
            dist.all_reduce(stats["count"], op=dist.ReduceOp.SUM, async_op=True),
        ]
        wait_for_comms(pending)

    def saliency(self, moe_name: str) -> torch.Tensor:
        """
        Return the mean saliency ``S_j`` per expert.

        The router-weighted mean of the expert's output norms over the
        tokens routed to it.

        :param moe_name: Name of the MoE layer
        :return: Per-expert saliency, indexed by expert
        """
        stats = self._moe_stats[moe_name]
        return stats["sum_saliency"] / stats["count"].clamp(min=1.0)

    def on_finalize(self, state: State, **kwargs) -> bool:
        ret = super().on_finalize(state, **kwargs)
        self._expert_norms.clear()
        return ret
