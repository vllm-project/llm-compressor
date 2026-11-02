"""
REAP (Router-weighted Expert Activation Pruning) modifier for MoE models.

See: https://arxiv.org/abs/2510.13999
"""

from collections import defaultdict

import torch
from compressed_tensors.config import MoESparsityMethod
from pydantic import PrivateAttr
from transformers.modeling_utils import local_torch_dtype

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
    stats_to_report = ("saliency", "counts")

    _expert_norms: dict[list[torch.Tensor]] = PrivateAttr(
        default_factory=defaultdict(list)
    )

    def init_moe_stats(
        self,
        moe_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        return {
            "sum_saliency": torch.zeros(
                num_experts, dtype=self.precision, device=device
            ),
            "counts": torch.zeros(num_experts, dtype=torch.int, device=device),
        }

    def expert_hook(
        self,
        moe_name: str,
        expert_idx: int,
        module: ExpertMLP,
        args: tuple,
        output: torch.Tensor,
    ):
        norm = torch.linalg.norm(output.float(), dim=-1).reshape(-1)
        self._expert_norms[moe_name][expert_idx] = norm

    def experts_block_hook(
        self,
        moe_name: str,
        module: torch.nn.Module,
        args: tuple,
        output: torch.Tensor,
    ):
        topk_indices: torch.Tensor = args[1].to(self.precision)
        topk_weights: torch.Tensor = args[2].to(self.precision)

        stats = self._moe_stats[moe_name]
        sum_saliency = stats["sum_saliency"]
        counts = stats["counts"]
        self._moe_stats[moe_name]["counts"] += topk_indices.numel()

        # TODO: I think the "counts" calculations for this function are incorrect
        # TODO: at some point, `_expert_norms` needs to be discarded

        with local_torch_dtype(self.precision):
            if get_calibrate_all_experts_flag():
                stacked_norms = torch.stack(self._expert_norms[moe_name], dim=1)
                flat_idx = topk_indices.reshape(-1)
                gathered_norms = stacked_norms.gather(1, topk_indices)
                contrib = topk_weights * gathered_norms
                counts.index_add_(0, flat_idx, torch.ones_like(flat_idx))
                sum_saliency.index_add_(0, flat_idx, contrib.reshape(-1))
            else:
                # Flatten in (slot, token) order to match torch.where order
                # in LinearExperts2D torch.where scans row-by-row (slot 0 all
                # tokens, then slot 1 all tokens, etc.)
                flat_idx = topk_indices.T.reshape(-1)
                flat_weights = topk_weights.T.reshape(-1)

                # Build flat norms tensor aligned with flat_idx
                flat_norms = torch.zeros_like(flat_weights)
                for expert_idx, expert_norms in enumerate(self._expert_norms[moe_name]):
                    mask = flat_idx == expert_idx

                    assert len(expert_norms) == mask.sum().item(), (
                        f"REAP saliency accumulation: expert {expert_idx} has "
                        f"{len(expert_norms)} norms but router sent "
                        f"{mask.sum().item()} tokens to it. This indicates a bug in "
                        f"the expert hook or routing extraction logic."
                    )

                    flat_norms[mask] = expert_norms

                # Vectorized computation using index_add_
                contrib = flat_weights * flat_norms
                sum_saliency.index_add_(0, flat_idx, contrib)
                counts.index_add_(0, flat_idx, torch.ones_like(flat_idx))

    def saliency(self, moe_name: str) -> torch.Tensor:
        """The mean saliency ``S_j`` per expert."""
        stats = self._moe_stats[moe_name]
        return stats["sum_saliency"] / stats["counts"].clamp(min=1.0)
