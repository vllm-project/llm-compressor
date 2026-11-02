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

    # subclass configuration
    sparsity_method = MoESparsityMethod.PRUNE

    # per-batch expert output norms, keyed by layer name then expert index.
    # One slot per expert (None until that expert's hook fires this batch),
    # populated by the expert hooks and discarded by the experts block hook
    # at the end of each batch
    _expert_norms: dict[str, list[torch.Tensor | None]] = PrivateAttr(
        default_factory=lambda: defaultdict(list)
    )

    def init_moe_stats(
        self,
        moe_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        with local_torch_dtype(self.precision), device:
            self._moe_stats[moe_name] = {
                "sum_saliency": torch.zeros(num_experts),
                "count": torch.zeros(num_experts, dtype=torch.int),
            }
            self._expert_norms[moe_name] = [None for _ in range(num_experts)]

    @torch.no_grad()
    def expert_hook(
        self,
        moe_name: str,
        expert_idx: int,
        module: ExpertMLP,
        args: tuple,
        output: torch.Tensor,
    ):
        if moe_name not in self._moe_stats:
            # layer has already been finalized
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
        if moe_name not in self._moe_stats:
            # layer has already been finalized: stop collecting and discard
            # any stale norms
            self._expert_norms.pop(moe_name, None)
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
                for expert_idx, expert_norms in enumerate(self._expert_norms[moe_name]):
                    mask = flat_idx == expert_idx

                    if expert_norms is None:
                        # this expert was not executed this batch, so the
                        # router must not have sent it any tokens
                        assert mask.sum().item() == 0, (
                            f"REAP saliency accumulation: router sent "
                            f"{mask.sum().item()} tokens to expert {expert_idx}, "
                            f"but its hook never fired. This indicates a bug in "
                            f"the expert hook registration or routing extraction "
                            f"logic."
                        )
                        continue

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
                counts.index_add_(
                    0, flat_idx, torch.ones_like(flat_idx, dtype=counts.dtype)
                )

        # discard this batch's norms; the next batch's expert hooks
        # repopulate them
        self._expert_norms[moe_name] = [None] * len(self._expert_norms[moe_name])

    def saliency(self, moe_name: str) -> torch.Tensor:
        """The mean saliency ``S_j`` per expert."""
        stats = self._moe_stats[moe_name]
        return stats["sum_saliency"] / stats["count"].clamp(min=1.0)
