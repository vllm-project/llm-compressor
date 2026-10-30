"""
REAP (Router-weighted Expert Activation Pruning) modifier for MoE models.

See: https://arxiv.org/abs/2510.13999
"""

import torch
from compressed_tensors.config import MoESparsityMethod

from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modifiers.pruning.base import ExpertPruningModifierBase
from llmcompressor.modifiers.pruning.reap.utils import REAP_PRECISION

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
    _expert_verb = "drop"
    _expert_verb_past = "dropped"

    # -- saliency metric ------------------------------------------------------

    def _init_layer_stats(
        self,
        layer_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        """Per-layer REAP accumulators: ``sum_saliency[j]`` accumulates
        ``g_j * ||f_j||_2`` over the tokens routed to expert ``j`` and
        ``counts[j]`` the number of routed tokens."""
        self._layer_stats[layer_name] = {
            name: torch.zeros(num_experts, dtype=REAP_PRECISION, device=device)
            for name in ("counts", "sum_saliency")
        }

    def _capture_expert_output(self, output: torch.Tensor) -> torch.Tensor:
        """Capture only the L2 norm ``||f_j||_2`` of each token's expert
        output -- all the REAP metric needs -- rather than the full output,
        to bound peak memory."""
        return torch.linalg.norm(output.float(), dim=-1).reshape(-1)

    def _update_stats(
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

        stats = self._layer_stats[layer_name]
        sum_saliency = stats["sum_saliency"]
        counts = stats["counts"]
        self._layer_counts[layer_name] += topk_indices.numel()

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

    def _saliency(self, layer_name: str) -> torch.Tensor:
        """The mean saliency ``S_j`` per expert."""
        stats = self._layer_stats[layer_name]
        return stats["sum_saliency"] / stats["counts"].clamp(min=1.0)
