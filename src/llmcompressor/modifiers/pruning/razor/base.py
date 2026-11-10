"""
RAZOR (pruning Replaceable experts via consensus residuals) modifier for
MoE models.

See: https://arxiv.org/abs/2609.30465
"""

from typing import Literal

import torch
import torch.nn.functional as F
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
from llmcompressor.modifiers.pruning.expert_sparsity.utils import get_experts_router
from llmcompressor.modifiers.utils.hooks import HooksMixin

__all__ = ["RazorPruningModifier"]

# token-level damage criteria (paper section 2.3): ``rcs`` scores each
# expert's routing-weighted consensus residual, ``rcs_loo`` additionally
# renormalizes over the surviving experts, and ``refill`` additionally
# credits the replacement expert the router would promote. RAZOR is
# ``refill`` under conditional RMS aggregation
DamageCriterion = Literal["rcs", "rcs_loo", "refill"]

# lower bound for the deletion denominators (paper appendix A.4.1 clamps
# them from below by 1e-6)
_DENOM_CLAMP = 1e-6

# lower bound for the selected-score sum, which normalizes the mixture
# weights and the refill pseudo-weight
_SCORE_SUM_CLAMP = 1e-20


def _apply_score_function(logits: torch.Tensor, score_function: str) -> torch.Tensor:
    """
    Transform raw router logits into nonnegative mixture scores.

    :param logits: router logits, ``[num_tokens, num_experts]``
    :param score_function: one of ``"softmax"``, ``"sigmoid"``,
        ``"sqrtsoftplus"``
    :return: the nonnegative scores, of the same shape as ``logits``
    """
    if score_function == "softmax":
        return logits.softmax(dim=-1)
    if score_function == "sigmoid":
        return logits.sigmoid()
    if score_function == "sqrtsoftplus":
        return F.softplus(logits).sqrt()
    raise ValueError(f"Unknown score function {score_function!r}")


def _token_damage(
    criterion: DamageCriterion,
    weights: torch.Tensor,
    residuals: torch.Tensor,
    refill_weights: torch.Tensor | None = None,
    refill_residuals: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the token-level deletion damage of routed (token, expert) pairs.

    :param criterion: which damage quantity to evaluate (paper section 2.3):
        ``"rcs"`` is the routing-weighted consensus residual
        ``w_i * ||r_i||_2``, ``"rcs_loo"`` the exact single-deletion output
        change when the vacated slot is not refilled, and ``"refill"`` the
        exact change when the router promotes a replacement expert
    :param weights: ``[num_pairs]`` normalized mixture weights ``w_i`` of
        each pair's expert
    :param residuals: ``[num_pairs, hidden_dim]`` consensus residuals
        ``r_i = f_i - c`` of each pair's expert
    :param refill_weights: ``[num_pairs]`` pseudo-weights ``w_r`` of each
        pair's promoted replacement expert, or ``None`` when no unselected
        expert is available to promote (the refill term then vanishes and
        the refill criterion reduces to the leave-one-out one)
    :param refill_residuals: ``[num_pairs, hidden_dim]`` consensus residuals
        of the promoted replacement experts, aligned with
        ``refill_weights``
    :return: ``[num_pairs]`` deletion damage, one value per pair
    """
    if criterion == "rcs":
        return weights * residuals.norm(dim=-1)

    if criterion == "rcs_loo":
        return weights * residuals.norm(dim=-1) / (1 - weights).clamp_min(_DENOM_CLAMP)

    # criterion == "refill" (proposition 2 of the paper)
    scaled_residual = weights.unsqueeze(-1) * residuals
    if refill_weights is None or refill_residuals is None:
        # no unselected expert is available to promote, so ``w_r = 0`` and
        # the refill formula reduces to its fixed-support special case
        denominator = (1 - weights).clamp_min(_DENOM_CLAMP)
    else:
        scaled_residual = scaled_residual - refill_weights.unsqueeze(-1) * (
            refill_residuals
        )
        denominator = (1 - weights + refill_weights).clamp_min(_DENOM_CLAMP)
    return scaled_residual.norm(dim=-1) / denominator


class RazorPruningModifier(ExpertPruningModifierBase):
    """
    Prunes replaceable experts from MoE layers using the RAZOR saliency
    metric.

    RAZOR scores each expert by the damage its deletion would do to the
    routed mixture output, evaluated at a fixed layer input. Writing the
    routed mixture (the *consensus*) as

        ``c(x) = sum_{i in S(x)} w_i(x) * f_i(x)``

    over the router-selected experts ``S(x)``, their renormalized mixture
    weights ``w_i(x)`` and outputs ``f_i(x)``, the consensus residual of
    expert ``i`` is ``r_i(x) = f_i(x) - c(x)``. Deleting expert ``i``
    renormalizes the surviving weights and, when the router refills the
    vacated slot, promotes the highest-ranked unselected expert ``r`` with
    pseudo-weight ``w_r(x) = s_r(x) / sum_{j in S(x)} s_j(x)`` (its score
    over the selected-score sum). The exact token-level output change of a
    single deletion under this counterfactual is (proposition 2 of the
    paper)

        ``delta_i(x) = ||w_i(x) * r_i(x) - w_r(x) * r_r(x)||_2 / (1 - w_i(x) + w_r(x))``

    RAZOR accumulates ``delta_i^2`` and the routed-token count per expert
    and ranks experts by their conditional root mean square,

        ``S_j = sqrt(sum_t delta_j(t)^2 / max(n_j, 1))``

    over the ``n_j`` calibration tokens routed to expert ``j``, retaining
    the highest-scoring experts within each layer's sparsity budget. The
    ``criterion`` argument selects the token-level damage quantity:

    - ``"refill"`` (default; RAZOR proper): the refill-aware damage above,
    - ``"rcs_loo"``: the leave-one-out damage
      ``w_i * ||r_i||_2 / (1 - w_i)``, the exact single-deletion change when
      the vacated slot is *not* refilled (proposition 3 of the paper), and
    - ``"rcs"``: the routing-weighted consensus residual ``w_i * ||r_i||_2``.

    All criteria use forward computation alone: no gradients, subset search,
    or recovery training.

    The router scores are recomputed each batch from the router's linear
    projection of the experts-block input (the activations the router itself
    consumed), transformed by the router's native score function, which is
    read from the model config (defaulting to ``softmax``). Router linear
    biases are not modeled; an ``e_score_correction_bias``, when present on
    the router or the MoE block, steers which unselected expert is promoted
    -- matching the router's selection rule -- but is not added to the
    mixture weights. Under group-limited routing (``n_group > 1``) the
    promoted expert is approximated by the globally highest-ranked
    unselected expert.

    :param sparsity: fraction of experts to remove per layer, in ``[0, 1)``.
        When ``0.0``, no experts are removed; saliency is still computed so
        that a ``report_path`` can be written, but the model weights and
        config are never modified.
    :param criterion: the token-level deletion damage to aggregate,
        ``"refill"`` (RAZOR), ``"rcs_loo"``, or ``"rcs"``.
    :param ignore: module name patterns to skip during MoE layer detection.
    :param report_path: optional path to a ``.json`` file where the per-expert
        statistics are written after calibration completes. The report is a JSON
        object mapping ``count`` and ``saliency`` to one inner list per MoE layer
        (ordered by layer). Use ``tools/plot_reap_report.py`` to generate a
        saliency visualization. Required when ``sparsity`` is ``0.0``, since the
        report is then the modifier's only output.

    Example recipe::

        RazorPruningModifier:
          sparsity: 0.25
    """

    # subclass configuration
    sparsity_method = MoESparsityMethod.PRUNE

    criterion: DamageCriterion = "refill"

    # per-batch expert outputs, keyed by layer name then expert index,
    # populated by the expert hooks and consumed and reset by the experts
    # block hook at the end of each batch
    _expert_outputs: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )

    # -- statistics and saliency ---------------------------------------------

    def init_moe_stats(
        self,
        moe_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        """
        Allocate the RAZOR accumulators for a layer.

        Creates the per-expert sums of squared token damage (from which the
        conditional RMS saliency is computed) and the routed-token counts,
        plus the per-batch expert-output buffers filled by
        :meth:`expert_hook`.

        :param moe_name: Name of the MoE layer
        :param num_experts: Number of experts in the layer
        :param hidden_dim: Unused by RAZOR
        :param device: The layer's execution device
        """
        with local_torch_dtype(self.precision), device:
            self._moe_stats[moe_name] = {
                "sum_sq_damage": torch.zeros(num_experts),
                "count": torch.zeros(num_experts, dtype=torch.int),
            }
        self._expert_outputs[moe_name] = {}

    def reduce_moe_stats(self, moe_name: str):
        """
        Reduce a layer's statistics across ranks, in place.

        Reduces the damage sums to the source rank, which alone computes
        the pruning decision and report from them, and all-reduces the
        per-expert counts so every rank agrees on whether the layer saw
        calibration data. Summing the squared damage and the counts before
        dividing pools the conditional RMS across ranks, rather than
        averaging per-rank RMS scores.

        :param moe_name: Name of the MoE layer whose stats to reduce
        """
        if not is_distributed():
            return

        stats = self._moe_stats[moe_name]
        pending = [
            dist.reduce(
                stats["sum_sq_damage"],
                dst=get_source_rank(),
                op=dist.ReduceOp.SUM,
                async_op=True,
            ),
            dist.all_reduce(stats["count"], op=dist.ReduceOp.SUM, async_op=True),
        ]
        wait_for_comms(pending)

    def saliency(self, moe_name: str) -> torch.Tensor:
        """
        Return the conditional RMS damage ``S_j`` per expert.

        The root mean square of the token-level deletion damage over the
        tokens routed to each expert; experts which saw no calibration
        tokens score zero.

        :param moe_name: Name of the MoE layer
        :return: Per-expert saliency, indexed by expert
        """
        stats = self._moe_stats[moe_name]
        return (stats["sum_sq_damage"] / stats["count"].clamp(min=1.0)).sqrt()

    # -- calibration hooks ----------------------------------------------------

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
        Record the expert's output for the current batch.

        Stores the output for :meth:`experts_block_hook` to consume at the
        end of the batch: the consensus, the residuals, and the deletion
        damage are all derived from these outputs.

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

        if isinstance(output, tuple):
            output = output[0]
        self._expert_outputs[moe_name][expert_idx] = output.detach()

    @torch.no_grad()
    def experts_block_hook(
        self,
        moe_name: str,
        moe: torch.nn.Module,
        module: torch.nn.Module,
        args: tuple,
        output: torch.Tensor,
    ):
        """
        Accumulate the deletion damage of every routed (token, expert) pair.

        Recomputes the router scores for the batch from the router's linear
        projection of the block input, so that the scores of unselected
        experts are available; renormalizes the selected scores into mixture
        weights and, under the refill criterion, determines the expert each
        token would promote on a deletion (evaluating its output, which the
        block's forward did not compute, since it is unselected). Then forms
        the consensus output of each token from the recorded expert outputs,
        and accumulates each pair's squared damage and count into the layer's
        statistics. Resets the per-batch output buffers afterwards.

        :param moe_name: Name of the MoE layer the block belongs to
        :param moe: The MoE layer the hooked experts block belongs to
        :param module: The experts block the hook is attached to
        :param args: The block's forward arguments, where ``args[0]`` is the
            block input, and ``args[1]`` and ``args[2]`` are the router's
            top-k indices and weights
        :param output: The block's output for the current batch
        """
        if moe_name not in self._moe_stats:
            logger.warning(
                f"Attempted to calibrate layer {moe_name} which was never "
                "initialized or has already been finalized"
            )
            return

        hidden_states: torch.Tensor = args[0]
        topk_indices: torch.Tensor = args[1].to(torch.long)
        num_tokens, top_k = topk_indices.shape
        num_experts = self._moe_config.num_experts
        hidden_dim = hidden_states.shape[-1]
        expert_outputs = self._expert_outputs[moe_name]

        # -- routing context --------------------------------------------------
        # recompute the full score vector (unselected experts included) from
        # the router's linear projection of the block input, which is what
        # the router itself consumed
        experts, router = get_experts_router(moe_name, moe)
        logits = F.linear(
            hidden_states.to(self.precision),
            router.weight.to(device=hidden_states.device, dtype=self.precision),
        )
        scores = _apply_score_function(logits, self._moe_config.score_function)

        # renormalized mixture weights over the selected experts: the damage
        # formula models the normalized mixture, so the selected unmodified
        # scores are renormalized (paper appendix A.4.3)
        selected_scores = scores.gather(1, topk_indices)
        score_sum = selected_scores.sum(dim=-1)
        score_sum = score_sum.clamp_min(_SCORE_SUM_CLAMP)
        weights = selected_scores / score_sum.unsqueeze(-1)

        # -- refill context (Assumption 1 of the paper) -----------------------
        # the rank-(top_k + 1) expert each token would promote on a deletion,
        # with pseudo-weight s_r / sum of selected scores
        refill_weights: torch.Tensor | None = None
        refill_outputs: torch.Tensor | None = None
        if self.criterion == "refill" and num_experts > top_k:
            selection_scores = scores
            # the correction bias steers selection only; it is not added to
            # the mixture weights, and some architectures (e.g. Hy3) place
            # it on the MoE block rather than the router
            bias = getattr(router, "e_score_correction_bias", None)
            if not isinstance(bias, torch.Tensor):
                bias = getattr(moe, "e_score_correction_bias", None)
            if isinstance(bias, torch.Tensor):
                selection_scores = selection_scores + bias.to(selection_scores)

            selected = torch.zeros_like(scores, dtype=torch.bool)
            selected.scatter_(1, topk_indices, True)
            promoted = selection_scores.masked_fill(selected, -float("inf")).argmax(
                dim=-1
            )
            refill_weights = (
                scores.gather(1, promoted.unsqueeze(1)).squeeze(1) / score_sum
            )

            # the promoted experts were not evaluated by the block's forward
            # (they are unselected), so evaluate them here on the tokens they
            # would refill: one additional expert output per token
            refill_outputs = hidden_states.new_zeros(num_tokens, hidden_dim)
            if get_calibrate_all_experts_flag():
                # every expert already ran on every token
                for expert_idx, expert_output in expert_outputs.items():
                    tokens = promoted == expert_idx
                    refill_outputs[tokens] = expert_output[tokens]
            else:
                # the refill forwards are issued by this hook, so disable the
                # expert hooks (including this modifier's) against recording
                # them; their outputs are consumed directly here
                with HooksMixin.disable_hooks():
                    for expert_idx in promoted.unique().tolist():
                        tokens = promoted == expert_idx
                        if expert_idx >= len(experts):
                            continue  # expert no longer exists (already pruned)
                        refill_outputs[tokens] = experts[expert_idx](
                            hidden_states[tokens]
                        )

        # -- consensus ---------------------------------------------------------
        # the original routed mixture output of each token,
        # c = sum_i w_i * f_i over the selected experts
        consensus = torch.zeros(
            num_tokens, hidden_dim, dtype=self.precision, device=hidden_states.device
        )
        calibrate_all = get_calibrate_all_experts_flag()
        for expert_idx, expert_output in expert_outputs.items():
            if calibrate_all:
                # every expert ran on every token, so outputs are token-indexed
                tokens, slots = (topk_indices == expert_idx).nonzero(as_tuple=True)
                contributions = weights[tokens, slots].unsqueeze(-1) * expert_output[
                    tokens
                ].to(self.precision)
            else:
                # experts ran on their routed tokens only, in the (slot,
                # token) order of LinearExperts2D's torch.where scatter
                slots, tokens = (topk_indices.T == expert_idx).nonzero(as_tuple=True)
                contributions = weights[tokens, slots].unsqueeze(-1) * expert_output.to(
                    self.precision
                )
            consensus.index_add_(0, tokens, contributions)

        # -- damage accumulation ------------------------------------------------
        stats = self._moe_stats[moe_name]
        for expert_idx, expert_output in expert_outputs.items():
            if calibrate_all:
                tokens, slots = (topk_indices == expert_idx).nonzero(as_tuple=True)
                expert_output = expert_output[tokens]
            else:
                slots, tokens = (topk_indices.T == expert_idx).nonzero(as_tuple=True)

            if expert_output.shape[0] == 0:
                continue

            pair_weights = weights[tokens, slots]
            residuals = expert_output.to(self.precision) - consensus[tokens]
            if refill_weights is not None:
                pair_refill_weights = refill_weights[tokens]
                pair_refill_residuals = (
                    refill_outputs[tokens].to(self.precision) - consensus[tokens]
                )
            else:
                pair_refill_weights = None
                pair_refill_residuals = None

            damage = _token_damage(
                self.criterion,
                pair_weights,
                residuals,
                pair_refill_weights,
                pair_refill_residuals,
            )
            stats["sum_sq_damage"][expert_idx] += damage.pow(2).sum()
            stats["count"][expert_idx] += damage.shape[0]

        # reset the per-batch output buffers
        self._expert_outputs[moe_name] = {}

    def on_finalize(self, state: State, **kwargs) -> bool:
        ret = super().on_finalize(state, **kwargs)
        self._expert_outputs.clear()
        return ret
