"""
HOPE (Higher-Order Pruning of Experts) modifier for MoE models.

See: https://arxiv.org/abs/2609.18916
"""

import torch
from compressed_tensors.config import MoESparsityMethod
from compressed_tensors.distributed import (
    get_source_rank,
    is_distributed,
    is_source_process,
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
from llmcompressor.modifiers.pruning.hope.utils import _require_scipy, solve_hope_qp

__all__ = ["HOPEPruningModifier"]


class HOPEPruningModifier(ExpertPruningModifierBase):
    """
    Prunes experts from MoE layers using the HOPE (Higher-Order Pruning of
    Experts) objective, which accounts for pairwise expert interactions on
    top of the per-expert importance scored by first-order methods such as
    REAP.

    During calibration, HOPE accumulates a per-layer ``[E, E]`` interaction
    matrix ``F``,

        ``F_ij = mean(g_i * g_j * ||f_i||_2 * ||f_j||_2)``

    where the mean is over the calibration tokens which co-select experts
    ``i`` and ``j`` (``F_ij = 0`` for pairs which are never co-selected),
    ``g`` is the router gate weight and ``||f||_2`` the expert output norm.
    The diagonal ``F_kk`` is expert ``k``'s mean squared gate-weighted output
    norm -- its self-importance, which recovers the squared REAP score up to
    a variance term -- while the off-diagonal ``F_ij`` captures how much the
    pair contributes jointly when co-selected. The prune-set ``P`` is chosen
    to minimize the interaction mass it removes,

        ``min_p p^T F p  s.t.  p in {0, 1}^E, sum_k p_k = |P|``

    where ``p_k = 1`` iff expert ``k`` is pruned. This provably minimizes an
    upper bound on the squared output error introduced by pruning, and
    reduces to REAP's independent expert ranking when the off-diagonal
    interactions are zeroed. The binary program is solved through a
    continuous relaxation (SciPy SLSQP) rounded to the ``|P|`` entries with
    the highest values, so the ``scipy`` package must be installed.

    :param sparsity: fraction of experts to remove per layer, in ``[0, 1)``.
        When ``0.0``, no experts are pruned; the interaction statistics are
        still computed so that a ``report_path`` can be written, but the
        model weights and config are never modified.
    :param ignore: module name patterns to skip during MoE layer pruning.
    :param report_path: optional path to a ``.json`` file where the per-expert
        statistics are written after calibration completes. The report is a JSON
        object mapping ``count`` and ``saliency`` (the diagonal of ``F``) to
        one inner list per MoE layer (ordered by layer). Required when
        ``sparsity`` is ``0.0``, since the report is then the modifier's only
        output.

    Example recipe::

        HOPEPruningModifier:
          sparsity: 0.5
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

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Read the model's MoE structure, failing fast on bad sparsity targets
        and on a missing scipy install.

        The prune-set quadratic program is solved with scipy once calibration
        completes; surfacing a missing install here avoids losing a
        calibration run to an ImportError.

        :param state: The current session state
        :return: True if there are no errors, False otherwise
        """
        initialized = super().on_initialize(state, **kwargs)
        if initialized and self._n_experts_to_drop > 0:
            # the report-only path (sparsity=0.0) never solves the program
            _require_scipy()
        return initialized

    def init_moe_stats(
        self,
        moe_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        """
        Allocate the HOPE accumulators for a layer.

        Creates the pairwise interaction statistics which define the
        interaction matrix -- the sums of gate-weighted norm products and the
        co-selection counts, both ``[num_experts, num_experts]`` and indexed
        by expert pair -- plus the per-expert routed-token counts read by the
        base class and the per-batch expert-norm buffers filled by
        :meth:`expert_hook`.

        :param moe_name: Name of the MoE layer
        :param num_experts: Number of experts in the layer
        :param hidden_dim: Unused by HOPE
        :param device: The layer's execution device
        """
        with local_torch_dtype(self.precision), device:
            self._moe_stats[moe_name] = {
                "count": torch.zeros(num_experts, dtype=torch.int),
                "sum_products": torch.zeros(num_experts, num_experts),
                "pair_count": torch.zeros(num_experts, num_experts, dtype=torch.int),
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
        Add each token's pairwise expert contributions to the layer's
        statistics.

        First gathers each token's per-slot gate-weighted output norms,
        combining the router's top-k weights with the per-batch norms
        recorded by :meth:`expert_hook`. Then, for every pair of slots the
        router selected for a token, accumulates the product of the two
        gate-weighted norms into the corresponding entry of the
        ``sum_products`` matrix and bumps that expert pair's co-selection
        count; each token's routed experts also bump the per-expert
        ``count``. Resets the per-batch norm buffers afterwards.

        Two token layouts are handled: under the calibrate-all-experts flag
        every expert is evaluated for every token, so norms are gathered by
        top-k index; otherwise indices are flattened in (slot, token) order
        to match the ``torch.where`` scatter order of ``LinearExperts2D``.

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

        sum_products = self._moe_stats[moe_name]["sum_products"]
        pair_count = self._moe_stats[moe_name]["pair_count"]
        count = self._moe_stats[moe_name]["count"]

        with local_torch_dtype(self.precision):
            if get_calibrate_all_experts_flag():
                # every expert ran on every token, so stack their per-token
                # norms into a [num_tokens, num_experts] matrix ordered by
                # expert index, then gather each token's routed experts
                norms = self._expert_norms[moe_name]
                stacked_norms = torch.stack(
                    [norms[idx] for idx in sorted(norms)], dim=1
                )
                weighted_norms = topk_weights * stacked_norms.gather(1, topk_indices)
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

                # reshape the slot-major flat values back to per-token rows
                num_tokens, top_k = topk_indices.shape
                weighted_norms = (
                    (flat_weights * flat_norms).reshape(top_k, num_tokens).T
                )

            # Accumulate each token's outer product of gate-weighted norms
            # over its pairs of routed slots: the (a, b) slot pair adds
            # w_a * w_b to the (i, j) entry of the interaction sums and one
            # to its co-selection count, for i and j the experts which occupy
            # slots a and b of that token
            ones = torch.ones(
                topk_indices.shape[0], dtype=pair_count.dtype, device=pair_count.device
            )
            for slot_a in range(topk_indices.shape[1]):
                indices_a = topk_indices[:, slot_a]
                weights_a = weighted_norms[:, slot_a]
                for slot_b in range(topk_indices.shape[1]):
                    indices_b = topk_indices[:, slot_b]
                    products = weights_a * weighted_norms[:, slot_b]
                    sum_products.index_put_(
                        (indices_a, indices_b), products, accumulate=True
                    )
                    pair_count.index_put_((indices_a, indices_b), ones, accumulate=True)

            # per-expert routed-token counts
            flat_routing = topk_indices.reshape(-1)
            count.index_add_(
                0, flat_routing, torch.ones_like(flat_routing, dtype=count.dtype)
            )

        # reset expert norms for next batch
        self._expert_norms[moe_name] = {}

    def reduce_moe_stats(self, moe_name: str):
        """
        Reduce a layer's statistics across ranks, in place.

        All-reduces the per-expert counts, which the base reads to decide
        whether the layer saw calibration data, so every rank makes the same
        decision; reduces the pairwise interaction statistics to the source
        rank, which is the only rank that consumes them (interaction matrix,
        prune set, report).

        :param moe_name: Name of the MoE layer whose stats to reduce
        """
        if not is_distributed():
            return

        stats = self._moe_stats[moe_name]
        pending = [
            dist.all_reduce(stats["count"], op=dist.ReduceOp.SUM, async_op=True),
            dist.reduce(
                stats["sum_products"],
                dst=get_source_rank(),
                op=dist.ReduceOp.SUM,
                async_op=True,
            ),
            dist.reduce(
                stats["pair_count"],
                dst=get_source_rank(),
                op=dist.ReduceOp.SUM,
                async_op=True,
            ),
        ]
        wait_for_comms(pending)

    def saliency(self, moe_name: str) -> torch.Tensor:
        """
        Return the per-expert self-importance, the diagonal of the
        interaction matrix.

        ``S_j = mean((g_j * ||f_j||_2)^2)`` over the tokens routed to expert
        ``j``: the squared REAP score up to a variance term over those
        tokens. Unlike in first-order methods this saliency alone does not
        decide the prune-set; the full interaction matrix does, in
        :meth:`compute_retained_experts`.

        :param moe_name: Name of the MoE layer
        :return: Per-expert saliency, indexed by expert
        """
        stats = self._moe_stats[moe_name]
        count = stats["count"].clamp(min=1).to(self.precision)
        return stats["sum_products"].diagonal() / count

    def _interaction_matrix(self, moe_name: str) -> torch.Tensor:
        """
        Return the layer's ``[num_experts, num_experts]`` interaction matrix.

        ``F_ij = mean(g_i * g_j * ||f_i||_2 * ||f_j||_2)`` over the
        calibration tokens which co-selected experts ``i`` and ``j``, and 0
        for pairs which were never co-selected (conditional normalization,
        which keeps rarely co-selected but strongly contributing pairs from
        being undervalued, following the paper). Symmetrized to absorb
        reduction-order noise.

        :param moe_name: Name of the MoE layer
        :return: The symmetric interaction matrix, indexed by expert pair
        """
        stats = self._moe_stats[moe_name]
        pair_count = stats["pair_count"].clamp(min=1).to(self.precision)
        interaction = stats["sum_products"] / pair_count
        return 0.5 * (interaction + interaction.T)

    def compute_retained_experts(
        self, moe_name: str, device: torch.device
    ) -> tuple[list[int], torch.Tensor]:
        """
        Decide which experts a layer retains by solving the HOPE quadratic
        program over its interaction matrix.

        The prune-set which minimizes the interaction mass it removes is
        computed on the source rank only -- from the globally reduced
        statistics, which only it holds -- and broadcast to every rank, so
        every rank prunes the layer identically; non-source ranks allocate
        placeholder tensors which the broadcast fills.

        :param moe_name: Name of the MoE layer whose stats to read
        :param device: Device on which to allocate the returned tensors
        :return: The retained expert indices in ascending order, and the
            layer's per-expert self-importance (the diagonal of the
            interaction matrix)
        """
        num_experts = self._moe_config.num_experts
        expected = num_experts - self._n_experts_to_drop

        if is_source_process():
            saliency = self.saliency(moe_name)
            interaction = self._interaction_matrix(moe_name)
            if self._n_experts_to_drop_per_group is None:
                dropped = solve_hope_qp(interaction, self._n_experts_to_drop)
            else:
                # group-limited routers require an equal number of drops per
                # group, so the budget applies to each group individually
                n_group = self._moe_config.n_group
                group_size = self._moe_config.group_size
                assert n_group is not None and group_size is not None
                dropped = solve_hope_qp(
                    interaction,
                    self._n_experts_to_drop_per_group,
                    group_sizes=[group_size] * n_group,
                )
            drop_set = set(dropped)
            retained_indices = [i for i in range(num_experts) if i not in drop_set]
            retained = torch.tensor(retained_indices, dtype=torch.int, device=device)
        else:
            # the interaction statistics are reduced to the source rank,
            # which alone computes the prune-set and broadcasts it; only
            # retained is sized to `expected`
            saliency = torch.empty(num_experts, dtype=self.precision, device=device)
            retained = torch.empty(expected, dtype=torch.int, device=device)

        if is_distributed():
            dist.broadcast(saliency, src=get_source_rank())
            dist.broadcast(retained, src=get_source_rank())

        assert (
            len(retained) == expected
        ), f"Expected {expected} retained experts, got {len(retained)}"

        return retained.tolist(), saliency

    def on_finalize(self, state: State, **kwargs) -> bool:
        ret = super().on_finalize(state, **kwargs)
        self._expert_norms.clear()
        return ret
