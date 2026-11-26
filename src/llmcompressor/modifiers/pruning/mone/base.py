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
from llmcompressor.modifiers.pruning.mone.utils import mone_collapse_moe_layer

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
    :param ignore: module name patterns to skip during MoE layer pruning.
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
    supports_group_limited_routers = False

    # per-batch expert outputs, keyed by layer name then expert index. One
    # slot per expert (empty until that expert's hook fires this batch),
    # populated by the expert hooks and consumed and reset by the experts
    # block hook at the end of each batch
    _expert_outputs: dict[str, dict[int, torch.Tensor]] = PrivateAttr(
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
        Allocate the per-layer MoNe accumulators.

        Output statistics (``sum_output``, ``sum_sq_output``, ``count``):
        the mean output is the novice vector, and the per-dimension variance
        gives the fluctuation term of the saliency metric. Routing statistics
        (``sum_gates``, ``token_count``): the frequency-weighted mean gate
        weight gives the routing score term.

        ``count`` (per-expert routed-token counts) is all-reduced by
        :meth:`reduce_moe_stats` so that every rank decides identically
        whether a layer saw calibration data; the remaining accumulators,
        including the scalar ``token_count``, are reduced to the source rank,
        which is the only rank that consumes them (saliency, novices, report).

        :param moe_name: Name of the MoE layer
        :param num_experts: Number of experts in the layer
        :param hidden_dim: Hidden dimension of the layer, sizing the output
            accumulators
        :param device: The layer's execution device
        """
        with local_torch_dtype(self.precision), device:
            self._moe_stats[moe_name] = {
                "count": torch.zeros(num_experts),
                "sum_gates": torch.zeros(num_experts),
                "sum_output": torch.zeros(num_experts, hidden_dim),
                "sum_sq_output": torch.zeros(num_experts, hidden_dim),
                "token_count": torch.zeros(()),
            }
        self._expert_outputs[moe_name] = {}

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
        Record the expert's full output for every token in this batch.

        Captures expert ``f_j``'s output for :meth:`experts_block_hook` to
        consume at the end of the batch; it feeds the novice mean and the
        output-fluctuation term of the saliency.

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
        Update the routing and novice accumulators for the batch.

        Combines the router's top-k decisions with the captured expert
        outputs, then resets the per-batch output buffers.

        :param moe_name: Name of the MoE layer the block belongs to
        :param moe: The MoE layer the hooked experts block belongs to
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

        expert_outputs = self._expert_outputs[moe_name]

        # update routing stats
        stats = self._moe_stats[moe_name]
        flat_idx = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1).to(self.precision)
        stats["sum_gates"].index_add_(0, flat_idx, flat_weights)
        stats["token_count"] += topk_indices.shape[0]

        # update output stats
        stats = self._moe_stats[moe_name]
        calibrate_all = get_calibrate_all_experts_flag()
        for expert_idx, output in expert_outputs.items():
            if calibrate_all:
                mask = (topk_indices == expert_idx).any(dim=-1)
                output = output[mask]
            if output.shape[0] == 0:
                continue
            output = output.to(self.precision)
            stats["sum_output"][expert_idx] += output.sum(dim=0)
            stats["sum_sq_output"][expert_idx] += output.pow(2).sum(dim=0)
            stats["count"][expert_idx] += output.shape[0]

        # reset the per-batch output buffers
        self._expert_outputs[moe_name] = {}

    def reduce_moe_stats(self, moe_name: str):
        """
        Reduce a layer's statistics across ranks, in place.

        All-reduces the per-expert counts, which the base reads to decide
        whether the layer saw calibration data, so every rank makes the same
        decision; reduces the remaining accumulators (including the scalar
        ``token_count``) to the source rank, which is the only rank that
        consumes them (saliency, novices, report).

        :param moe_name: Name of the MoE layer whose stats to reduce
        """
        if not is_distributed():
            return

        stats = self._moe_stats[moe_name]
        src_rank = get_source_rank()
        pending = [dist.all_reduce(stats["count"], op=dist.ReduceOp.SUM, async_op=True)]
        pending += [
            dist.reduce(tensor, dst=src_rank, op=dist.ReduceOp.SUM, async_op=True)
            for name, tensor in stats.items()
            if name != "count"
        ]
        wait_for_comms(pending)

    def saliency(self, moe_name: str) -> torch.Tensor:
        """
        Return the MoNe fusion saliency ``fluctuation * routing_score``.

        :param moe_name: Name of the MoE layer
        :return: Per-expert saliency, indexed by expert
        """
        return self._output_fluctuation(moe_name) * self._routing_score(moe_name)

    def _routing_score(self, moe_name: str) -> torch.Tensor:
        """
        Return the ``[num_experts]`` frequency-weighted mean gate weight.

        :param moe_name: Name of the MoE layer
        :return: Per-expert routing score, zero when the layer saw no tokens
        """
        stats = self._moe_stats[moe_name]
        token_count = stats["token_count"]
        if token_count <= 0:
            return torch.zeros_like(stats["sum_gates"])
        return stats["sum_gates"] / token_count

    def _output_fluctuation(self, moe_name: str) -> torch.Tensor:
        """
        Return the ``[num_experts]`` output fluctuation of each expert.

        The L2 norm (over hidden dims) of the per-dimension standard
        deviation of each expert's output.

        :param moe_name: Name of the MoE layer
        :return: Per-expert output fluctuation
        """
        stats = self._moe_stats[moe_name]
        count = stats["count"].clamp(min=1.0).unsqueeze(-1)
        mean = stats["sum_output"] / count
        variance = (stats["sum_sq_output"] / count - mean.pow(2)).clamp_min(0.0)
        std = variance.sqrt()
        return torch.linalg.norm(std, dim=1)

    def prune_moe_layer(
        self,
        moe_name: str,
        moe_module: torch.nn.Module,
        retained: list[int],
    ):
        """
        Collapse the layer's dropped experts into novices.

        Their mean outputs (computed on the source rank, which holds the
        globally reduced statistics, and broadcast to every rank) are
        installed as constant novice vectors while the router keeps its
        full width.

        :param moe_name: Name of the MoE layer to collapse
        :param moe_module: The MoE block module to collapse
        :param retained: Indices of the experts to keep, in their new order
        """
        num_experts = self._moe_config.num_experts
        retained_set = set(retained)
        dropped = [i for i in range(num_experts) if i not in retained_set]

        # novices are computed from the globally reduced statistics, which
        # only the source rank holds; they are broadcast to every rank
        sum_output = self._moe_stats[moe_name]["sum_output"]
        if is_source_process():
            novices = self._compute_novices(moe_name, dropped)
        else:
            novices = torch.empty(
                len(dropped),
                sum_output.shape[1],
                dtype=self.precision,
                device=sum_output.device,
            )

        if is_distributed():
            dist.broadcast(novices, src=get_source_rank())

        mone_collapse_moe_layer(moe_name, moe_module, retained, dropped, novices)

    def _compute_novices(self, moe_name: str, dropped: list[int]) -> torch.Tensor:
        """
        Return the novice vectors for the dropped experts.

        Each novice is its collapsed expert's mean output; rows are ordered
        to match ``dropped``.

        :param moe_name: Name of the MoE layer
        :param dropped: Indices of the experts being collapsed
        :return: The ``(len(dropped), hidden_dim)`` matrix of novice vectors
        """
        stats = self._moe_stats[moe_name]
        num_uncalibrated = (stats["count"] == 0).sum().item()
        if num_uncalibrated > 0:
            logger.warning(
                f"Found {num_uncalibrated} uncalibrated experts, using "
                "zero-vector novices for these experts"
            )

        count = stats["count"].clamp(min=1.0).unsqueeze(-1)
        mean_output = stats["sum_output"] / count
        drop_idx = torch.tensor(dropped, dtype=torch.long, device=mean_output.device)
        return mean_output[drop_idx]

    def on_finalize(self, state: State, **kwargs) -> bool:
        ret = super().on_finalize(state, **kwargs)
        self._expert_outputs.clear()
        return ret
