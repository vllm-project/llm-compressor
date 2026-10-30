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
)
from torch import distributed as dist

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
    _expert_verb = "collapse"
    _expert_verb_past = "collapsed"

    # n_group == 1 is a degenerate "group-limited" router: a single group
    # holds every expert, so routing is equivalent to ungrouped top-k and
    # MoNe's uniform collapse is correct. Only genuine group-limited routers
    # (n_group > 1), which would need group-balanced collapse, are unsupported
    supports_group_limited_routers = False

    # -- saliency metric ------------------------------------------------------

    def _init_layer_stats(
        self,
        layer_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        """Per-layer MoNe accumulators. Output statistics (``sum_output``,
        ``sum_sq_output``, ``counts``): the mean output is the novice vector,
        and the per-dimension variance gives the fluctuation term of the
        saliency metric. Routing statistics (``sum_gates``): the
        frequency-weighted mean gate weight gives the routing score term."""
        self._layer_stats[layer_name] = {
            "counts": torch.zeros(num_experts, dtype=torch.float32, device=device),
            "sum_gates": torch.zeros(num_experts, dtype=torch.float32, device=device),
            "sum_output": torch.zeros(
                num_experts, hidden_dim, dtype=torch.float32, device=device
            ),
            "sum_sq_output": torch.zeros(
                num_experts, hidden_dim, dtype=torch.float32, device=device
            ),
        }

    def _update_stats(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_outputs_dict: dict[int, torch.Tensor],
    ):
        """Combine the router decisions with the captured expert outputs to
        update the routing and novice accumulators."""
        self._update_routing(layer_name, topk_indices, topk_weights)
        self._update_outputs(layer_name, topk_indices, expert_outputs_dict)

    def _update_routing(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """
        Accumulate the frequency-weighted router gate weight per expert.

        ``sum_gates[j]`` collects ``g_j`` summed over the tokens for which ``j`` is
        in the top-k, and ``_layer_counts[layer_name]`` counts all tokens, so
        that ``routing_score_j = sum_gates[j] / layer_counts`` equals
        ``E_tokens[ g_j * 1(j in top-k) ]``.

        :param topk_indices: ``[num_tokens, top_k]`` selected expert ids
        :param topk_weights: ``[num_tokens, top_k]`` gate weight per selection
        """
        flat_idx = topk_indices.reshape(-1).to(torch.long)
        flat_weights = topk_weights.reshape(-1).to(torch.float32)
        self._layer_stats[layer_name]["sum_gates"].index_add_(0, flat_idx, flat_weights)
        self._layer_counts[layer_name] += topk_indices.shape[0]

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

        stats = self._layer_stats[layer_name]
        calibrate_all = get_calibrate_all_experts_flag()
        for expert_idx, output in expert_outputs_dict.items():
            if calibrate_all:
                mask = (topk_indices == expert_idx).any(dim=-1)
                output = output[mask]
            if output.shape[0] == 0:
                continue
            output = output.to(torch.float32)
            stats["sum_output"][expert_idx] += output.sum(dim=0)
            stats["sum_sq_output"][expert_idx] += output.pow(2).sum(dim=0)
            stats["counts"][expert_idx] += output.shape[0]

    def _routing_score(self, layer_name: str) -> torch.Tensor:
        """``[num_experts]`` frequency-weighted mean gate weight."""
        token_count = self._layer_counts[layer_name]
        sum_gates = self._layer_stats[layer_name]["sum_gates"]
        if token_count <= 0:
            return torch.zeros_like(sum_gates)
        return sum_gates / float(token_count)

    def _output_fluctuation(self, layer_name: str) -> torch.Tensor:
        """``[num_experts]`` L2 norm (over hidden dims) of the per-dimension
        standard deviation of each expert's output."""
        stats = self._layer_stats[layer_name]
        count = stats["counts"].clamp(min=1.0).unsqueeze(-1)
        mean = stats["sum_output"] / count
        variance = (stats["sum_sq_output"] / count - mean.pow(2)).clamp_min(0.0)
        std = variance.sqrt()
        return torch.linalg.norm(std, dim=1)

    def _saliency(self, layer_name: str) -> torch.Tensor:
        """The MoNe fusion saliency ``fluctuation * routing_score`` per expert."""
        return self._output_fluctuation(layer_name) * self._routing_score(layer_name)

    def _compute_novices(self, layer_name: str, dropped: list[int]) -> torch.Tensor:
        """Return the ``(len(dropped), hidden_dim)`` matrix of novice vectors --
        the mean output of each collapsed expert, ordered to match ``dropped``.
        """
        stats = self._layer_stats[layer_name]
        count = stats["counts"].clamp(min=1.0).unsqueeze(-1)
        mean_output = stats["sum_output"] / count
        drop_idx = torch.tensor(dropped, dtype=torch.long, device=mean_output.device)
        return mean_output[drop_idx]

    # -- collapse --------------------------------------------------------------

    def _apply_retained_experts(
        self,
        model: torch.nn.Module,
        layer_name: str,
        retained: list[int],
    ):
        """Collapse the layer's dropped experts into novices: their mean
        outputs (computed on the source rank, which holds the globally
        reduced statistics, and broadcast to every rank) are installed as
        constant novice vectors while the router keeps its full width."""
        sum_output = self._layer_stats[layer_name]["sum_output"]
        num_experts = sum_output.shape[0]
        retained_set = set(retained)
        dropped = [i for i in range(num_experts) if i not in retained_set]

        if is_source_process():
            novices = self._compute_novices(layer_name, dropped)
        else:
            novices = torch.empty(
                len(dropped),
                sum_output.shape[1],
                dtype=torch.float32,
                device=sum_output.device,
            )

        if is_distributed():
            dist.broadcast(novices, src=get_source_rank())

        collapse_moe_layer(model, layer_name, retained, novices, self._moe_attrs)
