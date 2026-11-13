"""
Unit tests for the HOPE pruning modifier.

The shared pruning lifecycle is tested in
``tests/llmcompressor/modifiers/pruning/test_base.py``; the tests here cover
HOPE's own behavior: the pairwise interaction statistics
``F_ij = mean(g_i * g_j * ||f_i|| * ||f_j||)``, which must agree between the
sparse and calibrate-all execution paths; the prune-set quadratic program,
which must recover REAP's independent ranking when interactions are zeroed
and must account for cooperative expert pairs when they are not; full-model
pruning with real routing; and distributed calibration with unevenly
distributed data.
"""

import json

import pytest
import torch
from torch import distributed as dist

from llmcompressor.core import Event, EventType
from llmcompressor.modeling.moe import context as moe_context
from llmcompressor.modeling.moe.helpers import MoEConfig
from llmcompressor.modeling.moe.linear_experts import ExpertMLP
from llmcompressor.modifiers.pruning.hope import HOPEPruningModifier
from llmcompressor.modifiers.pruning.hope.utils import solve_hope_qp
from tests.llmcompressor.modifiers.pruning.fake_moe import (
    FakeMoEConfig,
    FakeMoEModel,
    make_state,
)
from tests.testing_utils import torchrun


def _stats_modifier(num_experts: int = 4) -> HOPEPruningModifier:
    """A HOPE modifier with initialized statistics for a single layer."""
    modifier = HOPEPruningModifier(sparsity=0.5)
    modifier._moe_config = MoEConfig.from_config(FakeMoEConfig(num_experts=num_experts))
    modifier._n_experts_to_drop = num_experts // 2
    modifier.init_moe_stats("layer", num_experts, 2, torch.device("cpu"))
    return modifier


def _run_batch(
    modifier: HOPEPruningModifier,
    expert_outputs: dict[int, torch.Tensor],
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
):
    """Feed one calibration batch through the modifier's hooks.

    ``expert_outputs`` maps each expert index to the output its forward hook
    saw for the batch: the expert's routed tokens only under sparse
    execution, and every token under calibrate-all execution (where unrouted
    experts still run, on an empty batch of tokens or all tokens
    respectively)."""
    for expert_idx, output in expert_outputs.items():
        modifier.expert_hook("layer", expert_idx, None, (), output)
    hidden = torch.zeros(topk_indices.shape[0], 1)
    modifier.experts_block_hook(
        "layer", None, (hidden, topk_indices, topk_weights), None
    )


# ---------------------------------------------------------------------------
# Tests: interaction statistics accumulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("calibrate_all", [False, True])
def test_interaction_statistics_are_mean_pair_products(calibrate_all, monkeypatch):
    """``F_ij = mean(g_i * g_j * ||f_i|| * ||f_j||)`` over the tokens which
    co-select experts ``i`` and ``j`` (0 when never co-selected), and the
    per-expert counts are the routed-token counts, accumulating across
    batches. The sparse and calibrate-all execution paths must produce the
    same statistics from the same routing."""
    monkeypatch.setattr(moe_context, "_CALIBRATE_ALL_EXPERTS", calibrate_all)
    modifier = _stats_modifier(num_experts=4)

    # 2 tokens, top-2 routing; expert 3 is never routed
    topk_indices = torch.tensor([[0, 1], [0, 2]])
    topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]])

    if calibrate_all:
        # dense execution: every expert runs on every token
        outputs = {
            0: torch.tensor([[3.0], [1.0]]),
            1: torch.tensor([[2.0], [5.0]]),
            2: torch.tensor([[6.0], [4.0]]),
            3: torch.tensor([[7.0], [8.0]]),
        }
    else:
        # sparse execution: experts run on their routed tokens only, so an
        # unrouted expert still fires its hook on an empty batch of tokens
        outputs = {
            0: torch.tensor([[3.0], [1.0]]),
            1: torch.tensor([[2.0]]),
            2: torch.tensor([[4.0]]),
            3: torch.empty(0, 1),
        }
    _run_batch(modifier, outputs, topk_indices, topk_weights)

    # token 0 routes gate-weighted norms (1.8, 0.8) to experts (0, 1);
    # token 1 routes (0.5, 2.0) to experts (0, 2)
    expected = torch.tensor(
        [
            [1.745, 1.44, 1.0, 0.0],
            [1.44, 0.64, 0.0, 0.0],
            [1.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(
        modifier._interaction_matrix("layer"), expected, rtol=0, atol=1e-6
    )

    # the diagonal is the per-expert saliency, the squared gate-weighted
    # norm averaged over each expert's routed tokens
    assert modifier.saliency("layer").tolist() == pytest.approx([1.745, 0.64, 4.0, 0.0])
    assert modifier._moe_stats["layer"]["count"].tolist() == [2, 1, 1, 0]

    # a second batch accumulates on top of the first
    second_outputs = (
        {i: torch.tensor([[5.0]]) for i in range(4)}
        if calibrate_all
        else {i: torch.empty(0, 1) for i in range(4)}
    )
    second_outputs[0] = torch.tensor([[1.0]])
    _run_batch(modifier, second_outputs, torch.tensor([[0]]), torch.tensor([[0.4]]))

    # expert 0 now has three routed tokens with gate-weighted norms
    # (1.8, 0.5, 0.4), so F_00 is the mean of their squares
    assert modifier._interaction_matrix("layer")[0, 0].item() == pytest.approx(
        (1.8**2 + 0.5**2 + 0.4**2) / 3
    )
    assert modifier._moe_stats["layer"]["count"].tolist() == [3, 1, 1, 0]


@pytest.mark.unit
def test_interaction_statistics_agree_between_sparse_and_calibrate_all_execution(
    monkeypatch,
):
    """Real forward passes produce the same interaction statistics whether
    experts run on their routed tokens only or on every token
    (calibrate-all)."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    reference_model = FakeMoEModel(config)
    reference_model.eval()
    reference_state = reference_model.state_dict()

    stats_by_mode = {}
    for calibrate_all in (False, True):
        monkeypatch.setattr(moe_context, "_CALIBRATE_ALL_EXPERTS", calibrate_all)
        model = FakeMoEModel(config)
        model.load_state_dict(reference_state)
        model.eval()

        modifier = HOPEPruningModifier(sparsity=0.5)
        state = make_state(model)
        modifier.initialize(state)
        modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

        torch.manual_seed(100)  # same calibration data in both modes
        with torch.no_grad():
            for _ in range(5):
                model(torch.randn(2, 8, config.hidden_size))

        stats_by_mode[calibrate_all] = {
            layer_name: (
                modifier.saliency(layer_name).clone(),
                modifier._interaction_matrix(layer_name).clone(),
            )
            for layer_name in modifier._moe_stats
        }

    # the two paths sum the same contributions in a different order, so
    # allow for float32 reduction-order noise
    for layer_name, (saliency, interaction) in stats_by_mode[False].items():
        dense_saliency, dense_interaction = stats_by_mode[True][layer_name]
        torch.testing.assert_close(saliency, dense_saliency, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(interaction, dense_interaction, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: prune-set quadratic program
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_solve_qp_recovers_first_order_ranking_without_interactions():
    """With zeroed off-diagonals HOPE reduces to first-order ranking: the
    experts with the least self-importance are pruned, recovering REAP's
    prune-set (the paper's Figure S4)."""
    diagonal = torch.tensor([0.1, 0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4])
    interaction = torch.diag(diagonal)

    # the 3 lowest-diagonal experts (0, 5, 3) are pruned
    assert solve_hope_qp(interaction, num_to_drop=3) == [0, 3, 5]

    # a zero budget prunes nothing
    assert solve_hope_qp(interaction, num_to_drop=0) == []


@pytest.mark.unit
def test_solve_qp_accounts_for_cooperative_experts():
    """Second-order behavior: a cooperative expert pair is split up rather
    than jointly pruned, even though both members' self-importance is lower
    than the isolated experts'. A first-order method ranks by the diagonal
    alone and prunes the pair."""
    interaction = torch.tensor(
        [
            [3.9, 5.0, 0.0, 0.0],  # cooperative pair (0, 1): low diagonal,
            [5.0, 4.0, 0.0, 0.0],  # strong joint contribution
            [0.0, 0.0, 4.5, 0.0],  # isolated experts: higher diagonal
            [0.0, 0.0, 0.0, 4.6],
        ]
    )

    def prune_cost(dropped: list[int]) -> float:
        """The interaction mass removing ``dropped`` carries away."""
        indices = torch.tensor(dropped)
        return interaction[indices][:, indices].sum().item()

    dropped = solve_hope_qp(interaction, num_to_drop=2)

    # pruning both pair members (0, 1) costs 3.9 + 4.0 + 2 * 5.0 = 17.9,
    # while splitting the pair by pruning (0, 2) costs only 8.4; a
    # first-order ranking would prune the pair itself
    assert dropped == [0, 2]
    first_order = [0, 1]
    assert prune_cost(dropped) < prune_cost(first_order)


@pytest.mark.unit
def test_solve_qp_is_balanced_per_group():
    """Group-limited routers keep every group routable, so the budget is
    spent equally across groups even when the globally cheapest experts
    concentrate in one group."""
    # two groups of 4: [0, 4) and [4, 8); the two globally-cheapest experts
    # (1 and 2) share the first group
    diagonal = torch.tensor([5.0, 1.0, 1.2, 5.0, 2.0, 5.0, 5.0, 5.0])
    interaction = torch.diag(diagonal)

    dropped = solve_hope_qp(interaction, num_to_drop=1, group_sizes=[4, 4])

    # the cheapest expert of each group is pruned, not the two globally
    # cheapest
    assert dropped == [1, 4]

    # the groups must partition the layer's experts
    with pytest.raises(ValueError, match="partition"):
        solve_hope_qp(interaction, num_to_drop=1, group_sizes=[3, 3])


@pytest.mark.unit
def test_compute_retained_experts_is_balanced_per_group():
    """Group-limited routers must keep every group routable, so the drop is
    split equally across groups."""
    modifier = HOPEPruningModifier(sparsity=0.25)  # 2 of 8, 1 per group
    modifier.initialize(
        make_state(FakeMoEModel(FakeMoEConfig(num_experts=8, n_group=2, topk_group=1)))
    )
    modifier.init_moe_stats("layer", 8, 4, torch.device("cpu"))
    stats = modifier._moe_stats["layer"]

    # diagonal self-importance, no pairwise interactions: group [0, 4) drops
    # its cheapest expert 0, group [4, 8) drops its cheapest expert 5
    stats["sum_products"] = torch.diag(
        torch.tensor([0.1, 0.9, 0.8, 0.7, 0.6, 0.2, 0.5, 0.4])
    )
    stats["pair_count"] = torch.eye(8, dtype=torch.int)
    stats["count"] = torch.ones(8, dtype=torch.int)

    retained, _ = modifier.compute_retained_experts("layer", stats["count"].device)

    assert retained == [1, 2, 3, 4, 6, 7]


@pytest.mark.unit
def test_missing_scipy_fails_fast(monkeypatch):
    """A missing scipy install surfaces at initialization, before any
    calibration time is spent, and in the solver itself."""
    from llmcompressor.modifiers.pruning.hope import utils as hope_utils

    monkeypatch.setattr(hope_utils, "scipy_minimize", None)

    with pytest.raises(ImportError, match="scipy"):
        solve_hope_qp(torch.eye(4), num_to_drop=1)

    modifier = HOPEPruningModifier(sparsity=0.5)
    with pytest.raises(ImportError, match="scipy"):
        modifier.initialize(make_state(FakeMoEModel(FakeMoEConfig())))


@pytest.mark.unit
def test_report_only_mode_runs_without_scipy(tmp_path, monkeypatch):
    """The report-only path (``sparsity=0.0``) never solves the quadratic
    program, so it completes without scipy and still writes its report."""
    from llmcompressor.modifiers.pruning.hope import utils as hope_utils

    monkeypatch.setattr(hope_utils, "scipy_minimize", None)

    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    report_path = str(tmp_path / "report.json")
    modifier = HOPEPruningModifier(sparsity=0.0, report_path=report_path)
    state = make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    with torch.no_grad():
        model(torch.randn(2, 8, config.hidden_size))
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # no layer was pruned, but the report was still written
    for layer in model.layers:
        assert layer.experts.num_experts == 8
    with open(report_path) as f:
        report = json.load(f)
    assert len(report["saliency"]) == 2
    assert len(report["saliency"][0]) == 8


# ---------------------------------------------------------------------------
# Tests: full-model lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hope_full_lifecycle(tmp_path):
    """Calibration accumulates interaction statistics through real routing,
    ``SEQUENTIAL_EPOCH_END`` prunes every layer by solving the quadratic
    program, and the pruning is recorded in the expert-sparsity config while
    the model config is left untouched."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    report_path = str(tmp_path / "report.json")
    modifier = HOPEPruningModifier(sparsity=0.5, report_path=report_path)
    state = make_state(model)

    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert len(modifier._moe_stats) == 2

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    # every layer's statistics were consumed
    assert len(modifier._moe_stats) == 0

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    assert model.config.num_experts == 8
    assert model._moe_sparsity_config.method.value == "prune"
    assert model._moe_sparsity_config.num_experts == [4, 4]

    for layer in model.layers:
        expert_count = sum(
            1 for m in layer.experts.children() if isinstance(m, ExpertMLP)
        )
        assert expert_count == 4
        assert layer.experts.num_experts == 4
        assert layer.gate.weight.shape[0] == 4
        assert layer.gate.out_features == 4

    # the report holds the per-expert statistics, ordered by layer
    with open(report_path) as f:
        report = json.load(f)
    assert sorted(report) == ["count", "saliency"]
    for values in report.values():
        assert len(values) == 2
        assert len(values[0]) == 8

    # the pruned model still runs
    with torch.no_grad():
        out = model(torch.randn(2, 4, config.hidden_size))
    assert out.shape == (2, 4, config.hidden_size)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Tests: distributed calibration
# ---------------------------------------------------------------------------


@pytest.mark.unit
@torchrun(world_size=2, init_dist=False)
def test_hope_distributed_uneven_data():
    """Layers must be reduced and pruned identically on all ranks when
    calibration data is unevenly distributed across ranks.

    Every rank reduces every tracked layer (contributing zeros where it saw
    no tokens) and the decision of which layers collected samples is made
    from the total count summed across ranks. The prune-set is solved on the
    source rank from the globally reduced interaction statistics and
    broadcast, covering both the case where only the non-source rank saw
    data and the case where only the source rank saw data.
    """
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        rank = dist.get_rank()

        for data_rank in (1, 0):
            torch.manual_seed(42)
            config = FakeMoEConfig(
                num_experts=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
            )
            model = FakeMoEModel(config)
            model.eval()

            modifier = HOPEPruningModifier(sparsity=0.5)
            state = make_state(model)

            modifier.initialize(state)
            modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

            # only data_rank processes calibration data
            if rank == data_rank:
                with torch.no_grad():
                    for _ in range(5):
                        model(torch.randn(2, 8, config.hidden_size))

            modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
            modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
            modifier.finalize(state)

            # every layer was pruned from the globally summed statistics ...
            for layer in model.layers:
                assert layer.experts.num_experts == 4
                assert layer.gate.weight.shape[0] == 4

            # ... identically on every rank (retained experts are broadcast
            # from the source rank)
            gate = torch.stack([layer.gate.weight.data for layer in model.layers]).to(
                torch.float64
            )
            gathered = [torch.empty_like(gate) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, gate.contiguous())
            for other in gathered:
                torch.testing.assert_close(other, gate)

        dist.barrier()
    finally:
        dist.destroy_process_group()
