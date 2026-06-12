import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.modifiers.pruning.reap.utils import (
    GLM_DSA,
    INDICES_WEIGHTS,
    SCORES_LOGITS,
    SOFTMAX,
    WEIGHTS_INDICES,
    MoEModelAttrs,
    REAPSaliencyTracker,
    assert_routing_feasible,
    compute_retained_experts,
    detect_moe_attrs,
    extract_routing,
    find_moe_layers,
    get_num_experts,
    get_router_num_groups,
    prune_moe_layer,
)

# ---------------------------------------------------------------------------
# Helpers: tiny synthetic MoE model for testing
# ---------------------------------------------------------------------------


class FakeExpert(nn.Module):
    """A simple feedforward expert."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class FakeMoEBlock(nn.Module):
    """A minimal MoE block with a gate router and expert ModuleList."""

    calibrate_all_experts = True

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FakeExpert(hidden_dim, hidden_dim * 2) for _ in range(num_experts)]
        )
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        final = torch.zeros_like(flat)
        expert_mask = F.one_hot(topk_indices, self.num_experts).permute(2, 1, 0)

        for expert_idx, expert in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            out = expert(flat)
            if len(top_x) > 0:
                weighted = out[top_x] * topk_weights[top_x, idx, None]
                final.index_add_(0, top_x, weighted.to(flat.dtype))

        return final.view(batch_size, seq_len, hidden_dim), router_logits


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_experts, top_k):
        super().__init__()
        self.mlp = FakeMoEBlock(hidden_dim, num_experts, top_k)

    def forward(self, x):
        out, _ = self.mlp(x)
        return x + out


class FakeConfig:
    def __init__(self, num_experts, top_k):
        self.num_experts = num_experts
        self.num_experts_per_tok = top_k


class FakeMoEModel(nn.Module):
    """A tiny 2-layer MoE model for testing."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_experts: int = 8,
        top_k: int = 2,
        num_layers: int = 2,
    ):
        super().__init__()
        self.config = FakeConfig(num_experts, top_k)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [
                FakeDecoderLayer(hidden_dim, num_experts, top_k)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Tests: REAPSaliencyTracker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestREAPSaliencyTracker:
    def test_update_and_mean(self):
        tracker = REAPSaliencyTracker(num_experts=4)
        # 2 tokens, top_k=2, 4 experts; expert 3 is never routed
        topk_indices = torch.tensor([[0, 1], [0, 2]])
        topk_weights = torch.tensor([[0.6, 0.4], [0.5, 0.5]])
        expert_norms = torch.tensor(
            [
                [3.0, 2.0, 9.0, 5.0],  # token 0 output norm per expert
                [1.0, 9.0, 4.0, 5.0],  # token 1
            ]
        )

        tracker.update(topk_indices, topk_weights, expert_norms)

        mean = tracker.mean_saliency
        # expert 0: (0.6*3.0 + 0.5*1.0) / 2 = 2.3 / 2 = 1.15
        assert mean[0].item() == pytest.approx(1.15)
        # expert 1: 0.4*2.0 / 1 = 0.8
        assert mean[1].item() == pytest.approx(0.8)
        # expert 2: 0.5*4.0 / 1 = 2.0
        assert mean[2].item() == pytest.approx(2.0)
        # expert 3: never routed -> 0
        assert mean[3].item() == pytest.approx(0.0)

    def test_accumulates_across_batches(self):
        tracker = REAPSaliencyTracker(num_experts=2)
        idx = torch.tensor([[0]])
        norms = torch.tensor([[3.0, 0.0]])
        tracker.update(idx, torch.tensor([[0.6]]), norms)
        tracker.update(idx, torch.tensor([[0.4]]), torch.tensor([[1.0, 0.0]]))
        # expert 0: (0.6*3.0 + 0.4*1.0) / 2 = 2.2 / 2 = 1.1
        assert tracker.mean_saliency[0].item() == pytest.approx(1.1)
        assert tracker.total_count == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Tests: routing extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRoutingExtraction:
    def test_softmax_mode(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 8)
        block = nn.Module()
        block.norm_topk_prob = True
        attrs = MoEModelAttrs("gate", "experts", "num_experts", SOFTMAX)

        idx, weights = extract_routing(logits, block, attrs, top_k=2)

        rw = F.softmax(logits, dim=-1, dtype=torch.float32)
        exp_w, exp_idx = torch.topk(rw, 2, dim=-1)
        exp_w = exp_w / exp_w.sum(dim=-1, keepdim=True)
        assert torch.equal(idx, exp_idx)
        torch.testing.assert_close(weights, exp_w)

    def test_indices_weights_mode(self):
        attrs = MoEModelAttrs("gate", "experts", "n_routed_experts", INDICES_WEIGHTS)
        topk_indices = torch.tensor([[1, 3], [0, 2]])
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
        idx, w = extract_routing((topk_indices, topk_weights), nn.Module(), attrs, 2)
        assert torch.equal(idx, topk_indices)
        torch.testing.assert_close(w, topk_weights)

    def test_weights_indices_mode(self):
        attrs = MoEModelAttrs("router", "experts", "num_experts", WEIGHTS_INDICES)
        topk_weights = torch.tensor([[0.7, 0.3]])
        topk_indices = torch.tensor([[1, 3]])
        idx, w = extract_routing((topk_weights, topk_indices), nn.Module(), attrs, 2)
        assert torch.equal(idx, topk_indices)
        torch.testing.assert_close(w, topk_weights)

    def test_scores_logits_mode(self):
        attrs = MoEModelAttrs("router", "experts", "num_local_experts", SCORES_LOGITS)
        scores = torch.tensor([[0.1, 0.9, 0.5, 0.2]])
        logits = torch.tensor([[1.0, 5.0, 3.0, 0.5]])
        idx, w = extract_routing((scores, logits), nn.Module(), attrs, 2)
        # top-2 logits are experts 1 (5.0) and 2 (3.0)
        assert set(idx[0].tolist()) == {1, 2}
        # weights are the gathered scores at those experts
        gathered = scores.gather(-1, idx)
        torch.testing.assert_close(w, gathered)

    def test_glm_dsa_mode_single_group(self):
        # with one group, glm_dsa reduces to sigmoid scores + top-k
        attrs = MoEModelAttrs("gate", "experts", "n_routed_experts", GLM_DSA)
        block = nn.Module()
        block.n_group = 1
        block.topk_group = 1
        block.norm_topk_prob = False
        block.routed_scaling_factor = 1.0
        block.gate = nn.Module()
        block.gate.e_score_correction_bias = torch.zeros(4)

        logits = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
        idx, w = extract_routing(logits, block, attrs, top_k=2)

        scores = logits.sigmoid()
        # highest sigmoid scores are experts 3 and 0
        assert set(idx[0].tolist()) == {0, 3}
        torch.testing.assert_close(w, scores.gather(1, idx))


# ---------------------------------------------------------------------------
# Tests: MoE detection and pruning utilities
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMoEDetection:
    def test_detect_moe_attrs(self):
        model = FakeMoEModel()
        attrs = detect_moe_attrs(model)
        # FakeMoEBlock is not in the registry, but auto-detection should
        # find it via the "gate" + "experts" heuristic
        assert attrs is not None
        assert attrs.router_attr == "gate"
        assert attrs.experts_attr == "experts"
        assert attrs.routing_mode == SOFTMAX

    def test_find_moe_layers(self):
        model = FakeMoEModel(num_layers=3)
        attrs = detect_moe_attrs(model)
        layers = find_moe_layers(model, attrs)
        assert len(layers) == 3

    def test_find_moe_layers_with_ignore(self):
        model = FakeMoEModel(num_layers=3)
        attrs = detect_moe_attrs(model)
        layers = find_moe_layers(model, attrs, ignore=["layers.1"])
        assert len(layers) == 2
        assert all("layers.1" not in name for name in layers)

    def test_get_num_experts(self):
        model = FakeMoEModel(num_experts=8)
        attrs = detect_moe_attrs(model)
        layers = find_moe_layers(model, attrs)
        for module in layers.values():
            assert get_num_experts(module, attrs) == 8

    def test_get_router_num_groups_default(self):
        model = FakeMoEModel()
        attrs = detect_moe_attrs(model)
        module = next(iter(find_moe_layers(model, attrs).values()))
        assert get_router_num_groups(module, attrs) == 1

    def test_unsupported_arch_raises(self):
        class MixtralSparseMoeBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(8, 4, bias=False)
                self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

        model = nn.Module()
        model.block = MixtralSparseMoeBlock()
        with pytest.raises(NotImplementedError, match="Mixtral"):
            detect_moe_attrs(model)


# ---------------------------------------------------------------------------
# Tests: pruning
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPruning:
    def test_compute_retained_global(self):
        saliency = torch.tensor(
            [0.1, 0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4], dtype=torch.float64
        )
        retained = compute_retained_experts(saliency, n_experts_to_drop=3, n_group=1)
        # drop the 3 lowest: experts 0 (0.1), 5 (0.2), 3 (0.3)
        assert retained == [1, 2, 4, 6, 7]

    def test_compute_retained_per_group(self):
        # 2 groups of 4; drop 2 -> 1 per group (lowest within each group)
        saliency = torch.tensor(
            [0.1, 0.9, 0.8, 0.7, 0.6, 0.2, 0.5, 0.4], dtype=torch.float64
        )
        retained = compute_retained_experts(saliency, n_experts_to_drop=2, n_group=2)
        # group 0 [0,1,2,3]: drop expert 0 (0.1) -> keep 1,2,3
        # group 1 [4,5,6,7]: drop expert 5 (0.2) -> keep 4,6,7
        assert retained == [1, 2, 3, 4, 6, 7]

    def test_compute_retained_too_many_raises(self):
        saliency = torch.arange(4, dtype=torch.float64)
        with pytest.raises(ValueError, match="Cannot drop"):
            compute_retained_experts(saliency, n_experts_to_drop=4, n_group=1)

    def test_routing_feasible_ok(self):
        # non-group: 8 experts, drop 4 -> 4 remain, top_k=2 fits
        assert_routing_feasible(
            num_experts=8, n_experts_to_drop=4, n_group=1, topk_group=1, top_k=2
        )
        # group: 16 experts, 4 groups, topk_group=4 -> 2/group*4 = 8 reachable
        assert_routing_feasible(
            num_experts=16, n_experts_to_drop=8, n_group=4, topk_group=4, top_k=8
        )

    def test_routing_feasible_too_aggressive_raises(self):
        # group: 16 experts, 4 groups, only topk_group=1 group reachable, 2/group
        # remain -> only 2 reachable, but top_k=3 requested
        with pytest.raises(ValueError, match="too aggressive"):
            assert_routing_feasible(
                num_experts=16, n_experts_to_drop=8, n_group=4, topk_group=1, top_k=3
            )

    def test_routing_feasible_non_group_too_aggressive_raises(self):
        # 8 experts, drop 7 -> 1 remains, top_k=2 cannot be satisfied
        with pytest.raises(ValueError, match="too aggressive"):
            assert_routing_feasible(
                num_experts=8, n_experts_to_drop=7, n_group=1, topk_group=1, top_k=2
            )

    def test_prune_moe_layer(self):
        model = FakeMoEModel(num_experts=8)
        attrs = detect_moe_attrs(model)
        layers = find_moe_layers(model, attrs)
        layer_name = next(iter(layers.keys()))
        moe_block = model.get_submodule(layer_name)
        original_experts = list(moe_block.experts)
        original_gate = moe_block.gate.weight.detach().clone()

        retained = [1, 2, 4, 6, 7]
        result = prune_moe_layer(model, layer_name, retained, attrs)

        assert result == retained
        assert len(moe_block.experts) == 5
        assert moe_block.gate.weight.shape[0] == 5
        assert moe_block.gate.out_features == 5
        assert moe_block.num_experts == 5
        assert all(
            expert is original_experts[original_idx]
            for expert, original_idx in zip(moe_block.experts, retained, strict=True)
        )
        torch.testing.assert_close(moe_block.gate.weight, original_gate[retained])

    def test_prune_resizes_correction_bias(self):
        # group-limited routers carry a per-expert score-correction bias buffer
        class GroupRouter(nn.Module):
            def __init__(self, num_experts, hidden):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(num_experts, hidden))
                self.register_buffer(
                    "e_score_correction_bias", torch.arange(num_experts).float()
                )
                self.n_group = 2

        class GroupMoE(nn.Module):
            def __init__(self, num_experts, hidden):
                super().__init__()
                self.gate = GroupRouter(num_experts, hidden)
                self.experts = nn.ModuleList(
                    [nn.Linear(hidden, hidden) for _ in range(num_experts)]
                )
                self.n_routed_experts = num_experts

        model = nn.Module()
        model.block = GroupMoE(8, 16)
        attrs = MoEModelAttrs("gate", "experts", "n_routed_experts", INDICES_WEIGHTS)

        retained = [1, 2, 3, 5, 6, 7]
        prune_moe_layer(model, "block", retained, attrs)

        block = model.block
        assert block.gate.weight.shape[0] == 6
        assert block.gate.e_score_correction_bias.shape[0] == 6
        torch.testing.assert_close(
            block.gate.e_score_correction_bias,
            torch.tensor(retained, dtype=torch.float32),
        )
        assert block.n_routed_experts == 6


# ---------------------------------------------------------------------------
# Tests: REAPPruningModifier
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_reap_is_registered():
    modifier = ModifierFactory.create(
        type_="REAPPruningModifier",
        allow_experimental=False,
        allow_registered=True,
        sparsity=0.5,
    )
    assert isinstance(modifier, REAPPruningModifier)


@pytest.mark.unit
def test_reap_validation():
    with pytest.raises(ValueError, match="sparsity"):
        REAPPruningModifier()

    with pytest.raises(ValueError, match="sparsity"):
        REAPPruningModifier(sparsity=0.0)

    with pytest.raises(ValueError, match="sparsity"):
        REAPPruningModifier(sparsity=1.0)

    with pytest.raises(ValueError, match="sparsity"):
        REAPPruningModifier(sparsity=-0.1)


def _make_state(model):
    return State(
        model=model,
        teacher_model=None,
        optimizer=None,
        optim_wrapped=False,
        loss=None,
        batch_data=None,
    )


@pytest.mark.unit
def test_reap_sequential_epoch_end_finalizes_decision():
    """Decisions are finalized (and buffers freed) at SEQUENTIAL_EPOCH_END."""
    torch.manual_seed(0)
    hidden_dim = 16
    model = FakeMoEModel(hidden_dim=hidden_dim, num_experts=8, top_k=2)
    model.eval()

    modifier = REAPPruningModifier(sparsity=0.5)
    state = _make_state(model)
    modifier.initialize(state)

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_EPOCH_START))
    assert modifier.started_
    assert len(modifier._saliency_trackers) == 2

    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(2, 8, hidden_dim))

    # before the epoch end, nothing is decided yet
    assert modifier._prune_decisions == {}

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))

    # both layers' decisions are finalized and their trackers/buffers freed
    assert len(modifier._prune_decisions) == 2
    assert all(len(r) == 4 for r in modifier._prune_decisions.values())
    assert modifier._saliency_trackers == {}
    assert modifier._norm_buffers == {}

    decisions_after_epoch = {k: list(v) for k, v in modifier._prune_decisions.items()}

    # the sequential pipeline re-runs a finalized subgraph (propagate_error)
    # with hooks still registered; this must not crash or change decisions
    with torch.no_grad():
        model(torch.randn(2, 8, hidden_dim))
    assert modifier._saliency_trackers == {}
    assert modifier._prune_decisions == decisions_after_epoch

    modifier.finalize(state)
    assert model.config.num_experts == 4


@pytest.mark.unit
def test_reap_full_lifecycle():
    """End-to-end test: initialize, calibrate, finalize, verify pruning.

    Uses no SEQUENTIAL_EPOCH_END, exercising the on_end fallback path."""
    torch.manual_seed(42)

    hidden_dim = 16
    num_experts = 8
    top_k = 2
    model = FakeMoEModel(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    model.eval()

    modifier = REAPPruningModifier(sparsity=0.5)
    state = _make_state(model)

    modifier.initialize(state)
    assert modifier.initialized_
    assert modifier._n_experts_to_drop == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_EPOCH_START))
    assert modifier.started_

    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(2, 8, hidden_dim)
            model(x)

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_EPOCH_END))
    assert modifier.ended_

    # decisions finalized via the on_end fallback
    assert len(modifier._prune_decisions) == 2

    modifier.finalize(state)
    assert modifier.finalized_

    # Verify pruning results
    assert model.config.num_experts == 4

    for layer in model.model.layers:
        assert len(layer.mlp.experts) == 4
        assert layer.mlp.gate.weight.shape[0] == 4
        assert layer.mlp.gate.out_features == 4

    # Verify model still runs
    with torch.no_grad():
        x = torch.randn(2, 4, hidden_dim)
        out = model(x)
        assert out.shape == (2, 4, hidden_dim)
        assert not torch.isnan(out).any()
