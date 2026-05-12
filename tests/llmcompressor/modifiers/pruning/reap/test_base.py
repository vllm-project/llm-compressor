import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.modifiers.pruning.reap.utils import (
    REAPSaliencyTracker,
    detect_moe_attrs,
    find_moe_layers,
    get_num_experts,
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
    def test_update_single_expert(self):
        tracker = REAPSaliencyTracker(num_experts=4)
        gate_vals = torch.tensor([0.5, 0.3])
        norms = torch.tensor([2.0, 4.0])

        tracker.update(1, gate_vals, norms)

        expected = 0.5 * 2.0 + 0.3 * 4.0
        assert tracker.sum_saliency[1].item() == pytest.approx(expected)
        assert tracker.count[1].item() == 2
        assert tracker.count[0].item() == 0

    def test_mean_saliency(self):
        tracker = REAPSaliencyTracker(num_experts=3)

        tracker.update(0, torch.tensor([0.6]), torch.tensor([3.0]))
        tracker.update(0, torch.tensor([0.4]), torch.tensor([1.0]))
        tracker.update(1, torch.tensor([0.5, 0.5]), torch.tensor([2.0, 2.0]))

        mean = tracker.mean_saliency
        # Expert 0: (0.6*3.0 + 0.4*1.0) / 2 = 2.2 / 2 = 1.1
        assert mean[0].item() == pytest.approx(1.1)
        # Expert 1: (0.5*2.0 + 0.5*2.0) / 2 = 2.0 / 2 = 1.0
        assert mean[1].item() == pytest.approx(1.0)
        # Expert 2: never routed -> 0
        assert mean[2].item() == pytest.approx(0.0)


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


@pytest.mark.unit
class TestPruning:
    def test_prune_moe_layer(self):
        model = FakeMoEModel(num_experts=8)
        attrs = detect_moe_attrs(model)
        layers = find_moe_layers(model, attrs)
        layer_name = list(layers.keys())[0]
        moe_block = model.get_submodule(layer_name)
        original_experts = list(moe_block.experts)
        original_gate = moe_block.gate.weight.detach().clone()

        saliency = torch.tensor(
            [0.1, 0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4], dtype=torch.float64
        )
        retained = prune_moe_layer(model, layer_name, saliency, 3, attrs)

        assert retained == [1, 2, 4, 6, 7]
        assert len(moe_block.experts) == 5
        assert moe_block.gate.weight.shape[0] == 5
        assert moe_block.gate.out_features == 5
        assert all(
            expert is original_experts[original_idx]
            for expert, original_idx in zip(moe_block.experts, retained)
        )
        torch.testing.assert_close(moe_block.gate.weight, original_gate[retained])

    def test_prune_too_many_raises(self):
        model = FakeMoEModel(num_experts=4)
        attrs = detect_moe_attrs(model)
        layers = find_moe_layers(model, attrs)
        layer_name = list(layers.keys())[0]

        saliency = torch.arange(4, dtype=torch.float64)
        with pytest.raises(ValueError, match="Cannot drop"):
            prune_moe_layer(model, layer_name, saliency, 4, attrs)


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


@pytest.mark.unit
def test_reap_full_lifecycle():
    """End-to-end test: initialize, calibrate, finalize, verify pruning."""
    torch.manual_seed(42)

    hidden_dim = 16
    num_experts = 8
    top_k = 2
    model = FakeMoEModel(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)
    model.eval()

    modifier = REAPPruningModifier(sparsity=0.5)

    # Create a minimal State
    state = State(
        model=model,
        teacher_model=None,
        optimizer=None,
        optim_wrapped=False,
        loss=None,
        batch_data=None,
    )

    # Initialize
    modifier.initialize(state)
    assert modifier.initialized_
    assert modifier._n_experts_to_drop == 4

    # Simulate calibration: manually trigger start, run data, trigger end
    start_event = Event(type_=EventType.CALIBRATION_EPOCH_START)
    modifier.update_event(state, start_event)
    assert modifier.started_

    # Run calibration data (forward passes through model trigger hooks)
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(2, 8, hidden_dim)
            model(x)

    end_event = Event(type_=EventType.CALIBRATION_EPOCH_END)
    modifier.update_event(state, end_event)
    assert modifier.ended_

    # Verify saliency was accumulated
    for tracker in modifier._saliency_trackers.values():
        assert tracker.count.sum().item() > 0

    # Finalize (triggers pruning)
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
