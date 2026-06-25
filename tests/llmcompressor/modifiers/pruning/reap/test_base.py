import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from llmcompressor.modifiers.pruning.reap.utils import (
    MoeModelAttrs,
    REAPSaliencyTracker,
    get_moe_attrs,
    prune_moe_layer,
    update_model_config,
)
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D, ExpertMLP

# ---------------------------------------------------------------------------
# Helpers: tiny synthetic MoE model with LinearExperts2D for testing
# ---------------------------------------------------------------------------


class FakeMoEConfig(PretrainedConfig):
    """Minimal PreTrainedConfig for testing LinearExperts2D."""

    model_type = "fake_moe"

    def __init__(
        self,
        num_experts: int = 8,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        num_hidden_layers: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Required for MoEConfig.from_config
        self.num_experts = num_experts
        self.num_experts_per_tok = 2
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"
        self.use_bias = False
        self.dtype = torch.float32
        self.num_hidden_layers = num_hidden_layers
        # Router config
        self.norm_topk_prob = True


class FakeLinearExperts2D(LinearExperts2D):
    """
    LinearExperts2D subclass with class variables set for testing.
    These match the standard MoE expert format (gate + up projections).
    """
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = True

    @staticmethod
    def _apply_gate(gate_up_out: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU gating: splits the concatenated tensor into gate and up parts,
        applies SiLU to the gate part, and multiplies with the up part.

        Args:
            gate_up_out: [batch, 2 * intermediate_size] concatenated tensor

        Returns:
            [batch, intermediate_size] gated output
        """
        gate, up = gate_up_out.chunk(2, dim=-1)  # Split into two halves
        return torch.nn.functional.silu(gate) * up  # SwiGLU: SiLU(gate) ⊙ up


class FakeRouter(nn.Module):
    """
    Top-K router for MoE layer.
    Based on standard MoE router implementations (e.g., Qwen3MoeTopKRouter).

    Uses nn.Parameter for weight to match the structure expected by prune_moe_layer.
    """

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.out_features = config.num_experts

        # Router weight: [num_experts, hidden_dim]
        # Using nn.Parameter to match expected structure (not nn.Linear)
        self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        """
        Route tokens to top-k experts.

        Args:
            hidden_states: [batch_size * seq_len, hidden_dim] or [seq_len, hidden_dim]

        Returns:
            router_logits: [seq_len, num_experts] - raw router scores
            router_weights: [seq_len, top_k] - normalized weights for selected experts
            selected_experts: [seq_len, top_k] - indices of selected experts
        """
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)

        # Compute router logits using F.linear (weight transposed for correct matmul)
        router_logits = F.linear(hidden_states, self.weight)  # [seq_len, num_experts]

        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # Select top-k experts
        router_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # Both: [seq_len, top_k]

        # Normalize top-k weights if configured
        if self.norm_topk_prob:
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)

        # Convert back to original dtype
        router_weights = router_weights.to(router_logits.dtype)

        return router_logits, router_weights, selected_experts


class FakeMoEBlock(nn.Module):
    """
    Single MoE block with router and experts.
    Based on standard MoE block implementations (e.g., Qwen3MoeSparseMoeBlock).
    """

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.gate = FakeRouter(config)
        self.experts = FakeLinearExperts2D(config)
        self.num_experts = config.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE block.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten to [batch_size * seq_len, hidden_dim]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Route tokens to experts
        router_logits, router_weights, selected_experts = self.gate(hidden_states_flat)

        # Process tokens through selected experts
        expert_output = self.experts(hidden_states_flat, selected_experts, router_weights)

        # Reshape back to [batch_size, seq_len, hidden_dim]
        output = expert_output.view(batch_size, seq_len, hidden_dim)

        return output


class FakeMoEModel(nn.Module):
    """
    Complete MoE model with multiple layers.
    Simplified model with only MoE blocks (no attention, embeddings, etc.).
    """

    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            FakeMoEBlock(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all MoE layers.

        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


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

        # expert_norms_dict: only experts that received tokens
        expert_norms_dict = {
            0: torch.tensor([3.0, 1.0]),  # expert 0 got 2 tokens
            1: torch.tensor([2.0]),       # expert 1 got 1 token
            2: torch.tensor([4.0]),       # expert 2 got 1 token
            # expert 3 got 0 tokens, not in dict
        }

        tracker.update(topk_indices, topk_weights, expert_norms_dict)

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
        tracker.update(idx, torch.tensor([[0.6]]), {0: torch.tensor([3.0])})
        tracker.update(idx, torch.tensor([[0.4]]), {0: torch.tensor([1.0])})
        # expert 0: (0.6*3.0 + 0.4*1.0) / 2 = 2.2 / 2 = 1.1
        assert tracker.mean_saliency[0].item() == pytest.approx(1.1)
        assert tracker.total_count == pytest.approx(2.0)

    def test_compute_retained_global(self):
        # 8 experts, drop 3 globally
        tracker = REAPSaliencyTracker(num_experts=8)
        tracker.sum_saliency = torch.tensor(
            [0.1, 0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4], dtype=torch.float64
        )
        tracker.count = torch.ones(8, dtype=torch.float64)

        config = FakeMoEConfig(num_experts=8)
        moe_attrs = get_moe_attrs(FakeMoEModel(config), ignore=[])

        retained = tracker.compute_retained_experts(
            n_experts_to_drop=3,
            n_experts_to_drop_per_group=None,
            moe_attrs=moe_attrs
        )
        # drop the 3 lowest: experts 0 (0.1), 5 (0.2), 3 (0.3)
        assert retained == [1, 2, 4, 6, 7]

    def test_compute_retained_per_group(self):
        # 8 experts, 2 groups of 4; drop 2 total -> 1 per group
        tracker = REAPSaliencyTracker(num_experts=8)
        tracker.sum_saliency = torch.tensor(
            [0.1, 0.9, 0.8, 0.7, 0.6, 0.2, 0.5, 0.4], dtype=torch.float64
        )
        tracker.count = torch.ones(8, dtype=torch.float64)

        # Create a mock moe_attrs with group info
        config = FakeMoEConfig(num_experts=8)
        model = FakeMoEModel(config)
        moe_attrs = get_moe_attrs(model, ignore=[])
        # Override to simulate group-limited routing
        moe_attrs.n_group = 2
        moe_attrs.group_size = 4

        retained = tracker.compute_retained_experts(
            n_experts_to_drop=2,
            n_experts_to_drop_per_group=1,
            moe_attrs=moe_attrs
        )
        # group 0 [0,1,2,3]: drop expert 0 (0.1) -> keep 1,2,3
        # group 1 [4,5,6,7]: drop expert 5 (0.2) -> keep 4,6,7
        assert retained == [1, 2, 3, 4, 6, 7]


# ---------------------------------------------------------------------------
# Tests: MoE detection and utilities
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMoEDetection:
    def test_get_moe_attrs(self):
        config = FakeMoEConfig(num_experts=8, num_hidden_layers=2)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])

        assert attrs is not None
        assert attrs.router_attr == "gate"
        assert attrs.experts_attr == "experts"
        assert attrs.num_experts == 8
        assert attrs.top_k == 2
        assert len(attrs.moe_layer_names) == 2
        assert attrs.n_group is None
        assert attrs.top_k_group is None
        assert attrs.group_size is None

    def test_get_moe_attrs_with_ignore(self):
        config = FakeMoEConfig(num_experts=8, num_hidden_layers=3)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=["layers.1"])

        assert len(attrs.moe_layer_names) == 2
        assert all("layers.1" not in name for name in attrs.moe_layer_names)

    def test_get_moe_attrs_num_experts(self):
        config = FakeMoEConfig(num_experts=16)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])

        assert attrs.num_experts == 16

    def test_get_moe_attrs_top_k(self):
        config = FakeMoEConfig(num_experts=8)
        config.num_experts_per_tok = 3
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])

        assert attrs.top_k == 3


# ---------------------------------------------------------------------------
# Tests: pruning
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPruning:
    def test_prune_moe_layer(self):
        config = FakeMoEConfig(num_experts=8, num_hidden_layers=2)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])

        layer_name = attrs.moe_layer_names[0]
        moe_block = model.get_submodule(layer_name)
        original_gate_weight = moe_block.gate.weight.detach().clone()

        # Store references to original expert modules
        original_experts = [
            expert for expert in moe_block.experts.children()
            if isinstance(expert, ExpertMLP)
        ]

        retained = [1, 2, 4, 6, 7]
        result = prune_moe_layer(model, layer_name, retained, attrs)

        assert result == retained
        assert moe_block.num_experts == 5
        assert moe_block.experts.num_experts == 5
        assert moe_block.gate.num_experts == 5
        assert moe_block.gate.weight.shape[0] == 5
        assert moe_block.gate.out_features == 5

        # Verify correct experts retained (by object identity)
        pruned_experts = [
            expert for expert in moe_block.experts.children()
            if isinstance(expert, ExpertMLP)
        ]
        assert len(pruned_experts) == 5
        for i, original_idx in enumerate(retained):
            assert pruned_experts[i] is original_experts[original_idx]

        # Verify router weights correctly sliced
        torch.testing.assert_close(moe_block.gate.weight, original_gate_weight[retained])

    def test_prune_resizes_correction_bias(self):
        # group-limited routers carry a per-expert score-correction bias buffer
        # Use the real model but add the correction bias to the router
        config = FakeMoEConfig(num_experts=8, num_hidden_layers=1, hidden_size=16)
        model = FakeMoEModel(config)

        # Add e_score_correction_bias to the router
        layer = model.layers[0]
        layer.gate.register_buffer(
            "e_score_correction_bias", torch.arange(8, dtype=torch.float32)
        )

        attrs = get_moe_attrs(model, ignore=[])

        retained = [1, 2, 3, 5, 6, 7]
        prune_moe_layer(model, attrs.moe_layer_names[0], retained, attrs)

        assert layer.gate.weight.shape[0] == 6
        assert layer.gate.num_experts == 6
        assert layer.gate.out_features == 6
        assert layer.gate.e_score_correction_bias.shape[0] == 6
        torch.testing.assert_close(
            layer.gate.e_score_correction_bias,
            torch.tensor(retained, dtype=torch.float32),
        )
        assert layer.num_experts == 6
        assert layer.experts.num_experts == 6

    def test_prune_multiple_layers_different_retained(self):
        config = FakeMoEConfig(num_experts=8, num_hidden_layers=2)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])

        # Prune first layer
        retained_1 = [0, 2, 3, 5, 7]
        prune_moe_layer(model, attrs.moe_layer_names[0], retained_1, attrs)

        # Prune second layer with different retained set
        retained_2 = [1, 3, 4, 6]
        prune_moe_layer(model, attrs.moe_layer_names[1], retained_2, attrs)

        layer_0 = model.get_submodule(attrs.moe_layer_names[0])
        layer_1 = model.get_submodule(attrs.moe_layer_names[1])

        assert layer_0.num_experts == len(retained_1)
        assert layer_1.num_experts == len(retained_2)

        # Verify model still runs
        with torch.no_grad():
            x = torch.randn(2, 4, config.hidden_size)
            out = model(x)
            assert out.shape == (2, 4, config.hidden_size)
            assert not torch.isnan(out).any()


@pytest.mark.unit
class TestUpdateModelConfig:
    def test_update_config_without_text_config(self):
        config = FakeMoEConfig(num_experts=8)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])

        update_model_config(model, attrs, new_num_experts=4)

        assert model.config.num_experts == 4

    def test_update_config_with_text_config(self):
        # Create a config with text_config wrapper
        class WrapperConfig(PretrainedConfig):
            def __init__(self, text_config):
                super().__init__()
                self.text_config = text_config

        text_config = FakeMoEConfig(num_experts=8)
        wrapper_config = WrapperConfig(text_config)

        model = FakeMoEModel(text_config)
        model.config = wrapper_config

        attrs = get_moe_attrs(model, ignore=[])
        assert attrs.has_text_config

        update_model_config(model, attrs, new_num_experts=4)

        assert model.config.text_config.num_experts == 4


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
def test_reap_initialization():
    config = FakeMoEConfig(num_experts=8, num_hidden_layers=2)
    model = FakeMoEModel(config)

    modifier = REAPPruningModifier(sparsity=0.5)
    state = _make_state(model)

    modifier.initialize(state)

    assert modifier.initialized_
    assert modifier._n_experts_to_drop == 4
    assert modifier._moe_attrs is not None
    assert len(modifier._moe_attrs.moe_layer_names) == 2


@pytest.mark.unit
def test_reap_initialization_too_aggressive():
    config = FakeMoEConfig(num_experts=8, top_k=5)
    config.num_experts_per_tok = 5
    model = FakeMoEModel(config)

    # Dropping 75% (6 experts) would leave only 2 available, but top_k=5
    modifier = REAPPruningModifier(sparsity=0.75)
    state = _make_state(model)

    with pytest.raises(ValueError, match="too aggressive"):
        modifier.initialize(state)


@pytest.mark.unit
def test_reap_initialization_zero_drop():
    # Very small sparsity that rounds to 0 experts to drop
    config = FakeMoEConfig(num_experts=100)
    model = FakeMoEModel(config)

    modifier = REAPPruningModifier(sparsity=0.001)  # Would drop 0.1 experts
    state = _make_state(model)

    with pytest.raises(ValueError, match="0 experts to drop"):
        modifier.initialize(state)


@pytest.mark.unit
def test_reap_full_lifecycle():
    """End-to-end test: initialize, calibrate, finalize, verify pruning.

    Uses SEQUENTIAL_EPOCH_END to trigger pruning decisions."""
    torch.manual_seed(42)

    config = FakeMoEConfig(
        num_experts=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = REAPPruningModifier(sparsity=0.5)
    state = _make_state(model)

    # Initialize
    modifier.initialize(state)
    assert modifier.initialized_
    assert modifier._n_experts_to_drop == 4

    # Start calibration
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert modifier.started_
    assert len(modifier._saliency_trackers) == 2

    # Run forward passes to collect saliency
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(2, 8, config.hidden_size)
            model(x)

    # Sequential epoch end triggers pruning decisions
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))

    # Verify decisions finalized and buffers freed
    assert len(modifier._saliency_trackers) == 0
    assert len(modifier._norm_buffers) == 0

    # End calibration
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    assert modifier.ended_

    # Finalize
    modifier.finalize(state)
    assert modifier.finalized_

    # Verify pruning results
    assert model.config.num_experts == 4

    for layer in model.layers:
        assert layer.num_experts == 4
        expert_count = sum(
            1 for m in layer.experts.children() if isinstance(m, ExpertMLP)
        )
        assert expert_count == 4
        assert layer.gate.weight.shape[0] == 4
        assert layer.gate.out_features == 4
        assert layer.gate.num_experts == 4

    # Verify model still runs
    with torch.no_grad():
        x = torch.randn(2, 4, config.hidden_size)
        out = model(x)
        assert out.shape == (2, 4, config.hidden_size)
        assert not torch.isnan(out).any()


@pytest.mark.unit
def test_reap_with_ignore():
    """Test that layers matching ignore patterns are skipped."""
    config = FakeMoEConfig(num_experts=8, num_hidden_layers=3)
    model = FakeMoEModel(config)

    modifier = REAPPruningModifier(sparsity=0.5, ignore=["layers.1"])
    state = _make_state(model)

    modifier.initialize(state)

    # Should only track 2 layers (0 and 2), not layer 1
    assert len(modifier._moe_attrs.moe_layer_names) == 2
    assert all("layers.1" not in name for name in modifier._moe_attrs.moe_layer_names)


@pytest.mark.unit
def test_reap_different_sparsity_levels():
    """Test various sparsity levels."""
    config = FakeMoEConfig(num_experts=8)

    # 25% sparsity
    model = FakeMoEModel(config)
    modifier = REAPPruningModifier(sparsity=0.25)
    state = _make_state(model)
    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 2

    # 50% sparsity
    model = FakeMoEModel(config)
    modifier = REAPPruningModifier(sparsity=0.5)
    state = _make_state(model)
    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    # 75% sparsity
    model = FakeMoEModel(config)
    modifier = REAPPruningModifier(sparsity=0.75)
    state = _make_state(model)
    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 6


@pytest.mark.unit
def test_reap_forward_pass_after_pruning():
    """Verify that forward passes work correctly after pruning and don't crash."""
    torch.manual_seed(0)

    config = FakeMoEConfig(
        num_experts=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = REAPPruningModifier(sparsity=0.5)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    # Calibration passes
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(2, 8, config.hidden_size)
            model(x)

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # Multiple forward passes after pruning should all work
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(2, 4, config.hidden_size)
            out = model(x)
            assert out.shape == (2, 4, config.hidden_size)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()
