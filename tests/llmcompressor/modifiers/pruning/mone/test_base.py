from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.integrations.finegrained_fp8 import FP8Experts

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling.moe.linear_experts import (
    LinearExperts2D,
    NoviceExpertMLP,
)
from llmcompressor.modeling.moe.minimax_mone import MINIMAX_M2_STOCK_FP8_TRITON_PATCH
from llmcompressor.modeling.moe.mone import apply_mone_structure
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.mone import MoNEPruningModifier


class FakeMoEConfig(PretrainedConfig):
    model_type = "fake_moe"

    def __init__(
        self,
        num_experts: int = 4,
        hidden_size: int = 8,
        intermediate_size: int = 16,
        num_hidden_layers: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = 2
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = "silu"
        self.use_bias = False
        self.dtype = torch.float32
        self.num_hidden_layers = num_hidden_layers
        self.norm_topk_prob = True


class FakeNativeFP8MoEConfig(FakeMoEConfig):
    model_type = "minimax_m2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_local_experts = self.num_experts
        self.torch_dtype = torch.float32


class FakeLinearExperts2D(LinearExperts2D):
    is_concatenated = False
    is_transposed = False
    has_bias = False
    has_gate = True

    @staticmethod
    def _apply_gate(gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


class FakeRouter(nn.Module):
    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.out_features = config.num_experts
        self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        router_weights, selected_experts = torch.topk(
            router_probs,
            self.top_k,
            dim=-1,
        )
        if self.norm_topk_prob:
            router_weights = router_weights / router_weights.sum(
                dim=-1,
                keepdim=True,
            )
        return router_logits, router_weights.to(router_logits.dtype), selected_experts


class FakeMoEBlock(nn.Module):
    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.gate = FakeRouter(config)
        self.experts = FakeLinearExperts2D(config)
        self.num_experts = config.num_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        _, router_weights, selected_experts = self.gate(hidden_states_flat)
        output = self.experts(
            hidden_states_flat,
            selected_experts,
            router_weights,
        )
        return output.view(batch_size, seq_len, hidden_dim)


class FakeNativeFP8MoEBlock(nn.Module):
    def __init__(self, config: FakeNativeFP8MoEConfig):
        super().__init__()
        self.gate = FakeRouter(config)
        self.experts = FP8Experts(config, block_size=(128, 128), has_gate=True)
        self._replace_fp8_test_weights(config)

    @torch.no_grad()
    def _replace_fp8_test_weights(self, config: FakeNativeFP8MoEConfig):
        self.experts.gate_up_proj = nn.Parameter(
            torch.randn(
                config.num_experts,
                2 * config.intermediate_size,
                config.hidden_size,
            )
        )
        self.experts.down_proj = nn.Parameter(
            torch.randn(
                config.num_experts,
                config.hidden_size,
                config.intermediate_size,
            )
        )
        self.experts.gate_up_proj_scale_inv = nn.Parameter(
            torch.ones(config.num_experts, 1, 1),
            requires_grad=False,
        )
        self.experts.down_proj_scale_inv = nn.Parameter(
            torch.ones(config.num_experts, 1, 1),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        _, router_weights, selected_experts = self.gate(hidden_states_flat)
        output = self.experts(
            hidden_states_flat,
            selected_experts,
            router_weights,
        )
        return output.view(batch_size, seq_len, hidden_dim)


class FakeMoEModel(nn.Module):
    def __init__(self, config: FakeMoEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FakeMoEBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class FakeNativeFP8MoEModel(nn.Module):
    def __init__(self, config: FakeNativeFP8MoEConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [FakeNativeFP8MoEBlock(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


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
@pytest.mark.usefixtures("setup_modifier_factory")
def test_mone_is_registered():
    modifier = ModifierFactory.create(
        type_="MoNEPruningModifier",
        allow_experimental=False,
        allow_registered=True,
        preserve_n_experts=2,
    )
    assert isinstance(modifier, MoNEPruningModifier)


@pytest.mark.unit
def test_mone_validation():
    with pytest.raises(ValueError, match="Exactly one"):
        MoNEPruningModifier()

    with pytest.raises(ValueError, match="Exactly one"):
        MoNEPruningModifier(preserve_n_experts=2, sparsity=0.5)

    with pytest.raises(ValueError, match="sparsity"):
        MoNEPruningModifier(sparsity=0.0)


@pytest.mark.unit
def test_mone_full_lifecycle_replaces_novices():
    torch.manual_seed(7)
    config = FakeMoEConfig(num_experts=4, num_hidden_layers=2)
    model = FakeMoEModel(config).eval()

    original_router_shapes = [layer.gate.weight.shape for layer in model.layers]
    modifier = MoNEPruningModifier(preserve_n_experts=2)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(2, 5, config.hidden_size))

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    assert modifier.finalized_
    assert set(model.config.approximate_experts) == {"0", "1"}

    for layer_idx, layer in enumerate(model.layers):
        assert layer.gate.weight.shape == original_router_shapes[layer_idx]
        assert layer.gate.num_experts == config.num_experts
        assert layer.experts.num_experts == config.num_experts

        novice_indices = model.config.approximate_experts[str(layer_idx)]
        assert len(novice_indices) == 2
        assert all(
            isinstance(layer.experts[idx], NoviceExpertMLP) for idx in novice_indices
        )

    with torch.no_grad():
        out = model(torch.randn(2, 3, config.hidden_size))
    assert out.shape == (2, 3, config.hidden_size)
    assert not torch.isnan(out).any()


@pytest.mark.unit
def test_mone_native_fp8_lifecycle_attaches_constants():
    torch.manual_seed(17)
    config = FakeNativeFP8MoEConfig(num_experts=4, num_hidden_layers=1)
    model = FakeNativeFP8MoEModel(config).eval()
    modifier = MoNEPruningModifier(preserve_n_experts=2)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(2, 5, config.hidden_size))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    assert set(model.config.approximate_experts) == {"0"}
    novice_indices = model.config.approximate_experts["0"]
    assert len(novice_indices) == 2

    experts = model.layers[0].experts
    assert experts.constant_expert_ids == tuple(novice_indices)
    assert experts.constant_expert_values.shape == (
        len(novice_indices),
        config.hidden_size,
    )
    assert experts.layer_idx == 0
    assert getattr(model, "_llmcompressor_minimax_mone_layout")
    assert model.config.llmcompressor_mone_implementation == {
        "algorithm": "mone",
        "patches_enabled": [MINIMAX_M2_STOCK_FP8_TRITON_PATCH],
    }

    with torch.no_grad():
        out = model(torch.randn(2, 3, config.hidden_size))
    assert out.shape == (2, 3, config.hidden_size)
    assert not torch.isnan(out).any()

@pytest.mark.unit
def test_mone_zero_out_novices():
    torch.manual_seed(3)
    config = FakeMoEConfig(num_experts=4, num_hidden_layers=1)
    model = FakeMoEModel(config).eval()
    modifier = MoNEPruningModifier(
        preserve_n_experts=2,
        zero_out_novice=True,
    )
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    with torch.no_grad():
        model(torch.randn(2, 5, config.hidden_size))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))

    novice_indices = model.config.approximate_experts["0"]
    for idx in novice_indices:
        torch.testing.assert_close(
            model.layers[0].experts[idx].approx_value,
            torch.zeros_like(model.layers[0].experts[idx].approx_value),
        )


@pytest.mark.unit
def test_mone_structure_rewriter_enables_strict_reload():
    torch.manual_seed(11)
    config = FakeMoEConfig(num_experts=4, num_hidden_layers=2)
    model = FakeMoEModel(config).eval()
    modifier = MoNEPruningModifier(preserve_n_experts=2)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(2, 5, config.hidden_size))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))

    state_dict = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }

    plain_model = FakeMoEModel(deepcopy(model.config))
    with pytest.raises(RuntimeError):
        plain_model.load_state_dict(state_dict, strict=True)

    reloaded = FakeMoEModel(deepcopy(model.config)).eval()
    replaced = apply_mone_structure(reloaded)

    assert set(replaced) == {"layers.0", "layers.1"}
    for layer_idx, novice_indices in model.config.approximate_experts.items():
        assert replaced[f"layers.{layer_idx}"] == novice_indices

    incompatible = reloaded.load_state_dict(state_dict, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    x = torch.randn(2, 3, config.hidden_size)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x))


@pytest.mark.unit
def test_mone_requires_routed_expert_calibration():
    import llmcompressor.modeling.moe.context as context

    original_state = context._CALIBRATE_ALL_EXPERTS
    context._CALIBRATE_ALL_EXPERTS = True
    try:
        model = FakeMoEModel(FakeMoEConfig())
        modifier = MoNEPruningModifier(preserve_n_experts=2)
        state = _make_state(model)
        modifier.initialize(state)

        with pytest.raises(ValueError, match="moe_calibrate_all_experts=False"):
            modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    finally:
        context._CALIBRATE_ALL_EXPERTS = original_state
