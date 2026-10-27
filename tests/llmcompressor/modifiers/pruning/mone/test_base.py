import glob
import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch import distributed as dist
from transformers.configuration_utils import PretrainedConfig

from llmcompressor.core import Event, EventType, State
from llmcompressor.modeling.moe.linear_experts import ExpertMLP, LinearExperts2D
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.mone import MoNePruningModifier
from llmcompressor.modifiers.pruning.mone.utils import (
    MoNeSaliencyTracker,
    build_moe_num_experts,
    collapse_moe_layer,
)
from llmcompressor.modifiers.pruning.reap.utils import get_moe_attrs
from tests.testing_utils import requires_gpu, torchrun

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
        router_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(router_logits.dtype)
        return router_logits, router_weights, selected_experts


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
        expert_output = self.experts(
            hidden_states_flat, selected_experts, router_weights
        )
        return expert_output.view(batch_size, seq_len, hidden_dim)


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


def _make_state(model):
    return State(
        model=model,
        teacher_model=None,
        optimizer=None,
        optim_wrapped=False,
        loss=None,
        batch_data=None,
    )


# ---------------------------------------------------------------------------
# Tests: MoNeSaliencyTracker
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMoNeSaliencyTracker:
    def test_fusion_saliency_metric(self):
        """Saliency is ``||std(f_j)||_2 * E_tokens[g_j * 1(j in top-k)]``."""
        import llmcompressor.modeling.moe.context as context

        original_state = context._CALIBRATE_ALL_EXPERTS
        context._CALIBRATE_ALL_EXPERTS = False
        try:
            tracker = MoNeSaliencyTracker(
                num_experts=4, hidden_dim=2, device=torch.device("cpu")
            )

            # routing over two tokens: expert 0 and 1 are always selected
            tracker.update_routing(torch.tensor([[0, 1]]), torch.tensor([[0.6, 0.4]]))
            tracker.update_routing(torch.tensor([[0, 1]]), torch.tensor([[0.8, 0.2]]))

            # expert outputs over the tokens routed to each expert
            tracker.update_outputs(
                torch.tensor([[0, 1]]),
                {0: torch.tensor([[1.0, 2.0]]), 1: torch.tensor([[10.0, 10.0]])},
            )
            tracker.update_outputs(
                torch.tensor([[0, 1]]),
                {0: torch.tensor([[3.0, 4.0]]), 1: torch.tensor([[12.0, 14.0]])},
            )

            # routing_score = sum_gate / token_count (token_count == 2)
            routing = tracker.routing_score
            torch.testing.assert_close(
                routing,
                torch.tensor([0.7, 0.3, 0.0, 0.0], dtype=torch.float32),
            )

            # fluctuation = L2 norm over dims of per-dim std
            # expert 0: std=[1,1] -> sqrt(2); expert 1: std=[1,2] -> sqrt(5)
            fluc = tracker.output_fluctuation
            assert fluc[0].item() == pytest.approx(2**0.5)
            assert fluc[1].item() == pytest.approx(5**0.5)
            assert fluc[2].item() == pytest.approx(0.0)
            assert fluc[3].item() == pytest.approx(0.0)

            saliency = tracker.saliency
            assert saliency[0].item() == pytest.approx(2**0.5 * 0.7)
            assert saliency[1].item() == pytest.approx(5**0.5 * 0.3)
            assert saliency[2].item() == pytest.approx(0.0)
            assert saliency[3].item() == pytest.approx(0.0)

            # lowest-saliency experts (2, 3) are dropped
            retained = tracker.compute_retained_experts(n_experts_to_drop=2)
            assert sorted(retained.tolist()) == [0, 1]
        finally:
            context._CALIBRATE_ALL_EXPERTS = original_state

    def test_update_outputs_and_compute_novices(self):
        """Novices are the mean output over tokens routed to each expert."""
        import llmcompressor.modeling.moe.context as context

        original_state = context._CALIBRATE_ALL_EXPERTS
        context._CALIBRATE_ALL_EXPERTS = False
        try:
            tracker = MoNeSaliencyTracker(
                num_experts=4, hidden_dim=2, device=torch.device("cpu")
            )
            # expert 1 gets two tokens across two batches, expert 3 one token
            topk_indices = torch.tensor([[1, 3]])
            tracker.update_outputs(
                topk_indices,
                {1: torch.tensor([[1.0, 2.0]]), 3: torch.tensor([[7.0, 8.0]])},
            )
            tracker.update_outputs(
                torch.tensor([[1, 0]]),
                {1: torch.tensor([[3.0, 4.0]]), 0: torch.tensor([[0.0, 0.0]])},
            )
            novices = tracker.compute_novices(dropped=[1, 3])
            assert novices.shape == (2, 2)
            # expert 1 mean: ([1,2] + [3,4]) / 2 = [2, 3]
            torch.testing.assert_close(
                novices[0], torch.tensor([2.0, 3.0], dtype=torch.float32)
            )
            # expert 3 mean: [7, 8] / 1
            torch.testing.assert_close(
                novices[1], torch.tensor([7.0, 8.0], dtype=torch.float32)
            )
        finally:
            context._CALIBRATE_ALL_EXPERTS = original_state

    def test_compute_novices_ordering_matches_dropped(self):
        tracker = MoNeSaliencyTracker(
            num_experts=4, hidden_dim=1, device=torch.device("cpu")
        )
        tracker.sum_output = torch.tensor(
            [[0.0], [10.0], [20.0], [30.0]], dtype=torch.float32
        )
        tracker.output_count = torch.ones(4, dtype=torch.float32)
        novices = tracker.compute_novices(dropped=[3, 0])
        torch.testing.assert_close(
            novices[:, 0], torch.tensor([30.0, 0.0], dtype=torch.float32)
        )


# ---------------------------------------------------------------------------
# Tests: LinearExperts2D novice forward
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoviceForward:
    def test_forward_without_novices_matches_baseline(self):
        config = FakeMoEConfig(num_experts=4, hidden_size=8, intermediate_size=16)
        experts = FakeLinearExperts2D(config)
        assert experts.novices is None

        hidden = torch.randn(5, 8)
        top_k_index = torch.tensor([[0, 2], [1, 3], [2, 3], [0, 1], [3, 2]])
        top_k_weights = torch.rand(5, 2)

        out = experts(hidden, top_k_index, top_k_weights)
        assert out.shape == (5, 8)
        assert not torch.isnan(out).any()

    def test_forward_with_novices_matches_manual(self):
        torch.manual_seed(3)
        config = FakeMoEConfig(num_experts=4, hidden_size=8, intermediate_size=16)
        experts = FakeLinearExperts2D(config)

        # keep 2 experts, add 2 novices
        experts._modules = {
            "0": experts[0],
            "1": experts[1],
            "act_fn": experts.act_fn,
        }
        experts.num_experts = 2
        novices = torch.randn(2, 8)
        experts.register_parameter(
            "novices", nn.Parameter(novices, requires_grad=False)
        )

        hidden = torch.randn(5, 8)
        top_k_index = torch.tensor([[0, 2], [1, 3], [2, 3], [0, 1], [3, 2]])
        top_k_weights = torch.rand(5, 2)

        out = experts(hidden, top_k_index, top_k_weights)

        ref = torch.zeros_like(hidden)
        for t in range(5):
            for slot in range(2):
                e = top_k_index[t, slot].item()
                w = top_k_weights[t, slot].item()
                if e < 2:
                    ref[t] += w * experts[e](hidden[t : t + 1]).squeeze(0)
                else:
                    ref[t] += w * novices[e - 2]
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: collapse_moe_layer
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollapse:
    def test_collapse_keeps_router_width_and_sets_novices(self):
        config = FakeMoEConfig(num_experts=8, num_hidden_layers=1, hidden_size=16)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])

        layer_name = attrs.moe_layer_names[0]
        block = model.get_submodule(layer_name)
        original_experts = [
            e for e in block.experts.children() if isinstance(e, ExpertMLP)
        ]
        original_router = block.gate.weight.detach().clone()

        retained = [1, 2, 4, 6, 7]
        dropped = [0, 3, 5]
        novices = torch.arange(len(dropped) * 16, dtype=torch.float32).reshape(
            len(dropped), 16
        )

        collapse_moe_layer(model, layer_name, retained, novices, attrs)

        # retained experts left-shifted (object identity preserved)
        pruned = [e for e in block.experts.children() if isinstance(e, ExpertMLP)]
        assert len(pruned) == 5
        for i, orig_idx in enumerate(retained):
            assert pruned[i] is original_experts[orig_idx]
        assert block.experts.num_experts == 5

        # router keeps full R + N width, reordered to retained + dropped
        assert block.gate.weight.shape[0] == 8
        assert block.gate.out_features == 8
        torch.testing.assert_close(
            block.gate.weight, original_router[retained + dropped]
        )

        # novices installed, matching dropped order
        assert block.experts.novices is not None
        assert block.experts.novices.shape == (3, 16)
        torch.testing.assert_close(block.experts.novices, novices)

    def test_collapse_forward_preserves_retained_experts(self):
        """A collapsed model routed only to retained experts matches the original."""
        torch.manual_seed(7)
        config = FakeMoEConfig(num_experts=6, num_hidden_layers=1, hidden_size=8)
        model = FakeMoEModel(config)
        attrs = get_moe_attrs(model, ignore=[])
        layer_name = attrs.moe_layer_names[0]
        block = model.get_submodule(layer_name)

        hidden = torch.randn(4, 8)
        retained = [0, 1, 2, 3]
        # force routing to retained experts only
        top_k_index = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
        top_k_weights = torch.rand(4, 2)
        baseline = block.experts(hidden, top_k_index, top_k_weights)

        novices = torch.randn(2, 8)
        collapse_moe_layer(model, layer_name, retained, novices, attrs)

        # after collapse, retained experts occupy rows 0..3 unchanged
        out = block.experts(hidden, top_k_index, top_k_weights)
        torch.testing.assert_close(out, baseline, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Tests: modifier registration / validation / lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
def test_mone_is_registered():
    modifier = ModifierFactory.create(
        type_="MoNePruningModifier",
        allow_experimental=False,
        allow_registered=True,
        sparsity=0.5,
    )
    assert isinstance(modifier, MoNePruningModifier)


@pytest.mark.unit
def test_mone_validation():
    with pytest.raises(ValueError, match="sparsity"):
        MoNePruningModifier(sparsity=1.0)

    with pytest.raises(ValueError, match="sparsity"):
        MoNePruningModifier(sparsity=-0.1)

    # report_path must end with .json
    with pytest.raises(ValueError, match=r"\.json"):
        MoNePruningModifier(sparsity=0.5, report_path="report.pkl")

    # sparsity=0.0 (report-only) requires a report_path
    with pytest.raises(ValueError, match="report_path"):
        MoNePruningModifier(sparsity=0.0)

    # sparsity=0.0 with a report_path is valid
    MoNePruningModifier(sparsity=0.0, report_path="report.json")


@pytest.mark.unit
def test_mone_full_lifecycle():
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = MoNePruningModifier(sparsity=0.5)
    state = _make_state(model)

    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert len(modifier._saliency_trackers) == 2

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._saliency_trackers) == 0

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # router keeps full width; config num_experts unchanged (unlike REAP)
    assert model.config.num_experts == 8

    for layer in model.layers:
        assert layer.experts.num_experts == 4
        expert_count = sum(
            1 for m in layer.experts.children() if isinstance(m, ExpertMLP)
        )
        assert expert_count == 4
        assert layer.gate.weight.shape[0] == 8
        assert layer.experts.novices is not None
        assert layer.experts.novices.shape == (4, config.hidden_size)

    # moe_sparsity_config recorded for serialization
    assert hasattr(model, "_moe_sparsity_config")
    assert model._moe_sparsity_config.method.value == "novice"
    assert model._moe_sparsity_config.num_experts == [4, 4]

    with torch.no_grad():
        out = model(torch.randn(2, 4, config.hidden_size))
        assert out.shape == (2, 4, config.hidden_size)
        assert not torch.isnan(out).any()


@pytest.mark.unit
def test_mone_group_limited_router_not_implemented():
    """Group-limited routers are explicitly unsupported."""
    config = FakeMoEConfig(num_experts=8, num_hidden_layers=1)
    config.n_group = 2
    config.topk_group = 1
    model = FakeMoEModel(config)

    modifier = MoNePruningModifier(sparsity=0.5)
    state = _make_state(model)

    with pytest.raises(NotImplementedError, match="group-limited"):
        modifier.initialize(state)


@pytest.mark.unit
def test_mone_report_only_does_not_modify_model(tmp_path):
    """With sparsity=0.0, saliency is reported but weights/config are untouched."""
    torch.manual_seed(42)

    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    report_path = str(tmp_path / "report.json")
    modifier = MoNePruningModifier(sparsity=0.0, report_path=report_path)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(2, 8, config.hidden_size)
            model(x)

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # Model config and weights are unchanged; no sparsity config is recorded
    assert model.config.num_experts == 8
    assert not hasattr(model, "_moe_sparsity_config")
    for layer in model.layers:
        expert_count = sum(
            1 for m in layer.experts.children() if isinstance(m, ExpertMLP)
        )
        assert expert_count == 8
        assert layer.gate.weight.shape[0] == 8
        assert layer.experts.novices is None

    # Report holds per-expert statistics for each MoE layer, ordered by layer
    with open(report_path) as f:
        report = json.load(f)

    assert sorted(report) == ["count", "saliency"]
    for values in report.values():
        assert len(values) == 2

    for layer_counts, layer_saliency in zip(report["count"], report["saliency"]):
        assert len(layer_counts) == 8
        assert len(layer_saliency) == 8
        assert all(isinstance(value, float) for value in layer_saliency)
        # every expert was routed to at least once across the 5 batches
        assert all(count > 0 for count in layer_counts)


@pytest.mark.unit
def test_mone_layer_without_data_collapses_after_later_epoch():
    """A layer which saw no calibration tokens keeps its tracker (in case data
    arrives in a later epoch) and is collapsed once it does."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    modifier = MoNePruningModifier(sparsity=0.5)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    # first epoch end: no layer has seen any data yet, so nothing is
    # finalized and the trackers are kept
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._saliency_trackers) == 2
    for tracker in modifier._saliency_trackers.values():
        assert tracker.total_count == 0
    for layer in model.layers:
        assert layer.experts.num_experts == 8

    # second epoch: data arrives for every layer
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._saliency_trackers) == 0

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    for layer in model.layers:
        assert layer.experts.num_experts == 4
        assert layer.experts.novices is not None
        assert layer.experts.novices.shape == (4, config.hidden_size)


@pytest.mark.unit
def test_build_moe_num_experts_marks_dense_layers():
    config = FakeMoEConfig(num_experts=8, num_hidden_layers=4)
    model = FakeMoEModel(config)
    attrs = get_moe_attrs(model, ignore=[])
    # only layers 1 and 3 collapsed
    retained = {
        attrs.moe_layer_names[1]: [0, 1, 2, 3],
        attrs.moe_layer_names[3]: [0, 1, 2, 3, 4],
    }
    per_layer = build_moe_num_experts(model, attrs, retained)
    assert per_layer == [None, 4, None, 5]


# ---------------------------------------------------------------------------
# Tests: serialization (save-only; checkpoint layout + config)
# ---------------------------------------------------------------------------


def _tiny_qwen3_vl_moe():
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration

    from llmcompressor.utils.dev import skip_weights_initialize

    config = Qwen3VLMoeConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 256,
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
        },
    )
    with skip_weights_initialize():
        model = Qwen3VLMoeForConditionalGeneration(config)
    return model


@pytest.mark.unit
def test_linearized_state_dict_contains_novices():
    from llmcompressor.modeling.moe.linearize import linearize_moe

    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    attrs = get_moe_attrs(model, ignore=[])
    layer_name = attrs.moe_layer_names[0]
    collapse_moe_layer(model, layer_name, [0, 2], torch.randn(2, 64), attrs)

    experts = getattr(model.get_submodule(layer_name), attrs.experts_attr)
    keys = set(experts.state_dict())
    assert "novices" in keys
    assert experts.state_dict()["novices"].shape == (2, 64)
    # retained experts left-shifted to 0..R-1
    assert any(k.startswith("0.") for k in keys)
    assert not any(k.startswith("2.") for k in keys)


@pytest.mark.unit
def test_packed_repack_contains_novices():
    from llmcompressor.modeling.moe.linearize import linearize_moe, repack_moe

    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    attrs = get_moe_attrs(model, ignore=[])
    layer_name = attrs.moe_layer_names[0]
    collapse_moe_layer(model, layer_name, [0, 2], torch.randn(2, 64), attrs)

    block = model.get_submodule(layer_name)
    router = getattr(block, attrs.router_attr)
    assert router.weight.shape[0] == 4  # R + N preserved

    repack_moe(model)
    experts = getattr(model.get_submodule(layer_name), attrs.experts_attr)
    # packed retained experts (R), novices (N)
    assert experts.gate_up_proj.shape[0] == 2
    assert experts.down_proj.shape[0] == 2
    assert experts.novices.shape == (2, 64)


@pytest.mark.unit
def test_save_pretrained_writes_moe_sparsity_config_and_novices(tmp_path):
    from compressed_tensors.config import ExpertSparsityConfig, MoESparsityMethod

    from llmcompressor.modeling.moe.linearize import linearize_moe, repack_moe
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        modify_save_pretrained,
    )

    model = _tiny_qwen3_vl_moe()
    linearize_moe(model)
    attrs = get_moe_attrs(model, ignore=[])
    layer_name = attrs.moe_layer_names[0]
    collapse_moe_layer(model, layer_name, [0, 2], torch.randn(2, 64), attrs)

    model._moe_sparsity_config = ExpertSparsityConfig(
        method=MoESparsityMethod.NOVICE,
        num_experts=build_moe_num_experts(model, attrs, {layer_name: [0, 2]}),
    )
    repack_moe(model)

    modify_save_pretrained(model)
    out_dir = tmp_path / "mone_model"
    model.save_pretrained(out_dir, safe_serialization=True)

    # config.json: quantization_config.moe_sparsity_config
    with open(out_dir / "config.json") as f:
        config_data = json.load(f)
    msc = config_data["quantization_config"]["moe_sparsity_config"]
    assert msc["method"] == "novice"
    assert msc["num_experts"] == [2]

    # checkpoint: experts.novices tensor present, retained experts packed
    expert_keys = []
    for path in glob.glob(os.path.join(out_dir, "*.safetensors")):
        with safe_open(path, framework="pt") as handle:
            expert_keys.extend(k for k in handle.keys() if "mlp.experts" in k)
    assert any(k.endswith("experts.novices") for k in expert_keys)
    assert any(k.endswith("experts.gate_up_proj") for k in expert_keys)
    assert not any(".experts.0." in k for k in expert_keys)


# ---------------------------------------------------------------------------
# Tests: distributed (gloo) calibration with unevenly distributed data
# ---------------------------------------------------------------------------


@pytest.mark.unit
@torchrun(world_size=2, init_dist=False)
def test_mone_distributed_uneven_data():
    """Layers must be reduced and collapsed identically on all ranks when
    calibration data is unevenly distributed across ranks.

    Every rank reduces every tracked layer (contributing zeros where it saw
    no tokens) and the decision of which layers collected samples is made
    from the token count summed across ranks. This covers both the case
    where only the non-source rank saw data (deciding from the source rank's
    local count alone would skip the layer everywhere) and the case where
    only the source rank saw data (a rank without data must still join every
    reduction rather than deadlocking it).
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

            modifier = MoNePruningModifier(sparsity=0.5)
            state = _make_state(model)

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

            # every layer was collapsed from the globally summed statistics
            assert len(modifier._saliency_trackers) == 0
            for layer in model.layers:
                assert layer.experts.num_experts == 4
                assert layer.experts.novices is not None
                assert layer.experts.novices.shape == (4, config.hidden_size)

            # ... identically on every rank (retained experts and novices are
            # broadcast from the source rank)
            novices = torch.stack(
                [layer.experts.novices.data for layer in model.layers]
            ).to(torch.float64)
            gathered = [torch.empty_like(novices) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, novices.contiguous())
            for other in gathered:
                torch.testing.assert_close(other, novices)

        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.unit
@requires_gpu
@torchrun(world_size=1, init_dist=True)
def test_mone_distributed_nccl_reduces_on_gpu():
    """The saliency reduction runs on the gpu (float32) over the default
    nccl-backed process group, rather than on host copies."""
    device = torch.device("cuda", dist.get_rank())
    assert dist.get_backend() == "nccl"

    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config).to(device)
    model.eval()

    modifier = MoNePruningModifier(sparsity=0.5)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))

    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(2, 8, config.hidden_size, device=device))

    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # every layer collapsed, with novices installed on the gpu
    assert len(modifier._saliency_trackers) == 0
    for layer in model.layers:
        assert layer.experts.num_experts == 4
        assert layer.experts.novices is not None
        assert layer.experts.novices.shape == (4, config.hidden_size)
        assert layer.experts.novices.device == device
