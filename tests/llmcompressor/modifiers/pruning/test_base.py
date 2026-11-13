"""
Unit tests for the ``ExpertPruningModifierBase`` template shared by the REAP,
MoNe, and RAZOR expert pruning modifiers.

The base owns the pruning lifecycle: argument validation, MoE layer
detection, hook attachment, per-layer finalization at
``SEQUENTIAL_EPOCH_END``, retained-expert selection, expert-sparsity config
recording, and report writing. The lifecycle is driven through
``_RecordingModifier``, a minimal concrete subclass whose saliency is the
per-expert routed-token count, so the template is exercised without any real
saliency metric. REAP-, MoNe-, and RAZOR-specific behavior is tested in
their own modules.
"""

import json

import pytest
import torch
from compressed_tensors.config import MoESparsityMethod
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning import (
    ExpertPruningModifierBase,
    MoNePruningModifier,
    RazorPruningModifier,
    REAPPruningModifier,
)
from tests.llmcompressor.modifiers.pruning.fake_moe import (
    FakeMoEConfig,
    FakeMoEModel,
    make_state,
)


class _RecordingModifier(ExpertPruningModifierBase):
    """Minimal concrete subclass recording calls into the template hooks.

    Its saliency metric is the per-expert number of routed tokens, so
    retained-expert selections and reports can be checked against the
    recorded counts."""

    sparsity_method = MoESparsityMethod.PRUNE

    _expert_hook_calls: list = PrivateAttr(default_factory=list)
    _block_hook_calls: list = PrivateAttr(default_factory=list)
    _reduced_layers: list = PrivateAttr(default_factory=list)

    def init_moe_stats(
        self,
        layer_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        self._moe_stats[layer_name] = {
            "count": torch.zeros(num_experts, dtype=torch.float32, device=device),
        }

    def expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output: torch.Tensor,
    ):
        if layer_name in self._moe_stats:
            self._expert_hook_calls.append((layer_name, expert_idx))

    def experts_block_hook(
        self,
        layer_name: str,
        moe: torch.nn.Module,
        module: torch.nn.Module,
        args: tuple,
        output: torch.Tensor,
    ):
        if layer_name not in self._moe_stats:
            return
        self._block_hook_calls.append(layer_name)
        topk_indices = args[1]
        self._moe_stats[layer_name]["count"].index_add_(
            0,
            topk_indices.reshape(-1).to(torch.long),
            torch.ones(topk_indices.numel(), dtype=torch.float32),
        )

    def reduce_moe_stats(self, layer_name: str):
        self._reduced_layers.append(layer_name)

    def saliency(self, layer_name: str) -> torch.Tensor:
        return self._moe_stats[layer_name]["count"]


def _model(num_experts=8, num_hidden_layers=2, **config_kwargs) -> FakeMoEModel:
    config = FakeMoEConfig(
        num_experts=num_experts,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        **config_kwargs,
    )
    model = FakeMoEModel(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tests: registration and argument validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.usefixtures("setup_modifier_factory")
@pytest.mark.parametrize(
    "modifier_cls",
    [REAPPruningModifier, MoNePruningModifier, RazorPruningModifier],
)
def test_expert_pruning_modifiers_are_registered(modifier_cls):
    """All expert pruning modifiers resolve from their recipe names."""
    modifier = ModifierFactory.create(
        type_=modifier_cls.__name__,
        allow_experimental=False,
        allow_registered=True,
        sparsity=0.5,
    )
    assert isinstance(modifier, modifier_cls)


@pytest.mark.unit
@pytest.mark.parametrize(
    "modifier_cls", [REAPPruningModifier, MoNePruningModifier, RazorPruningModifier]
)
def test_validates_arguments(modifier_cls):
    # sparsity must be in [0, 1)
    with pytest.raises(ValueError, match="sparsity"):
        modifier_cls(sparsity=1.0)
    with pytest.raises(ValueError, match="sparsity"):
        modifier_cls(sparsity=-0.1)

    # report_path must be a .json file
    with pytest.raises(ValueError, match=r"\.json"):
        modifier_cls(sparsity=0.5, report_path="report.pkl")

    # report-only mode (sparsity == 0) requires a report_path, since the
    # report is then the modifier's only output
    with pytest.raises(ValueError, match="report_path"):
        modifier_cls(sparsity=0.0)

    modifier_cls(sparsity=0.0, report_path="report.json")


# ---------------------------------------------------------------------------
# Tests: initialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "modifier_cls", [REAPPruningModifier, MoNePruningModifier, RazorPruningModifier]
)
def test_initialize_rejects_sparsity_that_prunes_no_experts(modifier_cls):
    """A sparsity which rounds to zero experts pruned is a no-op, so it fails
    fast rather than silently doing nothing."""
    modifier = modifier_cls(sparsity=0.001)
    with pytest.raises(ValueError, match="0 experts to prune"):
        modifier.initialize(make_state(_model(num_experts=100)))


@pytest.mark.unit
@pytest.mark.parametrize("modifier_cls", [REAPPruningModifier, RazorPruningModifier])
def test_initialize_rejects_overly_aggressive_sparsity(modifier_cls):
    """Pruning below the router's top_k would leave it unable to route."""
    config = FakeMoEConfig(num_experts=8)
    config.num_experts_per_tok = 5
    model = FakeMoEModel(config)

    modifier = modifier_cls(sparsity=0.75)  # would leave only 2 of 8 experts
    with pytest.raises(ValueError, match="too aggressive"):
        modifier.initialize(make_state(model))


@pytest.mark.unit
def test_mone_initialize_accepts_aggressive_sparsity():
    """MoNe collapses dropped experts into constant novices which stay
    routable (the router keeps its full width), so no sparsity is too
    aggressive for it."""
    config = FakeMoEConfig(num_experts=8)
    config.num_experts_per_tok = 5
    model = FakeMoEModel(config)

    modifier = MoNePruningModifier(sparsity=0.75)  # would leave only 2 of 8 experts
    modifier.initialize(make_state(model))
    assert modifier.initialized
    assert modifier._n_experts_to_drop == 6


# ---------------------------------------------------------------------------
# Tests: retained-expert selection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compute_retained_experts_drops_lowest_saliency():
    """The lowest-saliency experts are dropped globally, with the retained
    experts returned in ascending index order."""
    modifier = REAPPruningModifier(sparsity=0.375)  # 3 of 8
    modifier.initialize(make_state(FakeMoEModel(FakeMoEConfig(num_experts=8))))
    modifier.init_moe_stats("layer", 8, 4, torch.device("cpu"))
    stats = modifier._moe_stats["layer"]
    stats["sum_saliency"] = torch.tensor(
        [0.1, 0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4], dtype=torch.float32
    )
    stats["count"] = torch.ones(8, dtype=torch.int)

    retained, saliency = modifier.compute_retained_experts(
        "layer", stats["count"].device
    )

    # the 3 lowest-saliency experts (0, 5, 3) are dropped
    assert retained == [1, 2, 4, 6, 7]
    assert saliency.tolist() == pytest.approx([0.1, 0.5, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4])


@pytest.mark.unit
def test_compute_retained_experts_is_balanced_per_group():
    """Group-limited routers must keep every group routable, so the drop is
    split equally across groups."""
    modifier = REAPPruningModifier(sparsity=0.25)  # 2 of 8, 1 per group
    modifier.initialize(
        make_state(FakeMoEModel(FakeMoEConfig(num_experts=8, n_group=2, topk_group=1)))
    )
    modifier.init_moe_stats("layer", 8, 4, torch.device("cpu"))
    stats = modifier._moe_stats["layer"]
    stats["sum_saliency"] = torch.tensor(
        [0.1, 0.9, 0.8, 0.7, 0.6, 0.2, 0.5, 0.4], dtype=torch.float32
    )
    stats["count"] = torch.ones(8, dtype=torch.int)

    retained, _ = modifier.compute_retained_experts("layer", stats["count"].device)

    # group [0, 4): drop expert 0; group [4, 8): drop expert 5
    assert retained == [1, 2, 3, 4, 6, 7]


# ---------------------------------------------------------------------------
# Tests: lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lifecycle_prunes_each_calibrated_layer(tmp_path):
    """The base detects the MoE layers, attaches the subclass's hooks, hands
    each completed layer's statistics to the subclass's saliency, prunes its
    lowest-saliency experts at ``SEQUENTIAL_EPOCH_END``, records the
    expert-sparsity config, and writes the subclass's statistics to
    ``report_path`` in MoE-layer order."""
    torch.manual_seed(42)
    model = _model(num_experts=8, num_hidden_layers=2)
    report_path = str(tmp_path / "report.json")
    modifier = _RecordingModifier(sparsity=0.5, report_path=report_path)
    state = make_state(model)

    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    # calibration start: both MoE layers are detected and initialized with
    # statistics
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert sorted(modifier._moe_stats) == ["layers.0", "layers.1"]

    # an epoch end before any data: every layer is still reduced (so
    # distributed ranks stay aligned) but stays pending and unpruned
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert sorted(modifier._reduced_layers) == ["layers.0", "layers.1"]
    assert len(modifier._moe_stats) == 2
    assert modifier._retained_experts == {}

    # calibration data flows through the subclass's hooks
    with torch.no_grad():
        model(torch.randn(2, 8, 16))
    assert modifier._block_hook_calls == ["layers.0", "layers.1"]
    assert len(modifier._expert_hook_calls) > 0
    assert all(stats["count"].sum() > 0 for stats in modifier._moe_stats.values())

    # epoch end: every layer saw tokens, so each is pruned to its retained
    # experts and its statistics are freed
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._moe_stats) == 0
    for moe_name, retained in modifier._retained_experts.items():
        assert len(retained) == 4
        layer = model.get_submodule(moe_name)
        assert layer.experts.num_experts == 4
        assert layer.gate.weight.shape[0] == 4

    # calibration end: the hooks are removed
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    assert len(modifier._hooks) == 0

    modifier.finalize(state)

    # the expert-sparsity config records the per-layer retained counts with
    # the subclass's sparsity method
    assert model._moe_sparsity_config.method == MoESparsityMethod.PRUNE
    assert model._moe_sparsity_config.num_experts == [4, 4]

    # the report holds the subclass's statistics, ordered by layer
    with open(report_path) as f:
        report = json.load(f)
    assert sorted(report) == ["count", "saliency"]
    for values in report.values():
        assert len(values) == 2
        assert len(values[0]) == 8


@pytest.mark.unit
def test_report_only_mode_skips_pruning(tmp_path):
    """With ``sparsity=0.0`` statistics are still collected and reported, but
    no experts are pruned and no sparsity config is recorded."""
    torch.manual_seed(42)
    model = _model(num_experts=8, num_hidden_layers=2)
    report_path = str(tmp_path / "report.json")
    modifier = _RecordingModifier(sparsity=0.0, report_path=report_path)
    state = make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    with torch.no_grad():
        model(torch.randn(2, 8, 16))
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # no layer was pruned or marked as pruned
    assert all(retained is None for retained in modifier._retained_experts.values())
    assert not hasattr(model, "_moe_sparsity_config")
    for layer in model.layers:
        assert layer.experts.num_experts == 8
        assert layer.gate.weight.shape[0] == 8

    with open(report_path) as f:
        report = json.load(f)
    assert len(report["saliency"]) == 2


@pytest.mark.unit
def test_ignored_layers_collect_stats_but_are_not_pruned():
    """Layers matching an ``ignore`` pattern are still tracked (their
    statistics join the report and the distributed collectives) but never
    pruned; the expert-sparsity config marks them with ``None``."""
    torch.manual_seed(42)
    model = _model(num_experts=8, num_hidden_layers=3)
    modifier = _RecordingModifier(sparsity=0.5, ignore=["layers.1"])
    state = make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert len(modifier._moe_stats) == 3

    with torch.no_grad():
        model(torch.randn(2, 8, 16))
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    assert model.layers[0].experts.num_experts == 4
    assert model.layers[1].experts.num_experts == 8  # ignored
    assert model.layers[1].gate.weight.shape[0] == 8
    assert model.layers[2].experts.num_experts == 4

    # ignored layers are marked as unpruned in the per-layer config
    assert model._moe_sparsity_config.num_experts == [4, None, 4]
