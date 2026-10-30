"""
Tests for the ``ExpertPruningModifierBase`` template shared by the REAP and
MoNe expert pruning modifiers: the subclass relationship, the abstract
interface, and the lifecycle wiring (module identification, overly-aggressive
sparsity validation, hook attachment, per-layer finalization at
``SEQUENTIAL_EPOCH_END``, expert-sparsity config recording, and report
writing).
"""

import json

import pytest
import torch
from compressed_tensors.config import MoESparsityMethod
from pydantic import PrivateAttr

from llmcompressor.core import Event, EventType
from llmcompressor.modifiers.pruning import ExpertPruningModifierBase
from llmcompressor.modifiers.pruning.mone import MoNePruningModifier
from llmcompressor.modifiers.pruning.reap import REAPPruningModifier
from tests.llmcompressor.modifiers.pruning.reap.test_base import (
    FakeMoEConfig,
    FakeMoEModel,
    _make_state,
)


class _RecordingModifier(ExpertPruningModifierBase):
    """Minimal concrete subclass recording calls into the template hooks."""

    sparsity_method = MoESparsityMethod.PRUNE

    # minimal per-layer tracking state: expert-hook firings mark the layer as
    # having collected calibration data
    _seen: dict[str, float] = PrivateAttr(default_factory=dict)
    _pruned: list = PrivateAttr(default_factory=list)
    _expert_hook_calls: list = PrivateAttr(default_factory=list)
    _block_hook_calls: list = PrivateAttr(default_factory=list)

    def on_calibration_start(self, state, event, **kwargs):
        for layer_name in self._moe_attrs.moe_layer_names:
            self._seen[layer_name] = 0.0
        super().on_calibration_start(state, event, **kwargs)

    def _expert_hook(
        self,
        layer_name: str,
        expert_idx: int,
        module: torch.nn.Module,
        args: tuple,
        output,
    ):
        self._expert_hook_calls.append((layer_name, expert_idx))
        self._seen[layer_name] = 1.0

    def _experts_block_hook(
        self,
        layer_name: str,
        module: torch.nn.Module,
        args: tuple,
        output,
    ):
        self._block_hook_calls.append(layer_name)

    def _prune_moe_layer(self, model: torch.nn.Module, layer_name: str) -> bool:
        if self._seen.get(layer_name, 0.0) == 0.0:
            # no calibration data yet; keep the layer pending
            return False
        self._report_values[layer_name] = {
            "count": list(range(self._moe_attrs.num_experts)),
            "saliency": [float(i) for i in range(self._moe_attrs.num_experts)],
        }
        if self._n_experts_to_drop > 0:
            retained = list(
                range(self._moe_attrs.num_experts - self._n_experts_to_drop)
            )
            self._retained_experts[layer_name] = retained
            self._pruned.append(layer_name)
        return True


@pytest.mark.unit
def test_expert_pruning_modifiers_share_base():
    assert issubclass(REAPPruningModifier, ExpertPruningModifierBase)
    assert issubclass(MoNePruningModifier, ExpertPruningModifierBase)
    assert REAPPruningModifier.sparsity_method == MoESparsityMethod.PRUNE
    assert MoNePruningModifier.sparsity_method == MoESparsityMethod.NOVICE


@pytest.mark.unit
def test_expert_pruning_base_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        ExpertPruningModifierBase(sparsity=0.5)


@pytest.mark.unit
def test_base_validates_overly_aggressive_sparsity():
    """The overly-aggressive sparsity check is shared by every subclass; MoNe
    gains it alongside REAP."""
    config = FakeMoEConfig(num_experts=8)
    config.num_experts_per_tok = 5
    model = FakeMoEModel(config)

    for modifier_cls in (REAPPruningModifier, MoNePruningModifier):
        modifier = modifier_cls(sparsity=0.75)
        with pytest.raises(ValueError, match="too aggressive"):
            modifier.initialize(_make_state(model))


@pytest.mark.unit
def test_base_lifecycle_drives_subclass_hooks(tmp_path):
    """The base identifies the MoE layers, attaches the subclass's capture
    hooks, hands each completed layer to ``_prune_moe_layer`` and releases it,
    records the expert-sparsity config from ``_retained_experts``, and writes
    the subclass's ``_report_values`` to ``report_path`` in MoE-layer order."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    report_path = str(tmp_path / "report.json")
    modifier = _RecordingModifier(sparsity=0.5, report_path=report_path)
    state = _make_state(model)

    modifier.initialize(state)
    assert modifier._n_experts_to_drop == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    assert len(modifier._pending_layers) == 2

    # a layer without calibration data yet stays pending across epochs
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._pending_layers) == 2
    assert modifier._pruned == []

    # the base attached the subclass's hooks: expert outputs flow through
    # _expert_hook and _experts_block_hook during calibration
    with torch.no_grad():
        model(torch.randn(2, 8, config.hidden_size))
    assert len(modifier._expert_hook_calls) > 0
    assert len(modifier._block_hook_calls) == 2

    # every layer saw tokens, so every layer is finalized and pruned
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._pending_layers) == 0
    assert sorted(modifier._pruned) == sorted(modifier._moe_attrs.moe_layer_names)

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # shared finalization: the expert-sparsity config is recorded from
    # _retained_experts with the subclass's sparsity method
    assert model._moe_sparsity_config.method == MoESparsityMethod.PRUNE
    assert model._moe_sparsity_config.num_experts == [4, 4]

    # report written from the subclass's _report_values, ordered by layer
    with open(report_path) as f:
        report = json.load(f)
    assert sorted(report) == ["count", "saliency"]
    for values in report.values():
        assert len(values) == 2
        assert values[0] == [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.unit
def test_base_report_only_mode_skips_pruning(tmp_path):
    """With ``sparsity=0.0`` the base still drives per-layer finalization (so
    the report is written) but no sparsity config is recorded."""
    torch.manual_seed(42)
    config = FakeMoEConfig(
        num_experts=8, hidden_size=16, intermediate_size=32, num_hidden_layers=2
    )
    model = FakeMoEModel(config)
    model.eval()

    report_path = str(tmp_path / "report.json")
    modifier = _RecordingModifier(sparsity=0.0, report_path=report_path)
    state = _make_state(model)

    modifier.initialize(state)
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_START))
    with torch.no_grad():
        model(torch.randn(2, 8, config.hidden_size))
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    assert modifier._pruned == []
    assert not hasattr(model, "_moe_sparsity_config")

    with open(report_path) as f:
        report = json.load(f)
    assert len(report["saliency"]) == 2
