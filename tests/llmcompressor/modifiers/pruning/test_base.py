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
    """Minimal concrete subclass recording calls into the template hooks.

    Its saliency metric is the per-expert number of routed tokens, and it
    uses the base class's default capture (full detached outputs, whose
    token counts are recorded) and default structural pruning."""

    sparsity_method = MoESparsityMethod.PRUNE

    _captured_experts: list = PrivateAttr(default_factory=list)
    _updated_layers: list = PrivateAttr(default_factory=list)

    def _init_layer_stats(
        self,
        layer_name: str,
        num_experts: int,
        hidden_dim: int,
        device: torch.device,
    ):
        self._layer_stats[layer_name] = {
            "counts": torch.zeros(num_experts, dtype=torch.float32, device=device)
        }

    def _capture_expert_output(self, output: torch.Tensor) -> torch.Tensor:
        self._captured_experts.append(output.shape[0])
        return torch.linalg.norm(output.float(), dim=-1).reshape(-1)

    def _update_stats(
        self,
        layer_name: str,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        captured: dict[int, torch.Tensor],
    ):
        self._updated_layers.append(layer_name)
        stats = self._layer_stats[layer_name]
        flat_idx = topk_indices.reshape(-1).to(torch.long)
        stats["counts"].index_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
        )
        self._layer_counts[layer_name] += topk_indices.shape[0]

    def _saliency(self, layer_name: str) -> torch.Tensor:
        return self._layer_stats[layer_name]["counts"]


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
    hooks, hands each completed layer's statistics to the subclass's metric
    and its default structural pruning, records the expert-sparsity config
    from the decisions it made, and writes the subclass's statistics to
    ``report_path`` in MoE-layer order."""
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
    assert len(modifier._layer_stats) == 2

    # a layer without calibration data yet stays pending across epochs
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._layer_stats) == 2
    assert modifier._retained_experts == {}

    # the base attached the subclass's hooks: expert outputs flow through
    # _capture_expert_output and _update_stats during calibration
    with torch.no_grad():
        model(torch.randn(2, 8, config.hidden_size))
    assert len(modifier._captured_experts) > 0
    assert len(modifier._updated_layers) == 2

    # every layer saw tokens, so every layer is finalized and pruned
    modifier.update_event(state, Event(type_=EventType.SEQUENTIAL_EPOCH_END))
    assert len(modifier._layer_stats) == 0
    assert sorted(modifier._retained_experts) == sorted(
        modifier._moe_attrs.moe_layer_names
    )
    for retained in modifier._retained_experts.values():
        assert len(retained) == 4

    modifier.update_event(state, Event(type_=EventType.CALIBRATION_END))
    modifier.finalize(state)

    # shared finalization: the expert-sparsity config is recorded from the
    # base class's decisions with the subclass's sparsity method
    assert model._moe_sparsity_config.method == MoESparsityMethod.PRUNE
    assert model._moe_sparsity_config.num_experts == [4, 4]

    # report written from the subclass's statistics, ordered by layer
    with open(report_path) as f:
        report = json.load(f)
    assert sorted(report) == ["count", "saliency"]
    for values in report.values():
        assert len(values) == 2
        assert len(values[0]) == 8
        assert all(isinstance(value, float) for value in values[0])


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

    assert modifier._retained_experts == {}
    assert not hasattr(model, "_moe_sparsity_config")

    with open(report_path) as f:
        report = json.load(f)
    assert len(report["saliency"]) == 2
