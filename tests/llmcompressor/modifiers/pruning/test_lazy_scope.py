from unittest.mock import patch

import pytest
import torch

from llmcompressor.core import CompressionSession, EventType, State, create_session
from llmcompressor.modifiers.pruning.constant import ConstantPruningModifier
from llmcompressor.modifiers.pruning.magnitude import MagnitudePruningModifier
from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier
from llmcompressor.modifiers.pruning.wanda import WandaPruningModifier
from llmcompressor.pipelines import CalibrationPipeline


@pytest.mark.unit
@pytest.mark.parametrize("modifier_cls", [SparseGPTModifier, WandaPruningModifier])
def test_sequential_start_registers_hooks_only_in_scope(modifier_cls):
    model = torch.nn.ModuleDict(
        {
            "layers": torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(4, 4)) for _ in range(2)]
            )
        }
    )
    modifier = modifier_cls(sparsity=0.5)
    state = State(model=model)
    modifier.on_initialize(state)
    old_module = model.layers[0][0]
    model.layers[0][0] = torch.nn.Linear(4, 4)

    modifier.on_sequential_epoch_start(
        state=state,
        event=None,
        modules=list(model.layers[0].modules()),
    )

    assert model.layers[0][0] in modifier._module_names
    assert model.layers[1][0] not in modifier._module_names
    assert len(modifier._hooks) == 1
    assert not old_module._forward_hooks


@pytest.mark.unit
def test_owl_defers_statistics_until_basic_pipeline_start():
    model = torch.nn.Sequential(
        torch.nn.Sequential(torch.nn.Linear(4, 4)),
        torch.nn.Sequential(torch.nn.Linear(4, 4)),
    )
    modifier = SparseGPTModifier(
        sparsity=0.5,
        sparsity_profile="owl",
        owl_m=1,
        owl_lmbda=1.0,
    )

    state = State(model=model)
    with patch.object(
        SparseGPTModifier,
        "_infer_owl_layer_sparsity",
        return_value={"0": 0.4, "1": 0.6},
    ) as infer:
        modifier.on_initialize(state, sequential_targets=["0", "1"])
        infer.assert_not_called()
        assert not modifier._hooks

        modifier.on_calibration_start(state, None)
        infer.assert_called_once()

    assert modifier._module_sparsities == {model[0][0]: 0.4, model[1][0]: 0.6}
    assert not modifier._hessians
    modifier.remove_hooks()


def test_owl_statistics_do_not_dispatch_compression_events():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x)

    model = Model()
    modifier = SparseGPTModifier(sparsity=0.5)
    with patch("llmcompressor.core.LifecycleCallbacks.calibration_start") as start:
        activations = modifier._get_activations(
            model, [{"x": torch.ones(1, 2, 4)}], nsamples=1
        )
        start.assert_not_called()
    torch.testing.assert_close(activations["linear"], torch.full((4,), 2**0.5))
    assert not modifier._hooks


@pytest.mark.parametrize("modifier_cls", [SparseGPTModifier, WandaPruningModifier])
def test_owl_basic_pipeline_completes(modifier_cls):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Sequential(torch.nn.Linear(32, 32)),
                torch.nn.Sequential(torch.nn.Linear(32, 32)),
            )

        def forward(self, x):
            return self.layers(x)

    torch.manual_seed(0)
    model = Model()
    batches = [{"x": torch.randn(1, 4, 32)} for _ in range(4)]
    modifier = modifier_cls(
        sparsity=0.5, sparsity_profile="owl", owl_m=1, owl_lmbda=0.05
    )
    with create_session() as session:
        session.initialize(
            model=model,
            recipe=modifier,
            calib_data=batches,
            sequential_targets=["layers.0", "layers.1"],
        )
        assert not modifier._hooks
        CalibrationPipeline.from_modifiers([modifier])(model, batches, None)
        assert not modifier._hooks
        session.finalize()

    for layer in model.layers:
        assert 0.4 < (layer[0].weight == 0).float().mean() < 0.6


@pytest.mark.parametrize(
    "modifier",
    [
        ConstantPruningModifier(targets="Linear", start=0),
        MagnitudePruningModifier(
            targets="Linear", start=0, end=1, init_sparsity=0.5, final_sparsity=0.5
        ),
    ],
)
def test_mask_setup_is_deferred_and_scoped(modifier):
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    session = CompressionSession()
    session.initialize(model=model, recipe=modifier)
    assert not modifier.started_
    assert all(not hasattr(module, "mask") for module in model)

    session.lifecycle.event(EventType.CALIBRATION_START, modules=[])
    assert not modifier.parameterized_layers_
    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_START, modules=[model[0]])
    assert hasattr(model[0], "mask")
    assert not hasattr(model[1], "mask")
    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_START, modules=[model[0]])
    assert len(modifier.parameterized_layers_) == 1
    session.lifecycle.event(EventType.SEQUENTIAL_EPOCH_END, modules=[model[0]])
    if isinstance(modifier, MagnitudePruningModifier):
        assert torch.count_nonzero(model[0].weight) == 8
    session.finalize()


@pytest.mark.parametrize("kind", ["constant", "magnitude"])
def test_training_start_initializes_masks(kind):
    model = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        model.weight[:, :2] = 0
    modifier = (
        ConstantPruningModifier(targets="Linear", start=0, end=1)
        if kind == "constant"
        else MagnitudePruningModifier(
            targets="Linear", start=0, end=1, init_sparsity=0.5, final_sparsity=0.5
        )
    )
    session = CompressionSession()
    session.initialize(model=model, recipe=modifier)
    assert not hasattr(model, "mask")
    session.lifecycle.event(EventType.BATCH_START)
    assert modifier.started_
    expected_mask = model.mask.clone()
    with torch.no_grad():
        model.weight.fill_(1)
    session.lifecycle.event(EventType.OPTIM_POST_STEP)
    torch.testing.assert_close(model.weight, expected_mask.to(model.weight.dtype))
    session.finalize()
