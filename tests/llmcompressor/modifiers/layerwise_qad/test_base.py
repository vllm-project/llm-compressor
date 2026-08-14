import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    enable_quantization,
)

from llmcompressor.core import Event, EventType, State
from llmcompressor.entrypoints.oneshot import Oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.layerwise_qad.base import (
    LayerwiseQADModifier,
    _LayerBatch,
    _masked_mse,
)
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import (
    apply_calibration_status,
    observe,
    update_qparams,
)
from llmcompressor.modifiers.utils.hooks import HooksMixin
from llmcompressor.pipelines.sequential.pipeline import _configure_modifier_pipeline
from llmcompressor.utils.helpers import DisableQuantization


class _ToyBlock(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, scale=1.0):
        return (torch.tanh(self.proj(hidden_states)) * scale,)


class _ToyModel(torch.nn.Module):
    _no_split_modules = ["_ToyBlock"]

    def __init__(self, depth=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyBlock() for _ in range(depth)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        return hidden_states


def _quantized_linear():
    layer = torch.nn.Linear(4, 4, bias=False)
    layer.quantization_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            symmetric=True,
            strategy="tensor",
        ),
    )
    layer.weight_scale = torch.nn.Parameter(
        layer.weight.detach().abs().max().reshape(1) / 127,
        requires_grad=False,
    )
    layer.weight_zero_point = torch.nn.Parameter(
        torch.zeros(1),
        requires_grad=False,
    )
    return layer


def _initialize_fake_quantization(block):
    modifier = QuantizationModifier(
        config_groups={
            "group": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "tensor",
                },
            }
        }
    )
    modifier.initialize_quantization(block)
    for module in block.modules():
        if getattr(module, "quantization_scheme", None) is None:
            continue
        modifier._initialize_observers(module)
        apply_calibration_status(module)
        observe([module], "weight")
        update_qparams([module], "weight")
    block.apply(enable_quantization)


def test_masked_mse_excludes_padding():
    prediction = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[1, 0]])

    assert _masked_mse(prediction, target, mask).item() == pytest.approx(5.0)


def test_initialize_requires_teacher():
    modifier = LayerwiseQADModifier()

    with pytest.raises(ValueError, match="requires a full-precision teacher"):
        modifier.on_initialize(
            State(model=_ToyModel()),
            sequential_targets=["_ToyBlock"],
        )


def test_train_uses_partial_accumulation_group():
    modifier = LayerwiseQADModifier(
        num_epochs=2,
        gradient_accumulation_steps=2,
        learning_rate=0.01,
    )
    student = _ToyBlock()
    teacher = _ToyBlock()
    batches = [_LayerBatch((torch.randn(1, 2, 4),), {}, None) for _ in range(3)]
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

    steps = modifier._train(
        student,
        teacher,
        batches,
        optimizer,
        list(student.parameters()),
        torch.device("cpu"),
        torch.float32,
    )

    assert steps == 4


def test_sequential_epoch_optimizes_current_block_only():
    torch.manual_seed(7)
    student = _ToyModel()
    teacher = _ToyModel()
    modifier = LayerwiseQADModifier(
        num_epochs=10,
        learning_rate=0.05,
        gradient_accumulation_steps=2,
        max_grad_norm=None,
        early_stopping_patience=10,
    )
    state = State(model=student, teacher_model=teacher)
    modifier.on_initialize(state, sequential_targets=["_ToyBlock"])
    modifier.on_calibration_start(
        state,
        Event(type_=EventType.CALIBRATION_START),
    )

    block = student.layers[0]
    block.proj = _quantized_linear()
    modifier._module_names = {
        student.layers[0]: "layers.0",
        student.layers[1]: "layers.1",
    }
    modifier._student_to_teacher = {
        student.layers[0]: teacher.layers[0],
        student.layers[1]: teacher.layers[1],
    }
    modifier._layer_inputs = {
        student.layers[0]: [],
        student.layers[1]: [],
    }
    modifier.remove_hooks()
    for layer in student.layers:
        modifier.register_hook(
            layer,
            modifier._capture_input,
            "forward_pre",
            with_kwargs=True,
        )

    untouched = student.layers[1].proj.weight.detach().clone()
    for batch_index in range(4):
        state.current_batch_idx = batch_index
        block(torch.randn(2, 3, 4), scale=0.75)

    with (
        patch(
            "llmcompressor.modifiers.layerwise_qad.base.is_module_quantized",
            side_effect=lambda module: hasattr(module, "quantization_scheme"),
        ),
        patch("llmcompressor.modifiers.layerwise_qad.base.enable_quantization"),
        patch(
            "llmcompressor.modifiers.layerwise_qad.base.get_execution_device",
            return_value=torch.device("cpu"),
        ),
        patch.object(
            modifier,
            "_materialize_quantized_weights",
        ),
    ):
        modifier.on_sequential_epoch_end(
            state,
            Event(type_=EventType.SEQUENTIAL_EPOCH_END),
            modules=list(block.modules()),
        )

    assert modifier.optimizer_steps["layers.0"] == 20
    assert torch.equal(student.layers[1].proj.weight, untouched)
    assert not modifier._layer_inputs[student.layers[0]]


def test_materialize_uses_existing_quantization_parameters():
    modifier = LayerwiseQADModifier()
    block = _ToyBlock()
    block.proj = _quantized_linear()

    with (
        patch(
            "llmcompressor.modifiers.layerwise_qad.base.is_module_quantized",
            side_effect=lambda module: hasattr(module, "quantization_scheme"),
        ),
        patch(
            "llmcompressor.modifiers.layerwise_qad.base.forward_quantize",
            return_value=torch.zeros_like(block.proj.weight),
        ) as quantize,
        patch(
            "llmcompressor.modifiers.layerwise_qad.base.update_offload_parameter"
        ) as update,
    ):
        modifier._materialize_quantized_weights(block)

    quantize.assert_called_once()
    update.assert_called_once()


def test_real_fake_quant_training_reduces_hidden_state_mse():
    torch.manual_seed(11)
    teacher = _ToyBlock()
    student = _ToyBlock()
    _initialize_fake_quantization(student)
    batches = [_LayerBatch((torch.randn(2, 3, 4),), {}, None) for _ in range(4)]
    modifier = LayerwiseQADModifier(
        num_epochs=30,
        learning_rate=0.02,
        gradient_accumulation_steps=2,
        max_grad_norm=None,
        early_stopping_patience=30,
    )
    modifier._module_names = {student: "block"}

    with torch.no_grad(), HooksMixin.disable_hooks():
        initial_loss = sum(
            _masked_mse(
                student(*batch.args)[0],
                teacher(*batch.args)[0],
                None,
            ).item()
            for batch in batches
        ) / len(batches)

    modifier._optimize_block(student, teacher, batches)

    with torch.no_grad(), HooksMixin.disable_hooks():
        final_loss = sum(
            _masked_mse(
                student(*batch.args)[0],
                teacher(*batch.args)[0],
                None,
            ).item()
            for batch in batches
        ) / len(batches)

    assert final_loss < initial_loss
    assert modifier.optimizer_steps["block"] == 60


def test_gptq_and_layerwise_qad_run_on_same_block():
    torch.manual_seed(17)
    teacher = _ToyModel(depth=1)
    student = _ToyModel(depth=1)
    state = State(model=student, teacher_model=teacher)
    quantization_config = {
        "group": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 8,
                "type": "int",
                "symmetric": True,
                "strategy": "tensor",
            },
        }
    }
    gptq = GPTQModifier(
        config_groups=quantization_config,
        actorder=None,
    )
    qad = LayerwiseQADModifier(
        num_epochs=20,
        learning_rate=0.02,
        gradient_accumulation_steps=2,
        max_grad_norm=None,
        early_stopping_patience=20,
    )
    gptq.on_initialize(state)
    qad.on_initialize(state, sequential_targets=["_ToyBlock"])
    event = Event(type_=EventType.CALIBRATION_START)
    gptq.on_calibration_start(state, event)
    qad.on_calibration_start(state, event)
    block = student.layers[0]
    inputs = [torch.randn(2, 3, 4) for _ in range(4)]

    with DisableQuantization(student):
        for batch_index, hidden_states in enumerate(inputs):
            state.current_batch_idx = batch_index
            block(hidden_states)

        modules = list(block.modules())
        sequential_event = Event(type_=EventType.SEQUENTIAL_EPOCH_END)
        gptq.on_sequential_epoch_end(state, sequential_event, modules)
        qad._capture_enabled = False
        block.apply(enable_quantization)

        with torch.no_grad(), HooksMixin.disable_hooks():
            initial_loss = sum(
                _masked_mse(
                    block(hidden_states)[0],
                    teacher.layers[0](hidden_states)[0],
                    None,
                ).item()
                for hidden_states in inputs
            ) / len(inputs)

        qad.on_sequential_epoch_end(state, sequential_event, modules)
        assert not gptq._num_samples
        assert not gptq._hessians

        with torch.no_grad(), HooksMixin.disable_hooks():
            final_loss = sum(
                _masked_mse(
                    block(hidden_states)[0],
                    teacher.layers[0](hidden_states)[0],
                    None,
                ).item()
                for hidden_states in inputs
            ) / len(inputs)

    assert final_loss < initial_loss
    assert qad.optimizer_steps["layers.0"] == 40


def test_validation_early_stopping_restores_best_weights():
    modifier = LayerwiseQADModifier(
        num_epochs=5,
        early_stopping_patience=1,
        max_grad_norm=None,
    )
    student = _ToyBlock()
    teacher = _ToyBlock()
    batches = [_LayerBatch((torch.randn(1, 2, 4),), {}, None) for _ in range(4)]
    trainable = list(student.parameters())
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    epoch = 0

    def train_epoch(*args, **kwargs):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            student.proj.weight.fill_(epoch)
        return 1

    with (
        patch.object(
            modifier,
            "_evaluate_indices",
            side_effect=[0.5, 0.4, 0.6],
        ),
        patch.object(modifier, "_train_epoch", side_effect=train_epoch),
    ):
        steps, epochs, best_loss = modifier._train_with_validation(
            student,
            teacher,
            batches,
            optimizer,
            trainable,
            [0, 1, 2],
            [3],
            torch.device("cpu"),
            torch.float32,
        )

    assert steps == 2
    assert epochs == 2
    assert best_loss == pytest.approx(0.4)
    assert torch.all(student.proj.weight == 1)


def test_validation_split_is_deterministic():
    modifier = LayerwiseQADModifier(validation_fraction=0.1, seed=17)

    first = modifier._split_batch_indices(10)
    second = modifier._split_batch_indices(10)

    assert first == second
    assert len(first[0]) == 9
    assert len(first[1]) == 1
    assert set(first[0]).isdisjoint(first[1])


def test_patience_tolerates_fluctuations_and_uses_cumulative_min_delta():
    modifier = LayerwiseQADModifier(
        num_epochs=10,
        early_stopping_patience=3,
        validation_relative_min_delta=0.001,
        max_grad_norm=None,
    )
    student = _ToyBlock()
    teacher = _ToyBlock()
    modifier._module_names = {student: "block"}
    batches = [_LayerBatch((torch.randn(1, 2, 4),), {}, None) for _ in range(4)]
    trainable = list(student.parameters())
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    epoch = 0

    def train_epoch(*args, **kwargs):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            student.proj.weight.fill_(epoch)
        return 1

    validation_losses = [1.0, 0.9995, 0.9989, 1.0, 1.0, 1.0]
    with (
        patch.object(
            modifier,
            "_evaluate_indices",
            side_effect=validation_losses,
        ),
        patch.object(modifier, "_train_epoch", side_effect=train_epoch),
    ):
        steps, epochs, best_loss = modifier._train_with_validation(
            student,
            teacher,
            batches,
            optimizer,
            trainable,
            [0, 1, 2],
            [3],
            torch.device("cpu"),
            torch.float32,
        )

    assert steps == 5
    assert epochs == 5
    assert best_loss == pytest.approx(0.9989)
    assert torch.all(student.proj.weight == 2)
    assert modifier.validation_histories["block"] == validation_losses


def test_factory_discovers_modifier():
    from llmcompressor.modifiers import ModifierFactory

    with patch.object(ModifierFactory, "_loaded", False):
        ModifierFactory.refresh()
        modifier = ModifierFactory.create(
            "LayerwiseQADModifier",
            allow_registered=True,
            allow_experimental=True,
        )

    assert isinstance(modifier, LayerwiseQADModifier)


def test_pipeline_requires_quantized_error_propagation():
    args = MagicMock(
        propagate_error=False,
        sequential_targets_per_subgraph=1,
    )

    with pytest.raises(ValueError, match="propagate_error=True"):
        _configure_modifier_pipeline([LayerwiseQADModifier()], args)


def test_pipeline_requires_one_block_per_subgraph():
    args = MagicMock(
        propagate_error=True,
        sequential_targets_per_subgraph=2,
    )

    with pytest.raises(ValueError, match="sequential_targets_per_subgraph=1"):
        _configure_modifier_pipeline([LayerwiseQADModifier()], args)


def test_oneshot_passes_teacher_to_compression_session():
    student = _ToyModel()
    teacher = _ToyModel()
    runner = Oneshot.__new__(Oneshot)
    runner.model = student
    runner.recipe = []
    runner.model_args = SimpleNamespace(distill_teacher=teacher)
    runner.recipe_args = SimpleNamespace(recipe_args=None)
    runner.dataset_args = SimpleNamespace(
        moe_calibrate_all_experts=False,
        sequential_targets=["_ToyBlock"],
        enable_compile=False,
        pipeline="sequential",
    )
    session = MagicMock()
    pipeline = MagicMock()
    oneshot_module = importlib.import_module("llmcompressor.entrypoints.oneshot")

    with (
        patch.object(
            oneshot_module,
            "active_session",
            return_value=session,
        ),
        patch.object(
            oneshot_module,
            "get_non_linearized_moes",
            return_value=[],
        ),
        patch.object(
            oneshot_module,
            "norm_calibration_context",
            return_value=nullcontext(),
        ),
        patch.object(
            oneshot_module.CalibrationPipeline,
            "from_modifiers",
            return_value=pipeline,
        ),
    ):
        runner.apply_recipe_modifiers(None)

    assert session.initialize.call_args.kwargs["teacher_model"] is teacher
