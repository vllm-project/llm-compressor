import os

import pytest
import torch

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.constant import ConstantPruningModifier
from llmcompressor.modifiers.pruning.utils.pytorch.layer_mask import param_mask_name
from llmcompressor.pytorch.utils import tensor_sparsity
from tests.llmcompressor.pytorch.helpers import ConvNet, LinearNet


def _induce_sparsity(model, sparsity=0.5):
    """
    Introduces sparsity to the given model by zeroing out weights
    with a probability of sparsity

    :param model: the model to introduce sparsity to
    :param sparsity: the probability of zeroing out a weight
    :return: the model with sparsity introduced
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data = param.mul_(torch.rand_like(param) > sparsity).float()
    return model


def _make_dense(model):
    """
    Makes a model dense by setting all weights to 1

    :param model: the model to make dense
    :return: the model with all dense params
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name:
                param.data = torch.ones_like(param.data).float()
    return model


def _test_models():
    return [
        _induce_sparsity(LinearNet()),
        _induce_sparsity(ConvNet()),
    ]


def _test_optims():
    return [
        torch.optim.Adam,
        torch.optim.SGD,
    ]


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.parametrize("model", _test_models())
@pytest.mark.parametrize("optimizer", _test_optims())
def test_constant_pruning_modifier_e2e(model, optimizer):
    expected_sparsities = {
        name: tensor_sparsity(param.data)
        for name, param in model.named_parameters()
        if "weight" in name
    }

    # init modifier with model

    state = State()
    state.update(
        model=model,
        optimizer=optimizer(model.parameters(), lr=0.1),
    )
    modifier = ConstantPruningModifier(
        targets="__ALL_PRUNABLE__",
        start=0,
        end=1,
        update=0.5,
    )
    modifier.initialize(state)

    # check mask is added and has correct sparsity

    for _, parameterized_layer in modifier.parameterized_layers_.items():
        mask_name = param_mask_name()
        mask_tensor = parameterized_layer.layer.get_buffer(mask_name)
        data_tensor = parameterized_layer.param.data
        # check mask and data tensors have 0 in the same places
        assert torch.all(mask_tensor == (data_tensor != 0))

    # mess up model sparsity

    model = _make_dense(model)
    manipulated_sparsities = {
        name: tensor_sparsity(param.data)
        for name, param in model.named_parameters()
        if "weight" in name
    }
    assert manipulated_sparsities != expected_sparsities, "Sparsity manipulation failed"

    # apply modifier

    modifier.on_update(state, event=Event(type_=EventType.OPTIM_PRE_STEP))
    modifier.on_update(state, event=Event(type_=EventType.OPTIM_POST_STEP))
    modifier.on_end(state, None)

    # copy old mask settings as finalize will remove them
    #  this is needed to check if a mask was persistent

    old_mask_settings = modifier._mask_settings.copy()
    modifier.finalize(state)

    # check mask is removed
    for layer_param_name, parameterized_layer in modifier.parameterized_layers_.items():
        mask_name = param_mask_name()

        if not old_mask_settings[layer_param_name].persistent:
            assert not hasattr(parameterized_layer.layer, mask_name)

        # mask name should not be in _mask_settings or
        #  _masked_layer_params
        assert layer_param_name not in modifier._mask_settings
        assert layer_param_name not in modifier._masked_layer_params

    # sparsity should restored by ConstantPruningModifier

    actual_sparsities = {
        name: tensor_sparsity(param.data)
        for name, param in model.named_parameters()
        if "weight" in name
    }
    assert actual_sparsities == expected_sparsities, "Sparsity was not constant"


@pytest.mark.skipif(
    os.getenv("NM_ML_SKIP_PYTORCH_TESTS", False),
    reason="Skipping pytorch tests",
)
@pytest.mark.usefixtures("setup_modifier_factory")
def test_constant_pruning_pytorch_is_registered():
    kwargs = dict(
        start=5.0,
        end=15.0,
        targets="__ALL_PRUNABLE__",
    )
    type_ = ModifierFactory.create(
        type_="ConstantPruningModifier",
        allow_experimental=False,
        allow_registered=True,
        **kwargs,
    )

    assert isinstance(
        type_, ConstantPruningModifier
    ), "PyTorch ConstantPruningModifier not registered"
