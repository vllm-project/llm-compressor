from unittest.mock import patch

import pytest
import torch

from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.utils.helpers import DisableQuantization

from .test_base import _calibrate, _prepare


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
def test_charles_reobservation_schedule_and_next_epoch_scales(kind):
    model, _, batches, state, quant, qad = _prepare(
        kind,
        num_epochs=2,
        learning_rate=0.003,
        early_stopping_patience=3,
    )
    modules = list(model.block.modules())
    epoch_inputs, epoch_outputs = [], []
    train = qad._train_epoch
    reobserve = qad._reobserve_weights

    def train_epoch(*args):
        epoch_inputs.append(qad._snapshot_qparams())
        return train(*args)

    def observe(stage):
        reobserve(stage)
        if stage.startswith("epoch_"):
            epoch_outputs.append(qad._snapshot_qparams())

    with DisableQuantization(model):
        _calibrate(model, batches, state)
        quant.on_sequential_epoch_end(state, None, modules)
        with (
            patch.object(qad, "_train_epoch", side_effect=train_epoch),
            patch.object(
                qad,
                "_reobserve_weights",
                side_effect=observe,
            ),
        ):
            qad.on_sequential_epoch_end(state, None, modules)
    assert qad.reobservations == {
        "block": ["before_training", "epoch_1", "epoch_2", "before_materialization"]
    }
    for before, after in zip(epoch_inputs[1], epoch_outputs[0]):
        for key in before:
            torch.testing.assert_close(
                before[key].float(), after[key].float(), rtol=0, atol=0
            )
    if kind == "gptq":
        assert not quant._hessians and not quant._num_samples
    qad.on_finalize(state)


def test_best_weights_restore_matching_qparams():
    module = torch.nn.Linear(1, 1, bias=False)
    module.weight.data.zero_()
    module.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.ones(1), requires_grad=False)
    )
    qad = QADModifier(num_epochs=3, early_stopping_patience=1)
    qad._name = "test"
    qad._weight_modules = [module]
    epoch = 0

    def train(*args):
        nonlocal epoch
        epoch += 1
        with torch.no_grad():
            module.weight.fill_(epoch)
        return 1

    def reobserve(stage):
        with torch.no_grad():
            module.weight_scale.fill_(epoch + 1)

    with (
        patch.object(qad, "_evaluate", side_effect=[0.3, 0.2, 0.4]),
        patch.object(
            qad,
            "_train_epoch",
            side_effect=train,
        ),
        patch.object(qad, "_reobserve_weights", side_effect=reobserve),
    ):
        qad._train_with_validation(None, [module.weight], [module.weight], [0], [1])
    assert module.weight.item() == 1
    assert module.weight_scale.item() == 2  # Scale from the same winning epoch.
    assert qad.best_validation_losses == {}  # Published by the outer block loop.


def test_reobservation_requires_live_observers():
    module = torch.nn.Linear(1, 1, bias=False)
    qad = QADModifier()
    qad._weight_modules = [module]
    with pytest.raises(ValueError, match="live weight observers"):
        qad._reobserve_weights("before_training")
