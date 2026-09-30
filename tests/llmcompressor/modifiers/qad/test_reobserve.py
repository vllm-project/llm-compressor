from unittest.mock import patch

import pytest
import torch

from llmcompressor.modifiers.qad import QADModifier
from llmcompressor.utils.helpers import DisableQuantization

from .test_base import _calibrate, _prepare


def _qparams(modules):
    return [
        {
            key: value.detach().clone()
            for key in ("weight_scale", "weight_zero_point", "weight_global_scale")
            if (value := getattr(module, key, None)) is not None
        }
        for module in modules
    ]


@pytest.mark.parametrize("kind", ["rtn", "gptq"])
def test_charles_reobservation_schedule_and_next_epoch_scales(kind):
    model, _, batches, state, quant, qad = _prepare(
        kind,
        num_epochs=2,
        lr=0.003,
    )
    modules = list(model.block.modules())
    weight_modules = [model.block.left, model.block.right, model.block.down]
    stages, epoch_inputs, epoch_outputs = [], [], []
    train = qad._train_epoch
    reobserve = qad._reobserve_weights

    def train_epoch(*args):
        epoch_inputs.append(_qparams(weight_modules))
        return train(*args)

    def observe(*args):
        stages.append(args[-1])
        reobserve(*args)
        if args[-1].startswith("epoch_"):
            epoch_outputs.append(_qparams(weight_modules))

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
    assert stages == ["before_training", "epoch_1", "epoch_2", "before_materialization"]
    for before, after in zip(epoch_inputs[1], epoch_outputs[0]):
        for key in before:
            torch.testing.assert_close(
                before[key].float(), after[key].float(), rtol=0, atol=0
            )
    if kind == "gptq":
        assert not quant._hessians and not quant._num_samples
    qad.on_finalize(state)


def test_reobservation_requires_live_observers():
    module = torch.nn.Linear(1, 1, bias=False)
    qad = QADModifier()
    with pytest.raises(ValueError, match="live weight observers"):
        qad._reobserve_weights(module, [module], "before_training")
