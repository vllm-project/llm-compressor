import unittest

import pytest

from llmcompressor.core.events import Event
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.quantization import QuantizationModifier
from tests.llmcompressor.modifiers.conf import setup_modifier_factory


@pytest.mark.unit
class TestQuantizationRegistered(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()
        self.kwargs = dict(
            index=0, group="quantization", start=2.0, end=-1.0, config_groups={}
        )

    def test_quantization_registered(self):
        quant_obj = ModifierFactory.create(
            type_="QuantizationModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(quant_obj, QuantizationModifier)


@pytest.mark.unit
class TestEndEpochs(unittest.TestCase):
    def setUp(self):
        self.start = 0.0
        self.scheme = dict(
            input_activations=dict(num_bits=8, symmetric=True),
            weights=dict(num_bits=6, symmetric=False),
        )

    def test_end_epochs(self):
        disable_quant_epoch = None
        obj_modifier = QuantizationModifier(
            start=self.start,
            scheme=self.scheme,
            disable_quantization_observer_epoch=disable_quant_epoch,
            config_groups={},
        )

        self.assertEqual(obj_modifier.calculate_disable_observer_epoch(), -1)

        for epoch in range(3):
            event = Event(steps_per_epoch=1, global_step=epoch)
            assert not obj_modifier.check_should_disable_observer(event)

        disable_quant_epoch = 3.5
        obj_modifier = QuantizationModifier(
            start=self.start,
            scheme=self.scheme,
            disable_quantization_observer_epoch=disable_quant_epoch,
            config_groups={},
        )

        self.assertEqual(
            obj_modifier.calculate_disable_observer_epoch(), disable_quant_epoch
        )

        for epoch in range(4):
            event = Event(steps_per_epoch=1, global_step=epoch)
            assert not obj_modifier.check_should_disable_observer(event)

        event = Event(steps_per_epoch=1, global_step=4)
        assert obj_modifier.check_should_disable_observer(event)

        for epoch in range(5, 8):
            event = Event(steps_per_epoch=1, global_step=epoch)
            assert obj_modifier.check_should_disable_observer(event)
