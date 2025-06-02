import unittest

import pytest
import torch
from parameterized import parameterized

from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from tests.llmcompressor.modifiers.conf import (
    LifecyleTestingHarness,
    setup_modifier_factory,
)
from tests.llmcompressor.pytorch.helpers import LinearNet


@pytest.mark.unit
class TestInvalidLayerwiseRecipesRaiseExceptions(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    @parameterized.expand(
        [
            [[0.5, 0.2], "__ALL__"],
            [[0.2, 0.1, 0.3], ["seq.fc1", "seq.fc2"]],
            [[0.3, 0.4], ["re:.*fc1", "re:.*fc2"]],
        ]
    )
    def test_invalid_layerwise_recipes_raise_exceptions(self, sparsity, targets):
        setup_modifier_factory()
        modifier = SparseGPTModifier(
            sparsity=sparsity,
            block_size=128,
            targets=targets,
        )
        testing_harness = LifecyleTestingHarness(model=LinearNet(), start=-1)

        # confirm invalid layerwise recipes fail at initialization
        with self.assertRaises(ValueError):
            modifier.initialize(testing_harness.get_state())


@pytest.mark.unit
class TestSuccessfulLayerwiseRecipe(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    def test_successful_layerwise_recipe(self):
        sparsities = [0.5, 0.2]
        targets = ["seq.fc1", "seq.fc2"]
        modifier = SparseGPTModifier(
            sparsity=sparsities, block_size=128, targets=targets
        )
        testing_harness = LifecyleTestingHarness(model=LinearNet(), start=-1)
        modifier.initialize(testing_harness.get_state())
        modifier.on_start(testing_harness.get_state(), None)

        model = testing_harness.state.model
        num_hooks = len(modifier._hooks)
        num_found = sum(len(module._forward_hooks) > 0 for module in model.modules())
        self.assertEqual(num_hooks, num_found)


@pytest.mark.unit
class TestApplyQuantization(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    def test_create_default_quant_modifier(self):
        modifier = GPTQModifier(block_size=128, targets=["Linear"], scheme="FP8")

        testing_harness = LifecyleTestingHarness(model=LinearNet(), start=-1)
        modifier.initialize(testing_harness.get_state())
        modifier.on_start(testing_harness.get_state(), None)

        model = testing_harness.state.model
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                assert hasattr(module, "quantization_scheme")
                assert hasattr(module, "input_observer")
                assert hasattr(module, "weight_observer")
                pre_hooks = list(module._forward_pre_hooks.values())
                post_hooks = list(module._forward_hooks.values())
                assert pre_hooks[0].__name__ == "calibrate_input_hook"
                assert post_hooks[0].__name__ == "calibrate_module"


class TestSetQuantInGPTQ(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()
        self.quant_kwargs = {
            "config_groups": {
                "config_group_0": {
                    "targets": ["Linear"],
                    "input_activations": {
                        "num_bits": 8,
                        "symmetric": False,
                        "strategy": "token",
                        "dynamic": True,
                        "kwargs": {},
                    },
                    "weights": {
                        "num_bits": 4,
                        "symmetric": True,
                        "strategy": "channel",
                        "kwargs": {},
                    },
                }
            }
        }

    def test_set_quant_in_gptq(self):
        modifier = GPTQModifier(block_size=128, **self.quant_kwargs)
        config = modifier.resolve_quantization_config()

        self._check_config(
            dict(config.config_groups["config_group_0"].weights),
            self.quant_kwargs["config_groups"]["config_group_0"]["weights"],
        )
        self._check_config(
            dict(config.config_groups["config_group_0"].input_activations),
            self.quant_kwargs["config_groups"]["config_group_0"]["input_activations"],
        )

    def _check_config(self, actual, expected):
        self.assertEqual(actual["num_bits"], expected["num_bits"])
        self.assertEqual(actual["symmetric"], expected["symmetric"])
        self.assertEqual(actual["strategy"], expected["strategy"])
