import unittest

import pytest
from compressed_tensors.quantization import QuantizationScheme
from parameterized import parameterized

from llmcompressor.modifiers.pruning import SparseGPTModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
from llmcompressor.utils.pytorch.module import qat_active
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

        model = testing_harness.state.model
        num_hooks = len(modifier._hooks)
        num_found = sum(len(module._forward_hooks) > 0 for module in model.modules())
        self.assertEqual(num_hooks, num_found)


@pytest.mark.unit
class TestCreateDefaultQuantModifier(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    def test_create_default_quant_modifier(self):
        modifier = GPTQModifier(block_size=128)
        assert modifier._quantization_modifier is None

        testing_harness = LifecyleTestingHarness(model=LinearNet())
        modifier.on_initialize_structure(testing_harness.get_state())
        assert modifier.quantize
        assert isinstance(modifier._quantization_modifier, QuantizationModifier)
        modifier._quantization_modifier.create_init_config()
        default_config_group_name = "group_0"
        should_be_default_quant_scheme = modifier._quantization_modifier.config_groups[
            default_config_group_name
        ]
        assert should_be_default_quant_scheme.input_activations is None
        assert should_be_default_quant_scheme.weights is None


@pytest.mark.unit
class TestSetQuantIfModifierAlreadyExists(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    def test_set_quant_if_modifer_already_exists(self):
        model = LinearNet()
        scheme = QuantizationScheme(
            targets=["Linear"],
            input_activations=dict(num_bits=8, symmetric=True),
            weights=dict(num_bits=4, symmetric=False),
        )

        modifier = QuantizationModifier(config_groups={"group_0": scheme})
        testing_harness = LifecyleTestingHarness(model=model, start=-1)

        assert not qat_active(testing_harness.get_state().model)
        modifier.initialize(testing_harness.get_state())
        assert qat_active(testing_harness.get_state().model)

        modifier = GPTQModifier(block_size=128)
        assert not modifier._quantization_modifier

        modifier.on_initialize_structure(testing_harness.get_state())
        # since quantization modifier is already applied, quantization must be set in
        # GPTQ
        assert modifier.quantize


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
                        "dynamic": "true",
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
        self.quant_config = {"QuantizationModifier": self.quant_kwargs}

    def test_set_quant_in_gptq(self):
        modifier = GPTQModifier(block_size=128, quantize=self.quant_config)
        assert modifier._quantization_modifier is None

        testing_harness = LifecyleTestingHarness(model=LinearNet())
        modifier.on_initialize_structure(testing_harness.get_state())
        assert modifier.quantize
        self.assertIsInstance(modifier._quantization_modifier, QuantizationModifier)

        dict_scheme = dict(modifier._quantization_modifier.config_groups)
        self._check_config(
            dict(dict_scheme["config_group_0"].weights),
            self.quant_kwargs["config_groups"]["config_group_0"]["weights"],
        )
        self._check_config(
            dict(dict_scheme["config_group_0"].input_activations),
            self.quant_kwargs["config_groups"]["config_group_0"]["input_activations"],
        )

    def _check_config(self, actual, expected):
        self.assertEqual(actual["num_bits"], expected["num_bits"])
        self.assertEqual(actual["symmetric"], expected["symmetric"])
        self.assertEqual(actual["strategy"], expected["strategy"])
