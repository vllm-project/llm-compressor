import unittest

import pytest
from compressed_tensors.quantization import QuantizationScheme
from parameterized import parameterized

from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.quantization.gptq import GPTQModifier
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
from llmcompressor.utils.pytorch.module import qat_active
from tests.llmcompressor.modifiers.conf import (
    LifecyleTestingHarness,
    setup_modifier_factory,
)
from tests.llmcompressor.pytorch.helpers import LinearNet
from tests.testing_utils import requires_torch


@pytest.mark.unit
@requires_torch
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
        kwargs = dict(
            sparsity=sparsity,
            block_size=128,
            targets=targets,
        )
        modifier = SparseGPTModifier(**kwargs)
        testing_harness = LifecyleTestingHarness(model=LinearNet(), start=-1)

        # confirm invalid layerwise recipes fail at initialization
        with self.assertRaises(ValueError):
            modifier.initialize(testing_harness.get_state())


@pytest.mark.unit
@requires_torch
class TestSuccessfulLayerwiseRecipe(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    def test_successful_layerwise_recipe(self):
        sparsities = [0.5, 0.2]
        targets = ["seq.fc1", "seq.fc2"]
        kwargs = dict(sparsity=sparsities, block_size=128, targets=targets)
        modifier = SparseGPTModifier(**kwargs)
        modifier.compressible_layers_ = {"seq.fc1": None, "seq.fc2": None}
        modifier.model = LinearNet()
        found_compressible_layers = modifier.compressible_layers()
        modifier.compressible_layers_ = found_compressible_layers
        modifier._validate_layerwise_sparsity()

        # ensure layers names successfully match up with model
        self.assertEqual(len(found_compressible_layers), len(targets))


@pytest.mark.unit
@requires_torch
class TestCreateDefaultQuantModifier(unittest.TestCase):
    def setUp(self):
        setup_modifier_factory()

    def test_create_default_quant_modifier(self):
        kwargs = dict(block_size=128)

        modifier = GPTQModifier(**kwargs)
        assert modifier.quantization_modifier_ is None

        testing_harness = LifecyleTestingHarness(model=LinearNet())
        modifier.on_initialize_structure(testing_harness.get_state())
        assert modifier.quantize
        assert isinstance(modifier.quantization_modifier_, QuantizationModifier)
        modifier.quantization_modifier_.create_init_config()
        default_config_group_name = "group_0"
        should_be_default_quant_scheme = modifier.quantization_modifier_.config_groups[
            default_config_group_name
        ]
        self.assertEqual(should_be_default_quant_scheme.input_activations.num_bits, 8)
        # input activations are symmetric by default in QuantizationModifier
        assert should_be_default_quant_scheme.input_activations.symmetric

        self.assertEqual(should_be_default_quant_scheme.weights.num_bits, 8)
        assert should_be_default_quant_scheme.weights.symmetric


@pytest.mark.unit
@requires_torch
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

        kwargs = dict(block_size=128)
        modifier = GPTQModifier(**kwargs)
        assert not modifier.quantization_modifier_

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
                        "strategy": "tensor",
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
        kwargs = dict(block_size=128, quantize=self.quant_config)

        modifier = GPTQModifier(**kwargs)
        assert modifier.quantization_modifier_ is None

        testing_harness = LifecyleTestingHarness(model=LinearNet())
        modifier.on_initialize_structure(testing_harness.get_state())
        assert modifier.quantize
        self.assertIsInstance(modifier.quantization_modifier_, QuantizationModifier)

        dict_scheme = dict(modifier.quantization_modifier_.config_groups)
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
