import unittest

import pytest
from parameterized import parameterized

from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.modifiers.pruning.alps.base import ALPSModifier
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
            targets=targets,
        )
        modifier = ALPSModifier(**kwargs)
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
        kwargs = dict(sparsity=sparsities, targets=targets)
        modifier = ALPSModifier(**kwargs)
        modifier.compressible_layers_ = {"seq.fc1": None, "seq.fc2": None}
        modifier.model = LinearNet()
        found_compressible_layers = modifier.compressible_layers()
        modifier.compressible_layers_ = found_compressible_layers
        modifier._validate_layerwise_sparsity()

        # ensure layers names successfully match up with model
        self.assertEqual(len(found_compressible_layers), len(targets))


@pytest.mark.unit
class TestALPSIsRegistered(unittest.TestCase):
    def setUp(self):
        self.kwargs = dict(
            sparsity=0.5,
            targets="__ALL_PRUNABLE__",
        )
        setup_modifier_factory()

    def test_alps_is_registered(self):
        type_ = ModifierFactory.create(
            type_="ALPSModifier",
            allow_experimental=False,
            allow_registered=True,
            **self.kwargs,
        )

        self.assertIsInstance(
            type_,
            ALPSModifier,
            "PyTorch ALPSModifier not registered",
        )
