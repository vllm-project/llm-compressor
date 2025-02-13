import unittest

import pytest


@pytest.mark.integration
class TestSGPTDefaults(unittest.TestCase):
    def test_sgpt_defaults(self):
        from llmcompressor.core.state import State
        from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier

        kwargs = {"sparsity": 0.5}
        sparsegpt_modifier_only_sparsity = SparseGPTModifier(**kwargs)
        self.assertEqual(sparsegpt_modifier_only_sparsity.block_size, 128)
        self.assertEqual(sparsegpt_modifier_only_sparsity.sparsity, 0.5)

        # fail if we don't pass a sparsity or enable quantization
        kwargs = {}
        sparsegpt_invalid = SparseGPTModifier(**kwargs)
        state_test = State()
        sparsegpt_invalid.initialized_structure_ = True
        with self.assertRaises(ValueError):
            sparsegpt_invalid.on_initialize(state=state_test)
