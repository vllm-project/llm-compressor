import pytest
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

from llmcompressor.modifiers.pruning.sparsegpt import SparseGPTModifier


@pytest.mark.integration
def test_infer_targets():
    modifier = SparseGPTModifier(sparsity=0.0)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained("nm-testing/tinysmokellama-3.2")

    inferred = modifier._infer_sequential_targets(model)
    assert inferred == ["LlamaDecoderLayer"]
