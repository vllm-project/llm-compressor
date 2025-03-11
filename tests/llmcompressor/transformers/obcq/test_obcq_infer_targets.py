import pytest
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM

from llmcompressor.modifiers.obcq import SparseGPTModifier


@pytest.mark.integration
def test_infer_targets():
    modifier = SparseGPTModifier(sparsity=0.0)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained("nm-testing/llama2.c-stories15M")

    inferred = modifier._infer_sequential_targets(model)
    assert inferred == ["LlamaDecoderLayer"]
