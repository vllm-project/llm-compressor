import pytest
from transformers import AutoTokenizer

from llmcompressor.args import ModelArguments


@pytest.fixture
def tiny_llama_path():
    return "nm-testing/tinysmokellama-3.2"


@pytest.fixture
def tiny_llama_model_args(tiny_llama_path):
    return ModelArguments(model=tiny_llama_path)


@pytest.fixture
def tiny_llama_tokenizer(tiny_llama_model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        tiny_llama_model_args.model,
        use_fast=True,
        revision=tiny_llama_model_args.model_revision,
    )
    return tokenizer
