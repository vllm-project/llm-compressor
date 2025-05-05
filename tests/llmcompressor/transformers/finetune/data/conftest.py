import pytest
from transformers import AutoTokenizer

from llmcompressor.args import ModelArguments


@pytest.fixture
def tiny_llama_path():
    return "nm-testing/llama2.c-stories15M"


@pytest.fixture
def tiny_llama_model_args(tiny_llama_path):
    return ModelArguments(model=tiny_llama_path)


@pytest.fixture
def tiny_llama_tokenizer(tiny_llama_model_args):
    tokenizer = AutoTokenizer.from_pretrained(
        tiny_llama_model_args.model,
        cache_dir=tiny_llama_model_args.cache_dir,
        use_fast=True,
        revision=tiny_llama_model_args.model_revision,
        use_auth_token=True if tiny_llama_model_args.use_auth_token else None,
    )
    return tokenizer
