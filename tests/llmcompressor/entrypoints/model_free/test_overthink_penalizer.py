import json
import math

import pytest
import torch
from compressed_tensors.entrypoints.convert import convert_checkpoint
from safetensors.torch import load_file
from tokenizers import Tokenizer, models
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Qwen3Config,
)

from llmcompressor.entrypoints.model_free import OverthinkPenalizer

LAM = 1.5
SCALE = math.exp(-LAM)
OVERTHINK_TOKENS = [" perhaps", " maybe", " wait", "However", " -"]

_COMMON = dict(
    vocab_size=64,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)


def _tiny_configs():
    return [
        pytest.param(
            LlamaConfig(**_COMMON, tie_word_embeddings=False), id="llama-untied"
        ),
        pytest.param(
            Qwen3Config(**_COMMON, head_dim=32, tie_word_embeddings=True),
            id="qwen3-tied",
        ),
    ]


def _save_tiny_tokenizer(path, tokens):
    vocab = {"[UNK]": 0}
    for token in tokens:
        vocab[token] = len(vocab)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(path)


def _save_tiny_model(config, path) -> torch.nn.Module:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    model.eval()
    model.save_pretrained(path)
    _save_tiny_tokenizer(path, OVERTHINK_TOKENS)
    return model


def _write_tokens_file(path, tokens) -> str:
    tokens_path = path / "overthink_tokens.json"
    with open(tokens_path, "w") as file:
        json.dump(tokens, file)
    return tokens_path


@pytest.mark.parametrize("config", _tiny_configs())
def test_overthink_penalty(config, tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    model = _save_tiny_model(config, source)

    # "_wait" repeats " wait" through the paper's leading-space convention and
    # " not_a_token" is not in the vocabulary: both are skipped
    tokens_path = _write_tokens_file(
        tmp_path, OVERTHINK_TOKENS + ["_wait", " not_a_token"]
    )

    converter = OverthinkPenalizer.from_pretrained(source, tokens_path, lam=LAM)
    tokenizer = AutoTokenizer.from_pretrained(source)
    expected_ids = [tokenizer.convert_tokens_to_ids(t) for t in OVERTHINK_TOKENS]
    assert sorted(converter.token_ids) == sorted(expected_ids)

    convert_checkpoint(source, target, converter, device="cpu")

    original = load_file(source / "model.safetensors")
    converted = load_file(target / "model.safetensors")

    # the penalty scales down the lm_head weights of the tokens, and input
    # embeddings are left untouched, so only output logits are penalized
    if config.tie_word_embeddings:
        embedding = original["model.embed_tokens.weight"]
        expected_lm_head = embedding.clone()
    else:
        embedding = original["model.embed_tokens.weight"]
        expected_lm_head = original["lm_head.weight"].clone()
    expected_lm_head[converter.token_ids] *= SCALE
    assert torch.equal(converted["lm_head.weight"], expected_lm_head)
    assert torch.equal(converted["model.embed_tokens.weight"], embedding)

    # every other tensor is untouched
    for name, tensor in original.items():
        if name != "lm_head.weight":
            assert torch.equal(converted[name], tensor), name

    # tied checkpoints are untied so the penalized lm_head is loaded
    with open(target / "config.json") as file:
        assert not json.load(file)["tie_word_embeddings"]

    # the converted checkpoint loads and its logits reflect the penalty
    reloaded = AutoModelForCausalLM.from_pretrained(target, dtype=torch.float32)
    reloaded.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        expected = model(input_ids).logits
        actual = reloaded(input_ids).logits
    assert not torch.allclose(actual, expected)


def test_overthink_penalty_process_and_config():
    weight = torch.randn(16, 8)
    tensors = {"lm_head.weight": weight.clone(), "model.norm.weight": torch.ones(8)}

    converter = OverthinkPenalizer(token_ids=[2, 7], lam=2.0)
    output = converter.process(tensors)
    expected = weight.clone()
    expected[[2, 7]] *= math.exp(-2.0)
    assert torch.equal(output["lm_head.weight"], expected)
    assert torch.equal(output["model.norm.weight"], tensors["model.norm.weight"])
    assert converter.update_config(None) is None
    assert converter.get_dependencies("lm_head.weight") == set()

    with pytest.raises(ValueError, match="quantized"):
        OverthinkPenalizer(token_ids=[2], lam=2.0).process(
            {"lm_head.weight": torch.zeros(16, 8, dtype=torch.uint8)}
        )


def test_overthink_penalty_no_known_tokens(tmp_path):
    source = tmp_path / "source"
    _save_tiny_model(LlamaConfig(**_COMMON, tie_word_embeddings=False), source)
    tokens_path = _write_tokens_file(tmp_path, [" not_a_token"])

    with pytest.raises(ValueError, match="No tokens"):
        OverthinkPenalizer.from_pretrained(source, tokens_path, lam=LAM)
