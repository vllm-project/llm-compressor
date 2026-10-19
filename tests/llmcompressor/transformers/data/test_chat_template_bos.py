"""Chat-template calibration must not prepend a second BOS token.

Llama-style chat templates already emit the BOS token. The tokenizer
post-processor prepends another when add_special_tokens stays at its default.
Inference via apply_chat_template(tokenize=True) does not. See
https://github.com/vllm-project/llm-compressor/issues/3284
"""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.data.base import TextGenerationDataset
from llmcompressor.transformers.data.perfectblend import PerfectBlendDataset
from llmcompressor.transformers.data.ultrachat_200k import UltraChatDataset

# Template emits bos_token, and the post-processor also prepends <s>.
_CHAT_TEMPLATE = (
    "{{ bos_token }} "
    "{% for message in messages %}"
    "{{ message['role'] }} {{ message['content'] }} {{ eos_token }} "
    "{% endfor %}"
)


def _bos_prepending_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "user": 3,
        "assistant": 4,
        "system": 5,
        "hi": 6,
        "ok": 7,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A",
        special_tokens=[("<s>", 1)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
    )
    tokenizer.chat_template = _CHAT_TEMPLATE
    tokenizer.model_max_length = 64
    return tokenizer


def _args(name: str) -> DatasetArguments:
    return DatasetArguments(dataset=name, max_seq_length=64)


@pytest.mark.unit
def test_perfectblend_matches_single_bos_chat_template():
    tokenizer = _bos_prepending_tokenizer()
    dataset = PerfectBlendDataset(
        _args("perfectblend"), split="train", processor=tokenizer
    )
    sample = {
        "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "ok"},
        ]
    }
    rendered = dataset.dataset_template(sample)
    rendered["prompt"] = rendered["text"]
    tokenized = dataset.tokenize(rendered)

    # <s> user hi </s> assistant ok </s>
    expected = [1, 3, 6, 2, 4, 7, 2]
    assert tokenized["input_ids"] == expected
    assert tokenized["prompt"] == expected


@pytest.mark.unit
def test_ultrachat_matches_single_bos_chat_template():
    tokenizer = _bos_prepending_tokenizer()
    dataset = UltraChatDataset(
        _args("ultrachat_200k"), split="train", processor=tokenizer
    )
    sample = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    tokenized = dataset.tokenize(dataset.dataset_template(sample))

    # empty system turn is inserted, then user hi, then assistant ok
    assert tokenized["input_ids"] == [1, 5, 2, 3, 6, 2, 4, 7, 2]


@pytest.mark.unit
def test_raw_text_still_receives_one_bos():
    tokenizer = _bos_prepending_tokenizer()
    dataset = TextGenerationDataset(
        _args("wikitext"), split="train", processor=tokenizer
    )
    tokenized = dataset.tokenize({"text": "hi ok"})

    assert tokenized["input_ids"] == [1, 6, 7]
