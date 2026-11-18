import io
from collections import Counter

import pytest
from datasets import Dataset
from loguru import logger

from llmcompressor.args import DatasetArguments
from llmcompressor.transformers.data import RazorCalDataset, TextGenerationDataset
from llmcompressor.transformers.data.razorcal import _clean_messages, _clean_tools

RAZORCAL_SAMPLES = 2048
RAZORCAL_DOMAINS = {
    "Coding": 512,
    "Math": 256,
    "Science (STEM)": 256,
    "Chinese-STEM": 256,
    "Instruction Following": 256,
    "Tool Calling": 256,
    "World Knowledge": 256,
}


def _load_manager(tiny_llama_tokenizer, split="train"):
    dataset_args = DatasetArguments(dataset="razorcal")
    return TextGenerationDataset.load_from_registry(
        "razorcal",
        dataset_args=dataset_args,
        split=split,
        processor=tiny_llama_tokenizer,
    )


@pytest.mark.unit
def test_razorcal_initializes(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(dataset="razorcal", max_seq_length=128)
    manager = TextGenerationDataset.load_from_registry(
        dataset_args.dataset,
        dataset_args=dataset_args,
        split=None,
        processor=tiny_llama_tokenizer,
    )
    assert isinstance(manager, RazorCalDataset)
    assert manager.dataset_args.dataset == "inference-optimization/RazorCal"
    assert manager.dataset_args.text_column == "messages"
    assert not manager.padding
    assert manager.max_seq_length == dataset_args.max_seq_length
    # the corpus is chat data, so the tokenizer must not add special tokens on
    # top of the ones rendered by the chat template
    assert manager.add_special_tokens is False


@pytest.mark.unit
def test_razorcal_sets_default_chat_template(tiny_llama_tokenizer):
    assert tiny_llama_tokenizer.chat_template is None

    _load_manager(tiny_llama_tokenizer)

    assert tiny_llama_tokenizer.chat_template == RazorCalDataset.DEFAULT_CHAT_TEMPLATE


@pytest.mark.unit
def test_razorcal_loads_one_row_per_sample(tiny_llama_tokenizer):
    """The corpus defines a single `train` split holding one row per sample."""
    manager = _load_manager(tiny_llama_tokenizer)

    dataset = manager.load_dataset()

    assert isinstance(dataset, Dataset)
    assert len(dataset) == RAZORCAL_SAMPLES
    assert set(dataset.column_names) == {
        "domain",
        "sub_source",
        "coding_subclass",
        "messages",
        "tools",
    }
    assert dict(Counter(dataset["domain"])) == RAZORCAL_DOMAINS

    sample = dataset[0]
    assert sample["messages"]
    assert all(
        "role" in message and "content" in message for message in sample["messages"]
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "split, expected",
    [
        ("train", RAZORCAL_SAMPLES),
        ("train[:512]", 512),
        ("train[100:356]", 256),
        ("train[-100:]", 100),
        ("train[:3000]", RAZORCAL_SAMPLES),  # clamped like datasets.load_dataset
        ("train[:25%]", 512),
        ("train[50%:75%]", 512),
    ],
)
def test_razorcal_split_slicing(tiny_llama_tokenizer, split, expected):
    manager = _load_manager(tiny_llama_tokenizer, split=split)

    dataset = manager.load_dataset()

    assert len(dataset) == expected


@pytest.mark.unit
def test_razorcal_decodes_json_payloads():
    """Tool call arguments and tool parameters are recorded as JSON strings
    and must be decoded back into objects, with undecodable values kept."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {
                        "name": "get_balance",
                        "arguments": '{"address": "0x00"}',
                    },
                },
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "broken",
                        "arguments": "{not json",
                    },
                },
            ],
        },
        {"role": "tool", "content": "result", "tool_calls": None},
    ]

    cleaned = _clean_messages(messages)

    assert cleaned[0]["tool_calls"][0]["function"]["arguments"] == {"address": "0x00"}
    # undecodable arguments are left unchanged
    assert cleaned[0]["tool_calls"][1]["function"]["arguments"] == "{not json"
    # empty optional fields are dropped
    assert "tool_calls" not in cleaned[1]
    assert "content" in cleaned[1]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_balance",
                "description": "query a balance",
                "parameters": '{"type": "object"}',
            },
        }
    ]
    cleaned_tools = _clean_tools(tools)
    assert cleaned_tools[0]["function"]["parameters"] == {"type": "object"}
    assert _clean_tools(None) is None
    assert _clean_tools([]) is None


@pytest.mark.unit
def test_razorcal_template_renders_messages(tiny_llama_tokenizer):
    manager = _load_manager(tiny_llama_tokenizer)

    raw_dataset = manager.load_dataset()
    result = manager.dataset_template(raw_dataset[0])

    assert set(result.keys()) == {"text"}
    # the default chat template renders the content of every message role,
    # including tool messages
    for message in raw_dataset[0]["messages"]:
        if message["content"]:
            assert message["content"] in result["text"]


@pytest.mark.unit
def test_razorcal_template_renders_tool_messages(tiny_llama_tokenizer):
    manager = _load_manager(tiny_llama_tokenizer)

    raw_dataset = manager.load_dataset()
    sample = next(
        sample
        for sample in raw_dataset
        if any(message["role"] == "tool" for message in sample["messages"])
    )

    result = manager.dataset_template(sample)

    for message in sample["messages"]:
        if message["content"]:
            assert message["content"] in result["text"]


@pytest.mark.unit
def test_razorcal_forwards_tools_to_chat_template(tiny_llama_tokenizer):
    tiny_llama_tokenizer.chat_template = (
        "{% if tools %}{{ tools | length }}{% else %}no-tools{% endif %}"
    )
    manager = _load_manager(tiny_llama_tokenizer)

    raw_dataset = manager.load_dataset()
    tool_sample = next(sample for sample in raw_dataset if sample["tools"])
    plain_sample = next(sample for sample in raw_dataset if not sample["tools"])

    assert manager.dataset_template(tool_sample)["text"] == str(
        len(tool_sample["tools"])
    )
    assert manager.dataset_template(plain_sample)["text"] == "no-tools"


@pytest.mark.unit
def test_razorcal_template_fallback(tiny_llama_tokenizer):
    """Samples which fail to render fall back to their raw message contents,
    logging a single warning for the failure."""
    tiny_llama_tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] | nosuchfilter }}"
    )
    manager = _load_manager(tiny_llama_tokenizer, split="train[:2]")

    raw_dataset = manager.load_dataset()
    log_output = io.StringIO()
    handler_id = logger.add(log_output, level="WARNING")
    try:
        first = manager.dataset_template(raw_dataset[0])
        second = manager.dataset_template(raw_dataset[1])
    finally:
        logger.remove(handler_id)

    for result, sample in zip((first, second), (raw_dataset[0], raw_dataset[1])):
        assert result["text"] == "\n".join(
            message["content"] for message in sample["messages"]
        )
    assert log_output.getvalue().count("falling back") == 1
