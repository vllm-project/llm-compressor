import importlib
import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

import pytest
from datasets import IterableDataset


@pytest.fixture
def qad(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "examples/qad"
    monkeypatch.syspath_prepend(str(path))
    return importlib.import_module("modelopt_mix")


class ChatTokenizer:
    def apply_chat_template(self, messages, tokenize):
        assert tokenize is False
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        assert kwargs == dict(
            truncation=True, max_length=16, padding=False, add_special_tokens=False
        )
        return {"input_ids": [int(text)], "attention_mask": [1]}


def test_parquet_optional_tool_fields_do_not_trigger_llama_tool_template(qad):
    from copy import deepcopy

    from jinja2 import Template

    call = {"type": "function", "function": {"name": "python", "arguments": {}}}
    messages = [
        {"role": "user", "content": "question", "tool_calls": [], "name": None},
        {"role": "assistant", "content": "answer", "tool_calls": None},
        {"role": "assistant", "tool_calls": [call]},
    ]
    original = deepcopy(messages)
    normalized = qad._normalize_messages(messages)
    # Match the key-presence branch used by the actual Llama 3.1 template.
    template = Template(
        "{% for m in messages %}{% if 'tool_calls' in m %}"
        "{{ m.tool_calls | length }}{% else %}{{ m.content }}{% endif %}|{% endfor %}"
    )
    assert template.render(messages=normalized) == "question|answer|1|"
    assert normalized[2]["tool_calls"] == [call]
    assert messages == original


@pytest.fixture
def loader(monkeypatch, qad):
    def load(source):
        index = qad.MODELOPT_QAD_SOURCES.index(source)

        def generate():
            for i in range(1000):
                yield {"messages": [{"role": "user", "content": str(index * 1000 + i)}]}

        return IterableDataset.from_generator(generate)

    mock = Mock(side_effect=load)
    monkeypatch.setattr(qad, "_stream_source", mock)
    return mock


@pytest.mark.parametrize("count", [1, 2, 7, 512, 1024, 19000, 20000])
def test_allocation_preserves_total_and_ratios(qad, count):
    counts = qad.modelopt_qad_sample_counts(count)
    assert sum(counts) == count
    assert all(n >= 0 for n in counts)
    for source, n in zip(qad.MODELOPT_QAD_SOURCES, counts):
        assert abs(n - count * source.weight / 19000) < 1
    if count == 1024:
        assert counts == [323, 135, 81, 81, 269, 81, 54]


def test_streaming_mix_counts_reproducibility_and_provenance(qad, loader):
    def build(seed):
        return qad.load_modelopt_qad_mix(
            ChatTokenizer(), 1024, 16, seed=seed, shuffle_buffer=8
        )

    data = build(42)
    assert len(data) == 1024
    assert Counter(row["input_ids"][0] // 1000 for row in data) == dict(
        enumerate([323, 135, 81, 81, 269, 81, 54])
    )
    assert loader.call_count == 7
    assert data.to_dict() == build(42).to_dict()
    assert data.to_dict() != build(43).to_dict()
    info = json.loads(data.info.description)
    assert info["config_url"] == qad.MODELOPT_BLEND_URL
    assert sum(s["samples"] for s in info["sources"]) == 1024
    assert all(len(s["revision"]) == 40 for s in info["sources"])
    assert set(data.column_names) == {"input_ids", "attention_mask"}


def test_small_mix_does_not_load_zero_count_sources(qad, loader):
    assert len(qad.load_modelopt_qad_mix(ChatTokenizer(), 1, 16)) == 1
    assert loader.call_count == 1


def test_json_source_uses_pinned_streaming_split(qad, monkeypatch):
    load = Mock()
    monkeypatch.setattr(qad, "load_dataset", load)
    source = qad.MODELOPT_QAD_SOURCES[0]
    assert qad._stream_source(source) is load.return_value
    load.assert_called_once_with(
        source.dataset, split=source.split, revision=source.revision, streaming=True
    )


def test_parquet_source_streams_pinned_files_and_can_stop_early(
    qad, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "data.parquet"
    rows = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(100)]
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=50)
    builder = Mock(
        return_value=SimpleNamespace(
            config=SimpleNamespace(data_files={"medium": [str(path)]})
        )
    )
    monkeypatch.setattr(qad, "load_dataset_builder", builder)
    source = qad.MODELOPT_QAD_SOURCES[1]
    stream = qad._stream_source(source)
    assert list(stream.take(3)) == rows[:3]
    assert list(stream) == rows
    builder.assert_called_once_with(source.dataset, revision=source.revision)
    iterator = qad._iter_parquet_rows([str(path)])
    assert next(iterator) == rows[0]
    iterator.close()


@pytest.mark.parametrize("failure", ["access", "short", "schema"])
def test_source_failure_is_not_silently_skipped(qad, monkeypatch, failure):
    def load(*args, **kwargs):
        if failure == "access":
            raise PermissionError("GatedRepo")
        return IterableDataset.from_generator(
            lambda: iter([] if failure == "short" else [{"messages": []}])
        )

    monkeypatch.setattr(qad, "_stream_source", load)
    with pytest.raises(RuntimeError, match="No sources are skipped") as error:
        qad.load_modelopt_qad_mix(ChatTokenizer(), 7, 16)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_samples": 0},
        {"num_samples": -1},
        {"max_seq_length": 0},
        {"shuffle_buffer": 0},
    ],
)
def test_invalid_limits_fail_before_network(qad, loader, kwargs):
    with pytest.raises(ValueError):
        qad.load_modelopt_qad_mix(ChatTokenizer(), **kwargs)
    loader.assert_not_called()


def test_example_routes_mix_to_shared_dataset(qad, loader):
    example = importlib.import_module("llama3_example")
    args = example.parse_args(
        [
            "--output",
            "unused",
            "--dataset",
            "modelopt_qad_mix",
            "--samples",
            "1024",
            "--max-seq-length",
            "16",
        ]
    )
    data = example.prepare_dataset(
        ChatTokenizer(),
        args.dataset,
        args.split,
        args.samples,
        args.max_seq_length,
        args.text_column,
    )
    assert len(data) == 1024
    assert loader.call_count == 7
    assert Counter(row["input_ids"][0] // 1000 for row in data) == dict(
        enumerate([323, 135, 81, 81, 269, 81, 54])
    )
