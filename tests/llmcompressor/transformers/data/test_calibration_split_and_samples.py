"""
Tests for calibration split resolution and num_calibration_samples handling.

Covers the behavior that:
  * each dataset class resolves a sensible default split when none is given
    (TextGenerationDataset.DEFAULT_SPLIT),
  * an explicitly provided split (including slices and HF "+" concatenation) is
    respected,
  * num_calibration_samples limits the data *before* tokenization rather than after,
  * a multi-split DatasetDict is collapsed to a single split for calibration.
"""

import pytest
from datasets import Dataset, DatasetDict

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets import get_processed_dataset
from llmcompressor.transformers import TextGenerationDataset


def _make_manager(processor, registry="open_platypus", split="train", **dataset_kwargs):
    """Instantiate a dataset manager without triggering any download."""
    return TextGenerationDataset.load_from_registry(
        registry,
        dataset_args=DatasetArguments(**dataset_kwargs),
        split=split,
        processor=processor,
    )


def _text_dataset(n: int) -> Dataset:
    return Dataset.from_dict({"text": [f"sample {i}" for i in range(n)]})


# --------------------------------------------------------------------------- #
# Default split resolution (network-free: only exercises __init__)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize(
    "registry,expected",
    [
        ("open_platypus", "train"),
        ("wikitext", "train"),
        ("c4", "train"),
        ("gsm8k", "train"),
        ("evolcodealpaca", "train"),
        ("cnn_dailymail", "train"),
        ("perfectblend", "train"),
        ("ultrachat_200k", "train_sft"),
        ("flickr", "test"),
        ("peoples_speech", "test"),
        ("custom", None),
    ],
)
def test_default_split_resolution(registry, expected, tiny_llama_tokenizer):
    # split=None => the class resolves its own DEFAULT_SPLIT
    manager = _make_manager(tiny_llama_tokenizer, registry=registry, split=None)
    assert manager.split == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "registry,split,expected",
    [
        # ultrachat remaps the bare train/test aliases to its *_sft splits
        ("ultrachat_200k", "train", "train_sft"),
        ("ultrachat_200k", "test", "test_sft"),
        # anything else is passed through untouched, even for datasets whose
        # default split is not "train"
        ("ultrachat_200k", "train_gen", "train_gen"),
        ("ultrachat_200k", "train_sft[:5]", "train_sft[:5]"),
        ("flickr", "test[:10]", "test[:10]"),
        ("open_platypus", "train[5%:6%]", "train[5%:6%]"),
        ("gsm8k", "train+test", "train+test"),
    ],
)
def test_explicit_split_is_respected(registry, split, expected, tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer, registry=registry, split=split)
    assert manager.split == expected


# --------------------------------------------------------------------------- #
# _limit_calibration_samples (network-free: operates on an in-memory Dataset)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_limit_selects_first_n_when_not_shuffling(tiny_llama_tokenizer):
    manager = _make_manager(
        tiny_llama_tokenizer,
        num_calibration_samples=8,
        shuffle_calibration_samples=False,
    )
    out = manager._limit_calibration_samples(_text_dataset(100))
    assert len(out) == 8
    assert out["text"] == [f"sample {i}" for i in range(8)]


@pytest.mark.unit
def test_limit_shuffles_deterministically(tiny_llama_tokenizer):
    manager = _make_manager(
        tiny_llama_tokenizer,
        num_calibration_samples=8,
        shuffle_calibration_samples=True,
    )
    dataset = _text_dataset(100)
    out = manager._limit_calibration_samples(dataset)
    expected = dataset.shuffle(seed=42).select(range(8))

    assert len(out) == 8
    # matches a fixed-seed shuffle+select and is not simply the first N
    assert out["text"] == expected["text"]
    assert out["text"] != [f"sample {i}" for i in range(8)]


@pytest.mark.unit
def test_limit_is_noop_when_num_samples_is_none(tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer, num_calibration_samples=None)
    dataset = _text_dataset(50)
    assert manager._limit_calibration_samples(dataset) is dataset


@pytest.mark.unit
def test_limit_is_noop_when_dataset_smaller_than_num_samples(tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer, num_calibration_samples=100)
    out = manager._limit_calibration_samples(_text_dataset(10))
    assert len(out) == 10


@pytest.mark.unit
def test_limit_skips_non_map_style_dataset(tiny_llama_tokenizer):
    # streaming/iterable datasets are not map-style Datasets and are left untouched
    manager = _make_manager(tiny_llama_tokenizer, num_calibration_samples=5)
    sentinel = object()
    assert manager._limit_calibration_samples(sentinel) is sentinel


# --------------------------------------------------------------------------- #
# _select_split (network-free: operates on an in-memory DatasetDict)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_select_split_passes_through_single_dataset(tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer)
    dataset = _text_dataset(3)
    assert manager._select_split(dataset) is dataset


@pytest.mark.unit
def test_select_split_returns_sole_split(tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer)
    out = manager._select_split(DatasetDict({"only": _text_dataset(3)}))
    assert len(out) == 3


@pytest.mark.unit
def test_select_split_prefers_train(tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer)
    dd = DatasetDict({"validation": _text_dataset(2), "train": _text_dataset(3)})
    assert len(manager._select_split(dd)) == 3


@pytest.mark.unit
def test_select_split_prefers_calibration_when_no_train(tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer)
    dd = DatasetDict({"validation": _text_dataset(2), "calibration": _text_dataset(4)})
    assert len(manager._select_split(dd)) == 4


@pytest.mark.unit
def test_select_split_falls_back_to_first_split(tiny_llama_tokenizer):
    manager = _make_manager(tiny_llama_tokenizer)
    dd = DatasetDict({"foo": _text_dataset(2), "bar": _text_dataset(5)})
    # neither train nor calibration present -> first inserted split
    assert len(manager._select_split(dd)) == 2


# --------------------------------------------------------------------------- #
# End-to-end (small downloads via sliced splits)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_num_samples_trims_before_tokenization(tiny_llama_tokenizer, monkeypatch):
    # Only the trimmed samples should ever be tokenized, not the whole split.
    calls = {"n": 0}
    original_tokenize = TextGenerationDataset.tokenize

    def counting_tokenize(self, data):
        calls["n"] += 1
        return original_tokenize(self, data)

    monkeypatch.setattr(TextGenerationDataset, "tokenize", counting_tokenize)

    dataset_args = DatasetArguments(
        dataset="open_platypus",
        splits="train[:20]",
        num_calibration_samples=8,
        shuffle_calibration_samples=False,
        overwrite_cache=True,  # force re-tokenization so the counter is exercised
    )
    dataset = get_processed_dataset(
        dataset_args=dataset_args, processor=tiny_llama_tokenizer
    )

    assert len(dataset) == 8
    assert calls["n"] == 8


@pytest.mark.unit
def test_explicit_slice_composes_with_num_samples(tiny_llama_tokenizer):
    dataset_args = DatasetArguments(
        dataset="open_platypus",
        splits="train[:20]",
        num_calibration_samples=8,
    )
    dataset = get_processed_dataset(
        dataset_args=dataset_args, processor=tiny_llama_tokenizer
    )
    assert len(dataset) == 8


@pytest.mark.unit
def test_concatenated_splits_are_combined_and_trimmed(tiny_llama_tokenizer):
    # HF "+" concatenation yields a single combined Dataset for calibration
    combined = get_processed_dataset(
        dataset_args=DatasetArguments(
            dataset="gsm8k",
            dataset_config_name="main",
            splits="train[:6]+test[:4]",
            num_calibration_samples=None,
        ),
        processor=tiny_llama_tokenizer,
    )
    assert len(combined) == 10

    trimmed = get_processed_dataset(
        dataset_args=DatasetArguments(
            dataset="gsm8k",
            dataset_config_name="main",
            splits="train[:6]+test[:4]",
            num_calibration_samples=5,
        ),
        processor=tiny_llama_tokenizer,
    )
    assert len(trimmed) == 5
