import pytest

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets.utils import DataCollatorWithTruncation


def _make_features(lengths: list[int]) -> list[dict[str, list[int]]]:
    return [
        {"input_ids": [1] * length, "attention_mask": [1] * length}
        for length in lengths
    ]


class TestMaxSeqLengthDefault:
    @pytest.mark.unit
    def test_default_is_none(self):
        args = DatasetArguments()
        assert args.max_seq_length is None


class TestDataCollatorWithTruncation:
    @pytest.mark.unit
    def test_truncates_to_min_in_batch_when_no_max(self):
        collator = DataCollatorWithTruncation(max_seq_length=None)
        result = collator(_make_features([10, 20]))
        assert result["input_ids"].shape[1] == 10

    @pytest.mark.unit
    def test_truncates_to_max_seq_length_when_shorter_than_batch(self):
        collator = DataCollatorWithTruncation(max_seq_length=5)
        result = collator(_make_features([10, 20]))
        assert result["input_ids"].shape[1] == 5

    @pytest.mark.unit
    def test_truncates_to_min_when_shorter_than_max_seq_length(self):
        collator = DataCollatorWithTruncation(max_seq_length=15)
        result = collator(_make_features([10, 20]))
        assert result["input_ids"].shape[1] == 10

    @pytest.mark.unit
    def test_uniform_lengths_with_max_seq_length(self):
        collator = DataCollatorWithTruncation(max_seq_length=5)
        result = collator(_make_features([10, 10]))
        assert result["input_ids"].shape[1] == 5
        assert result["attention_mask"].shape[1] == 5

    @pytest.mark.unit
    def test_truncates_mm_token_type_ids(self):
        collator = DataCollatorWithTruncation(max_seq_length=None)
        features = [
            {
                "input_ids": [1] * length,
                "attention_mask": [1] * length,
                "mm_token_type_ids": [0] * length,
            }
            for length in (10, 20)
        ]
        result = collator(features)
        assert result["input_ids"].shape[1] == 10
        assert result["attention_mask"].shape[1] == 10
        assert result["mm_token_type_ids"].shape[1] == 10
