import pytest
import torch

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

    @pytest.mark.unit
    @pytest.mark.parametrize("max_seq_length", [None, 4])
    def test_truncates_other_token_aligned_features(self, max_seq_length):
        # e.g. `token_type_ids` from a multimodal processor, or `position_ids`
        # in a pre-tokenized dataset
        features = _make_features([6, 9])
        for feature in features:
            length = len(feature["input_ids"])
            feature["token_type_ids"] = [0] * length
            feature["position_ids"] = torch.arange(length)

        result = DataCollatorWithTruncation(max_seq_length)(features)

        expected_len = 6 if max_seq_length is None else max_seq_length
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            assert result[key].shape == (2, expected_len), key
        assert result["position_ids"][1].tolist() == list(range(expected_len))

    @pytest.mark.unit
    def test_does_not_truncate_non_token_features(self):
        features = _make_features([6, 9])
        for feature in features:
            feature["pixel_values"] = torch.ones(3, 2, 2)
            feature["image_grid_thw"] = [[1, 2, 2]]

        result = DataCollatorWithTruncation()(features)

        assert result["input_ids"].shape == (2, 6)
        assert result["pixel_values"].shape == (2, 3, 2, 2)
        assert result["image_grid_thw"].shape == (2, 1, 3)

    @pytest.mark.unit
    def test_truncates_known_keys_without_input_ids(self):
        features = [{"labels": [1] * 6}, {"labels": [1] * 9}]

        result = DataCollatorWithTruncation(max_seq_length=4)(features)

        assert result["labels"].shape == (2, 4)
