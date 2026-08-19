import pytest
from datasets import Dataset

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets.utils import (
    SEQ_LEN_REPORT_THRESHOLD,
    DataCollatorWithTruncation,
    untruncated_sequence_summary,
)


def _make_features(lengths: list[int]) -> list[dict[str, list[int]]]:
    return [
        {"input_ids": [1] * length, "attention_mask": [1] * length}
        for length in lengths
    ]


def _make_tokenized_dataset(
    lengths: list[int], feature_name: str = "input_ids"
) -> Dataset:
    return Dataset.from_dict(
        {
            feature_name: [[0] * length for length in lengths],
            "attention_mask": [[1] * length for length in lengths],
        }
    )


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


class TestUntruncatedSequenceSummary:
    """Tests for the long-sample summary used in OOM error messages (#3011)."""

    @pytest.mark.unit
    def test_summary_when_max_seq_length_unset_and_samples_long(self):
        args = DatasetArguments(num_calibration_samples=2)
        longest = SEQ_LEN_REPORT_THRESHOLD + 100
        dataset = _make_tokenized_dataset([16, longest])

        summary = untruncated_sequence_summary(args, dataset)

        assert summary is not None
        assert "`max_seq_length` is not set" in summary
        assert f"1 sample(s) longer than {SEQ_LEN_REPORT_THRESHOLD} tokens" in summary
        assert f"longest is {longest} tokens" in summary

    @pytest.mark.unit
    def test_summary_for_decoder_input_ids(self):
        args = DatasetArguments(num_calibration_samples=2)
        longest = SEQ_LEN_REPORT_THRESHOLD + 1
        dataset = _make_tokenized_dataset(
            [16, longest], feature_name="decoder_input_ids"
        )

        summary = untruncated_sequence_summary(args, dataset)

        assert summary is not None
        assert f"longest is {longest} tokens" in summary

    @pytest.mark.unit
    def test_summary_respects_dataset_indices_mapping(self):
        # select() keeps only short samples; the arrow indices mapping must be
        # applied so removed long samples are not reported
        args = DatasetArguments(num_calibration_samples=2)
        dataset = _make_tokenized_dataset([16, SEQ_LEN_REPORT_THRESHOLD + 100, 32])
        dataset = dataset.select([0, 2])

        assert untruncated_sequence_summary(args, dataset) is None

    @pytest.mark.unit
    def test_none_when_samples_at_or_below_threshold(self):
        args = DatasetArguments(num_calibration_samples=2)
        dataset = _make_tokenized_dataset([16, SEQ_LEN_REPORT_THRESHOLD])

        assert untruncated_sequence_summary(args, dataset) is None

    @pytest.mark.unit
    def test_none_when_max_seq_length_set(self):
        args = DatasetArguments(
            num_calibration_samples=2, max_seq_length=SEQ_LEN_REPORT_THRESHOLD
        )
        dataset = _make_tokenized_dataset([16, SEQ_LEN_REPORT_THRESHOLD + 100])

        assert untruncated_sequence_summary(args, dataset) is None

    @pytest.mark.unit
    def test_none_without_input_ids_column(self):
        args = DatasetArguments(num_calibration_samples=2)
        dataset = Dataset.from_dict({"pixel_values": [[0.0], [1.0]]})

        assert untruncated_sequence_summary(args, dataset) is None
