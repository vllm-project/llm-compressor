import pytest
from datasets import Dataset

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets.utils import (
    SEQ_LEN_ERROR_THRESHOLD,
    DataCollatorWithTruncation,
    format_calibration_data,
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


class TestUntruncatedSequenceError:
    """Tests for the long-sample error when `max_seq_length` is unset (#3011)."""

    @pytest.mark.unit
    def test_raises_when_max_seq_length_unset_and_samples_long(self):
        args = DatasetArguments(num_calibration_samples=2)
        longest = SEQ_LEN_ERROR_THRESHOLD + 100
        dataset = _make_tokenized_dataset([16, longest])

        with pytest.raises(ValueError) as excinfo:
            format_calibration_data(args, dataset, processor=None)

        message = str(excinfo.value)
        assert f"longest is {longest} tokens" in message
        assert "1 sample(s)" in message
        assert f"`max_seq_length={SEQ_LEN_ERROR_THRESHOLD}`" in message
        assert f"`max_seq_length={longest}`" in message

    @pytest.mark.unit
    def test_raises_for_decoder_input_ids(self):
        args = DatasetArguments(num_calibration_samples=2)
        longest = SEQ_LEN_ERROR_THRESHOLD + 1
        dataset = _make_tokenized_dataset(
            [16, longest], feature_name="decoder_input_ids"
        )

        with pytest.raises(ValueError) as excinfo:
            format_calibration_data(args, dataset, processor=None)

        assert f"longest is {longest} tokens" in str(excinfo.value)

    @pytest.mark.unit
    def test_no_error_when_samples_at_or_below_threshold(self):
        args = DatasetArguments(num_calibration_samples=2)
        dataset = _make_tokenized_dataset([16, SEQ_LEN_ERROR_THRESHOLD])

        format_calibration_data(args, dataset, processor=None)

    @pytest.mark.unit
    def test_no_error_when_max_seq_length_set(self):
        args = DatasetArguments(
            num_calibration_samples=2, max_seq_length=SEQ_LEN_ERROR_THRESHOLD
        )
        dataset = _make_tokenized_dataset([16, SEQ_LEN_ERROR_THRESHOLD + 100])

        format_calibration_data(args, dataset, processor=None)

    @pytest.mark.unit
    def test_no_error_without_input_ids_column(self):
        args = DatasetArguments(num_calibration_samples=2)
        dataset = Dataset.from_dict({"pixel_values": [[0.0], [1.0]]})

        format_calibration_data(args, dataset, processor=None)
