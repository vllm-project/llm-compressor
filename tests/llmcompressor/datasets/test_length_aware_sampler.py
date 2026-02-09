from unittest.mock import patch

import pytest
from datasets import Dataset

from llmcompressor.datasets.utils import LengthAwareSampler


def _create_mock_dataset(lengths: list[int]) -> Dataset:
    """Create a mock dataset with input_ids of specified lengths."""
    return Dataset.from_dict({"input_ids": [[0] * length for length in lengths]})


class TestLengthAwareSampler:
    """Tests for LengthAwareSampler batch statistics logging."""

    @pytest.mark.unit
    def test_batch_size_parameter(self):
        dataset = _create_mock_dataset([100, 200, 300])
        sampler = LengthAwareSampler(dataset, batch_size=4)
        assert sampler.batch_size == 4

    @pytest.mark.unit
    def test_logging_called_when_batch_size_greater_than_one(self):
        dataset = _create_mock_dataset([100, 150, 200, 250])

        with patch("llmcompressor.datasets.utils.logger") as mock_logger:
            LengthAwareSampler(dataset, batch_size=2)
            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any("Calculating batch statistics" in c for c in debug_calls)

    @pytest.mark.unit
    def test_tokens_added_calculation(self):
        dataset = _create_mock_dataset([100, 200, 300, 150])

        with patch("llmcompressor.datasets.utils.logger") as mock_logger:
            LengthAwareSampler(dataset, batch_size=2)

            debug_calls = [str(c) for c in mock_logger.debug.call_args_list]
            assert any(
                "added (padding): 150" in c for c in debug_calls
            ), f"Expected 'added (padding): 150' in {debug_calls}"
