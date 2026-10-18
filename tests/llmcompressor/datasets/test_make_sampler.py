from unittest.mock import patch

import pytest
from datasets import Dataset

import llmcompressor.datasets.utils as dataset_utils
from llmcompressor.args import DatasetArguments


def _create_dataset(num_rows: int) -> Dataset:
    return Dataset.from_dict(
        {"input_ids": [[0] * 4] * num_rows, "row": list(range(num_rows))}
    )


def _make_rank_sampler(args, dataset, rank, world_size):
    with (
        patch.object(dataset_utils, "_is_dist_and_same_ds", return_value=True),
        patch.object(dataset_utils.dist, "get_rank", return_value=rank),
        patch.object(dataset_utils.dist, "get_world_size", return_value=world_size),
    ):
        return dataset_utils._make_sampler(args, dataset)


@pytest.mark.unit
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize(
    "num_rows, num_calibration_samples, world_size, expected_total",
    [
        (100, 256, 2, 100),  # more samples requested than rows
        (5, 512, 2, 5),  # uneven partition
        (100, 60, 2, 60),  # fewer samples requested than rows
        (100, None, 3, 100),  # use all rows
    ],
)
def test_distributed_partitions_stay_within_dataset(
    shuffle, num_rows, num_calibration_samples, world_size, expected_total
):
    dataset = _create_dataset(num_rows)
    args = DatasetArguments(
        num_calibration_samples=num_calibration_samples,
        shuffle_calibration_samples=shuffle,
    )

    rows = []
    for rank in range(world_size):
        sampler = _make_rank_sampler(args, dataset, rank, world_size)
        assert len(sampler) == len(sampler.data_source)
        rows.extend(sampler.data_source["row"])

    # every requested row is assigned to exactly one rank
    assert sorted(rows) == list(range(expected_total))


@pytest.mark.unit
def test_distributed_partition_does_not_warn_for_valid_request():
    dataset = _create_dataset(100)
    args = DatasetArguments(
        num_calibration_samples=100, shuffle_calibration_samples=False
    )

    with patch.object(dataset_utils, "logger") as mock_logger:
        _make_rank_sampler(args, dataset, rank=0, world_size=2)

    mock_logger.warning.assert_not_called()
