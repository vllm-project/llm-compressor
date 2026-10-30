import pytest
import torch.distributed as dist
from datasets import Dataset

from llmcompressor.args import DatasetArguments
from llmcompressor.datasets.utils import format_calibration_data
from tests.testing_utils import torchrun


def _make_dataset(num_rows: int) -> Dataset:
    """Row i holds input_ids [i] * (i + 1), so the first token of a sample
    identifies its row index in the original (unpartitioned) dataset"""
    return Dataset.from_dict({"input_ids": [[i] * (i + 1) for i in range(num_rows)]})


def _rows_seen(args: DatasetArguments, dataset: Dataset) -> set[int]:
    dataloader = format_calibration_data(args, dataset, processor=None)
    return {int(batch["input_ids"][0, 0]) for batch in dataloader}


def _gather_rows(rows: set[int]) -> list[set[int]]:
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, rows)
    return gathered


@pytest.mark.unit
@torchrun(world_size=2)
def test_same_dataset_partitions_disjointly():
    """When every rank passes the same dataset, each rank must calibrate on a
    disjoint partition of it, and the partitions must cover the whole dataset"""
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()

    args = DatasetArguments(
        num_calibration_samples=None, shuffle_calibration_samples=False, batch_size=1
    )
    rows = _rows_seen(args, _make_dataset(10))
    assert rows == set(range(5 * rank, 5 * (rank + 1)))

    # partitions are disjoint and cover the dataset
    all_rows = _gather_rows(rows)
    assert set().union(*all_rows) == set(range(10))
    assert len(set().intersection(*all_rows)) == 0

    dist.destroy_process_group()


@pytest.mark.unit
@torchrun(world_size=2)
def test_num_calibration_samples_partitions_subset():
    """num_calibration_samples must be partitioned across ranks rather than
    re-read from the head of the full dataset on every rank (issue #3413)"""
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()

    args = DatasetArguments(
        num_calibration_samples=8, shuffle_calibration_samples=False, batch_size=1
    )
    rows = _rows_seen(args, _make_dataset(10))
    assert rows == set(range(4 * rank, 4 * (rank + 1)))

    all_rows = _gather_rows(rows)
    assert set().union(*all_rows) == set(range(8))
    assert len(set().intersection(*all_rows)) == 0

    dist.destroy_process_group()


@pytest.mark.unit
@torchrun(world_size=2)
def test_shuffle_preserves_partition():
    """Shuffling may reorder samples within a rank but must not change which
    rows each rank calibrates on"""
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()

    args = DatasetArguments(
        num_calibration_samples=None, shuffle_calibration_samples=True, batch_size=1
    )
    rows = _rows_seen(args, _make_dataset(10))
    assert rows == set(range(5 * rank, 5 * (rank + 1)))

    all_rows = _gather_rows(rows)
    assert set().union(*all_rows) == set(range(10))
    assert len(set().intersection(*all_rows)) == 0

    dist.destroy_process_group()


@pytest.mark.unit
@torchrun(world_size=3)
def test_uneven_partition_covers_dataset():
    """With a row count that doesn't divide evenly by world size, the partitions
    must still be disjoint and cover the dataset"""
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()

    args = DatasetArguments(
        num_calibration_samples=None, shuffle_calibration_samples=False, batch_size=1
    )
    rows = _rows_seen(args, _make_dataset(10))

    # _get_partition_start_end(10, rank, 3): [0, 3), [3, 6), [6, 10)
    expected_by_rank = [set(range(0, 3)), set(range(3, 6)), set(range(6, 10))]
    assert rows == expected_by_rank[rank]

    all_rows = _gather_rows(rows)
    assert set().union(*all_rows) == set(range(10))
    assert len(set().intersection(*all_rows)) == 0

    dist.destroy_process_group()


@pytest.mark.unit
def test_single_process_reads_full_dataset():
    """Without distributed initialization the dataloader must keep reading the
    full dataset (the fix only redirects the dataset the sampler indexes)"""
    args = DatasetArguments(
        num_calibration_samples=None, shuffle_calibration_samples=False, batch_size=1
    )
    rows = _rows_seen(args, _make_dataset(10))
    assert rows == set(range(10))
