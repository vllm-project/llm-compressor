"""
Tests for State.has_calibration_data
"""

from torch.utils.data import DataLoader, TensorDataset

from llmcompressor.core import State


def _dataloader(num_samples: int) -> DataLoader:
    import torch

    dataset = TensorDataset(torch.randn(num_samples, 8))
    return DataLoader(dataset, batch_size=2)


def test_no_calibration_data():
    state = State()
    assert state.data.calib is None
    assert state.has_calibration_data is False


def test_calibration_data_with_samples():
    state = State()
    state.data.calib = _dataloader(num_samples=4)
    assert state.has_calibration_data is True


def test_calibration_data_without_samples():
    state = State()
    state.data.calib = _dataloader(num_samples=0)
    assert state.has_calibration_data is False


def test_unsized_dataloader_is_assumed_to_have_data():
    import torch

    class _IterableCalibDataset:
        def __iter__(self):
            yield {"input_ids": torch.randint(0, 100, (8,))}

    state = State()
    state.data.calib = DataLoader(_IterableCalibDataset())
    # dataloaders over iterable datasets have no __len__; assume data exists
    assert state.has_calibration_data is True
