import pytest
from datasets import Dataset

from llmcompressor.args import DatasetArguments, parse_args
from llmcompressor.datasets.utils import _make_sampler

NUM_SAMPLES = 8
DATASET_SIZE = 32


def _dataset() -> Dataset:
    return Dataset.from_dict({"input_ids": [[i] for i in range(DATASET_SIZE)]})


def _draw(seed: int | None) -> list[int]:
    args = DatasetArguments(
        num_calibration_samples=NUM_SAMPLES,
        shuffle_calibration_samples=True,
        calibration_seed=seed,
    )
    return list(_make_sampler(args, _dataset()))


class TestCalibrationSamplerSeed:
    @pytest.mark.unit
    @pytest.mark.parametrize("seed", [0, 42])
    def test_same_seed_same_order(self, seed: int):
        assert _draw(seed) == _draw(seed)

    @pytest.mark.unit
    def test_different_seeds_differ(self):
        assert _draw(0) != _draw(1)

    @pytest.mark.unit
    def test_omitted_seed_draws_valid_indices(self):
        drawn = _draw(None)
        assert len(drawn) == NUM_SAMPLES
        assert len(set(drawn)) == len(drawn)
        assert set(drawn) <= set(range(DATASET_SIZE))

    @pytest.mark.unit
    def test_parse_args_threads_seed(self):
        _, dataset_args, _, _ = parse_args(
            model="unused",
            calibration_seed=5,
            num_calibration_samples=NUM_SAMPLES,
        )
        assert dataset_args.calibration_seed == 5
        assert list(_make_sampler(dataset_args, _dataset())) == _draw(5)
