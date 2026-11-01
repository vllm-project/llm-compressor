import inspect

import pytest
import torch
from datasets import Dataset

from llmcompressor.args import DatasetArguments, parse_args
from llmcompressor.datasets.utils import _make_sampler
from llmcompressor.entrypoints.oneshot import oneshot

NUM_SAMPLES = 8
DATASET_SIZE = 32


def _dataset() -> Dataset:
    return Dataset.from_dict({"input_ids": [[i] for i in range(DATASET_SIZE)]})


def _draw(
    seed: int | None,
    generator: torch.Generator | None = None,
) -> list[int]:
    args = DatasetArguments(
        num_calibration_samples=NUM_SAMPLES,
        shuffle_calibration_samples=True,
        calibration_seed=seed,
        calibration_generator=generator,
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
    def test_generator_matches_seed(self):
        seed = 7
        generator = torch.Generator().manual_seed(seed)
        assert _draw(None, generator=generator) == _draw(seed)

    @pytest.mark.unit
    def test_same_generator_seed_same_order(self):
        first = torch.Generator().manual_seed(3)
        second = torch.Generator().manual_seed(3)
        assert _draw(None, generator=first) == _draw(None, generator=second)

    @pytest.mark.unit
    def test_seed_and_generator_together_rejected(self):
        generator = torch.Generator().manual_seed(1)
        with pytest.raises(ValueError, match="calibration_seed"):
            _draw(1, generator=generator)

    @pytest.mark.unit
    def test_parse_args_threads_seed_and_generator(self):
        generator = torch.Generator().manual_seed(5)
        _, dataset_args, _, _ = parse_args(
            model="unused",
            calibration_seed=5,
            calibration_generator=None,
        )
        assert dataset_args.calibration_seed == 5
        assert dataset_args.calibration_generator is None

        _, dataset_args, _, _ = parse_args(
            model="unused",
            calibration_generator=generator,
        )
        assert dataset_args.calibration_seed is None
        assert dataset_args.calibration_generator is generator

        params = inspect.signature(oneshot).parameters
        assert params["calibration_seed"].default is None
        assert params["calibration_generator"].default is None
