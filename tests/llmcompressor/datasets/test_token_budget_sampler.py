import random

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, SequentialSampler

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.datasets.utils import (
    DataCollatorWithTruncation,
    TokenBudgetBatchSampler,
    batch_by_tokens,
    format_calibration_data,
)


def _dataset(lengths: list[int]) -> Dataset:
    return Dataset.from_dict({"input_ids": [[1] * length for length in lengths]})


@pytest.mark.unit
@pytest.mark.parametrize("max_tokens,spread", [(64, 0.1), (200, 0.3), (30, 0.0)])
def test_batches_fit_budget_and_spread(max_tokens, spread):
    rng = random.Random(0)
    lengths = [rng.randint(1, 40) for _ in range(100)]
    sampler = TokenBudgetBatchSampler(range(100), lengths, max_tokens, spread)

    indices = [index for batch in sampler for index in batch]
    assert sorted(indices) == list(range(100))
    assert len(sampler) == len(sampler.batches)

    for batch in sampler:
        batch_lengths = [lengths[index] for index in batch]
        longest = max(batch_lengths)
        assert len(batch) == 1 or len(batch) * longest <= max_tokens
        assert min(batch_lengths) >= (1 - spread) * longest

    # batches follow the sampler's order rather than descending length
    firsts = [batch[0] for batch in sampler]
    assert firsts == sorted(firsts)
    assert all(batch == sorted(batch) for batch in sampler)


@pytest.mark.unit
def test_batches_keep_the_sampler_order_not_length_order():
    # lengths alternate, so length buckets interleave in the sampler's order
    lengths = {0: 4, 1: 20, 2: 4, 3: 20, 4: 4}
    sampler = TokenBudgetBatchSampler([4, 1, 0, 3, 2], lengths, max_tokens=60)
    assert sampler.batches == [[4, 0, 2], [1, 3]]


@pytest.mark.unit
def test_batches_use_only_the_sampled_subset():
    lengths = [5, 6, 7, 8, 9]
    sampler = TokenBudgetBatchSampler(
        [3, 1, 4], lengths, max_tokens=100, max_length_spread=0.5
    )
    assert sampler.batches == [[3, 1, 4]]


@pytest.mark.unit
def test_equal_lengths_keep_the_sampler_order():
    sampler = TokenBudgetBatchSampler([2, 0, 3, 1], [4] * 4, max_tokens=8)
    assert sampler.batches == [[2, 0], [3, 1]]


@pytest.mark.unit
def test_sample_over_budget_gets_its_own_batch():
    sampler = TokenBudgetBatchSampler(range(3), [50, 10, 10], max_tokens=20)
    assert sampler.batches == [[0], [1, 2]]


@pytest.mark.unit
def test_rejects_empty_budget():
    with pytest.raises(ValueError):
        TokenBudgetBatchSampler(range(2), [1, 1], max_tokens=0)


@pytest.mark.unit
def test_batch_by_tokens_keeps_dataset_and_collate_fn():
    dataset = _dataset([10, 4, 4, 3])
    collate_fn = DataCollatorWithTruncation(max_seq_length=4)
    dataloader = DataLoader(
        dataset, batch_size=1, sampler=SequentialSampler(dataset), collate_fn=collate_fn
    )

    rebuilt = batch_by_tokens(dataloader, max_tokens=12)
    assert rebuilt.dataset is dataset
    assert rebuilt.collate_fn is collate_fn

    # the collator truncates the first sample to 4 tokens, so it shares a batch
    shapes = [tuple(batch["input_ids"].shape) for batch in rebuilt]
    assert shapes == [(3, 4), (1, 3)]


@pytest.mark.unit
def test_batch_by_tokens_does_not_advance_random_state():
    dataset = _dataset([3, 4, 5, 6])
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=DataCollatorWithTruncation()
    )

    state = torch.random.get_rng_state()
    batch_by_tokens(dataloader, max_tokens=100)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.unit
def test_batch_by_tokens_does_not_advance_a_dataloaders_generator():
    dataset = _dataset([3, 4, 5, 6])
    generator = torch.Generator().manual_seed(0)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, generator=generator, collate_fn=list
    )

    state = generator.get_state()
    rebuilt = batch_by_tokens(dataloader, max_tokens=100)
    assert torch.equal(generator.get_state(), state)
    assert rebuilt.generator is generator


@pytest.mark.unit
def test_batch_by_tokens_reads_only_sampled_lengths():
    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, index):
            if index >= 3:
                raise AssertionError(f"read unsampled sample {index}")
            return {"input_ids": [1] * (index + 1)}

    dataloader = DataLoader(Samples(), batch_size=1, sampler=[0, 1, 2])
    rebuilt = batch_by_tokens(dataloader, max_tokens=100)
    # lengths 1, 2 and 3 differ by more than the length spread
    assert rebuilt.batch_sampler.batches == [[0], [1], [2]]


@pytest.mark.unit
def test_batch_by_tokens_keeps_loading_options():
    def worker_init_fn(worker_id):
        pass

    dataset = _dataset([3, 4])
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        prefetch_factor=3,
        timeout=5,
        worker_init_fn=worker_init_fn,
        collate_fn=list,
    )
    rebuilt = batch_by_tokens(dataloader, max_tokens=100)
    assert rebuilt.num_workers == 1
    assert rebuilt.prefetch_factor == 3
    assert rebuilt.timeout == 5
    assert rebuilt.worker_init_fn is worker_init_fn


@pytest.mark.unit
def test_batch_by_tokens_falls_back_without_input_ids():
    samples = [{"inputs_embeds": torch.zeros(2, 4)} for _ in range(3)]
    dataloader = DataLoader(samples, batch_size=1)
    assert batch_by_tokens(dataloader, max_tokens=100) is dataloader


@pytest.mark.unit
def test_batch_by_tokens_falls_back_for_other_iterables():
    batches = [{"input_ids": torch.ones(1, 4, dtype=torch.long)}]
    assert batch_by_tokens(batches, max_tokens=100) is batches


@pytest.mark.unit
def test_auto_dataloader_starts_with_batch_size_1():
    args = DatasetArguments(batch_size="auto", shuffle_calibration_samples=False)
    dataloader = format_calibration_data(args, _dataset([3, 4, 5]), processor=None)
    assert dataloader.batch_size == 1


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", ["auto", 1, 8])
def test_dataset_arguments_accept_batch_size(batch_size):
    assert DatasetArguments(batch_size=batch_size).batch_size == batch_size


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [0, -2, "max", 1.5, True])
def test_dataset_arguments_reject_invalid_batch_size(batch_size):
    with pytest.raises(ValueError):
        DatasetArguments(batch_size=batch_size)
