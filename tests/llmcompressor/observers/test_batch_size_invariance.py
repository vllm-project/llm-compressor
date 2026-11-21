import pytest
import torch
from compressed_tensors.quantization.quant_args import QuantizationArgs

from llmcompressor.observers import Observer
from llmcompressor.observers.helpers import (
    lerp,
    packed_sample_lengths,
    sample_lengths_from_position_ids,
)


def _load(name, base_name, strategy, **observer_kwargs):
    args = QuantizationArgs(
        num_bits=8,
        strategy=strategy,
        symmetric=False,
        observer=name,
        observer_kwargs=observer_kwargs,
    )
    return Observer.load_from_registry(name, base_name=base_name, args=args)


def _samples(base_name, num_samples, seq_len=7):
    torch.manual_seed(0)
    # scale each sample differently so that per-batch and per-sample averaging
    # give clearly different statistics
    scales = torch.linspace(0.5, 4.0, num_samples)
    if base_name in ("q", "k", "v"):
        shape = (1, 4, seq_len, 8)  # (batch, num_heads, seq_len, head_dim)
    else:
        shape = (1, seq_len, 32)  # (batch, seq_len, hidden_dim)
    return [torch.randn(shape) * scale for scale in scales]


@pytest.mark.parametrize("name", ["minmax", "mse"])
@pytest.mark.parametrize(
    "base_name,strategy",
    [
        ("input", "tensor"),
        ("output", "tensor"),
        ("k", "tensor"),
        ("v", "attn_head"),
    ],
)
@pytest.mark.parametrize("batch_size", [2, 4])
def test_moving_average_independent_of_batch_size(
    name, base_name, strategy, batch_size
):
    samples = _samples(base_name, num_samples=8)

    one_at_a_time = _load(name, base_name, strategy, averaging_constant=0.1)
    for sample in samples:
        one_at_a_time(sample)

    batched = _load(name, base_name, strategy, averaging_constant=0.1)
    for start in range(0, len(samples), batch_size):
        batched(torch.cat(samples[start : start + batch_size]))

    assert torch.equal(batched.min_vals, one_at_a_time.min_vals)
    assert torch.equal(batched.max_vals, one_at_a_time.max_vals)


def test_minmax_batch_averages_each_sample():
    first = torch.tensor([[[-1.0, 1.0]]])
    second = torch.tensor([[[-3.0, 5.0]]])

    observer = _load("minmax", "input", "tensor", averaging_constant=0.25)
    observer(torch.cat([first, second]))

    assert torch.allclose(observer.min_vals, lerp(torch.tensor(-1.0), -3.0, 0.25))
    assert torch.allclose(observer.max_vals, lerp(torch.tensor(1.0), 5.0, 0.25))


def test_minmax_unbatched_input_is_one_observation():
    # e.g. the routed tokens of an MoE expert carry no sample boundaries
    first = torch.tensor([[-1.0, 1.0]])
    second = torch.tensor([[-3.0, 2.0], [-2.0, 5.0]])

    observer = _load("minmax", "input", "tensor", averaging_constant=0.25)
    observer(first)
    observer(second)

    assert torch.allclose(observer.min_vals, lerp(torch.tensor(-1.0), -3.0, 0.25))
    assert torch.allclose(observer.max_vals, lerp(torch.tensor(1.0), 5.0, 0.25))


@pytest.mark.parametrize("name", ["static_minmax", "memoryless_minmax"])
def test_other_minmax_observers_observe_whole_batch(name):
    batch = torch.tensor([[[-1.0, 1.0]], [[-3.0, 5.0]]])

    observer = _load(name, "input", "tensor")
    observer(batch)

    assert observer.min_vals.item() == -3.0
    assert observer.max_vals.item() == 5.0


@pytest.mark.parametrize("name", ["minmax", "mse"])
@pytest.mark.parametrize(
    "base_name,strategy", [("input", "tensor"), ("k", "tensor"), ("v", "attn_head")]
)
@pytest.mark.parametrize("lengths", [[7, 7, 7, 7], [3, 11, 1, 6]])
def test_moving_average_splits_packed_rows(name, base_name, strategy, lengths):
    samples = [
        sample.narrow(2 if base_name in ("q", "k", "v") else 1, 0, length)
        for sample, length in zip(
            _samples(base_name, num_samples=4, seq_len=11), lengths
        )
    ]
    seq_dim = 2 if base_name in ("q", "k", "v") else 1

    one_at_a_time = _load(name, base_name, strategy, averaging_constant=0.1)
    for sample in samples:
        one_at_a_time(sample)

    packed = _load(name, base_name, strategy, averaging_constant=0.1)
    with packed_sample_lengths(lengths):
        packed(torch.cat(samples, dim=seq_dim))

    assert torch.equal(packed.min_vals, one_at_a_time.min_vals)
    assert torch.equal(packed.max_vals, one_at_a_time.max_vals)


def test_packed_row_without_lengths_is_one_observation():
    row = torch.tensor([[[-1.0, 1.0], [-3.0, 5.0]]])  # two packed 1-token samples

    observer = _load("minmax", "input", "tensor", averaging_constant=0.25)
    observer(row)

    assert observer.min_vals.item() == -3.0
    assert observer.max_vals.item() == 5.0


def test_packed_lengths_must_match_the_row():
    row = torch.tensor([[[-1.0, 1.0], [-3.0, 5.0]]])

    observer = _load("minmax", "input", "tensor", averaging_constant=0.25)
    with packed_sample_lengths([1, 2]):  # 3 tokens, but the row has 2
        observer(row)

    assert observer.max_vals.item() == 5.0


@pytest.mark.parametrize(
    "position_ids,lengths",
    [
        ([[0, 1, 2, 0, 1]], [3, 2]),
        ([[0, 0, 1]], [1, 2]),
        ([[0, 1, 2]], None),  # one sample
        ([[1, 2, 0, 1]], None),  # does not start a sample
        ([[0, 1, 0], [0, 1, 0]], None),  # more than one row
    ],
)
def test_sample_lengths_from_position_ids(position_ids, lengths):
    assert sample_lengths_from_position_ids(torch.tensor(position_ids)) == lengths
