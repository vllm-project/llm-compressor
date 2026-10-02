import math
from typing import Any

import torch

from llmcompressor.pipelines.cache import IntermediatesCache

__all__ = [
    "compute_sqnr",
    "compute_subgraph_sqnr",
    "process_batch_error",
]


def _get_largest_tensor(outputs: dict[str, Any]) -> torch.Tensor | None:
    """
    Select the largest tensor (by number of elements) from a subgraph output dict.
    This is used as a heuristic to identify the "main activation" (e.g. hidden
    states) among a subgraph's outputs, which may also include pass-through values
    such as attention masks or position ids.

    :param outputs: dictionary of subgraph output values
    :return: the largest tensor value, or None if no tensor values are present
    """
    largest = None
    largest_numel = 0
    for value in outputs.values():
        if isinstance(value, torch.Tensor) and value.numel() > largest_numel:
            largest = value
            largest_numel = value.numel()
    return largest


def compute_sqnr(signal_power_sum: float, noise_power_sum: float) -> float:
    """
    Compute the Signal-to-Quantization-Noise Ratio (SQNR), in decibels, from
    accumulated signal and noise power sums; higher values indicate less
    distortion. Callers should sum power across every batch in a subgraph
    (see :func:`_accumulate_batch_power`) and call this once on the totals,
    since SQNR is a logarithmic ratio and cannot be meaningfully averaged
    batch-by-batch.

    :param signal_power_sum: sum of squared pre-compression activation values
    :param noise_power_sum: sum of squared compression noise values
    :return: SQNR in dB, or ``inf`` if there is no noise
    """
    if noise_power_sum == 0:
        return float("inf")
    return 10 * math.log10(signal_power_sum / noise_power_sum)


def compute_subgraph_sqnr(batch_powers: list[tuple[float, float]]) -> float:
    """
    Aggregate per-batch signal/noise power sums and compute the subgraph's
    overall SQNR in one call.

    :param batch_powers: list of (signal power sum, noise power sum) tuples,
        one per batch, as returned by :func:`process_batch_error`
    :return: SQNR in dB, or ``inf`` if there is no noise
    """
    signal_power_sum = sum(signal for signal, _ in batch_powers)
    noise_power_sum = sum(noise for _, noise in batch_powers)
    return compute_sqnr(signal_power_sum, noise_power_sum)


def _accumulate_batch_power(
    cache: IntermediatesCache,
    batch_idx: int,
    output: dict[str, Any],
) -> tuple[float, float] | None:
    """
    Compute the signal and noise power sums for a single batch by comparing
    the cached pre-compression outputs against the post-compression output,
    then remove the cached entry to free memory. Power is summed (not
    averaged) so callers can add it across batches and compute a single
    SQNR from the totals via :func:`compute_subgraph_sqnr`.

    :param cache: intermediates cache holding the pre-compression outputs
    :param batch_idx: index of the current propagation batch
    :param output: subgraph output dict from the propagation (post-compression) pass
    :return: (signal power sum, noise power sum) for this batch, or None if
        either tensor is unavailable
    """
    pre_outputs = cache.fetch(batch_idx)
    cache.delete(batch_idx)

    pre_tensor = _get_largest_tensor(pre_outputs)
    post_tensor = _get_largest_tensor(output)
    if pre_tensor is None or post_tensor is None:
        return None

    noise = post_tensor.float() - pre_tensor.float()
    signal_power_sum = (pre_tensor.float() ** 2).sum().item()
    noise_power_sum = (noise**2).sum().item()
    return signal_power_sum, noise_power_sum


def process_batch_error(
    cache: IntermediatesCache,
    activations: IntermediatesCache,
    batch_idx: int,
    output: dict[str, Any],
    propagate_error: bool,
    consumed_names: set[str],
) -> tuple[float, float] | None:
    """
    Process SQNR tracking for a single batch during the propagation pass. When
    propagate_error is disabled, first transfer this subgraph's unquantized
    outputs into activations for the next subgraph, since pass 2 doesn't do
    so itself in that case. Then compare pre/post-compression outputs to
    accumulate this batch's signal/noise power (see
    :func:`_accumulate_batch_power`).

    :param cache: intermediates cache dedicated to log_sequential_error
    :param activations: intermediates cache shared across the pipeline
    :param batch_idx: index of the current propagation batch
    :param output: subgraph output dict from the propagation (post-compression) pass
    :param propagate_error: whether quantized outputs are fed to the next subgraph
    :param consumed_names: names no longer needed by any subsequent subgraph
    :return: (signal power sum, noise power sum) for this batch, or None if
        either tensor is unavailable
    """
    if not propagate_error:
        activations.transfer(cache, batch_idx)
        activations.delete(batch_idx, consumed_names)

    return _accumulate_batch_power(cache, batch_idx, output)
