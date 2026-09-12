import math
from typing import Any

import torch

from llmcompressor.pipelines.cache import IntermediatesCache

__all__ = ["compute_sqnr", "cache_pre_compression_output", "accumulate_batch_power"]

_PRE_ERROR_KEY = "__pre_error__"


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
    (see :func:`accumulate_batch_power`) and call this once on the totals,
    since SQNR is a logarithmic ratio and cannot be meaningfully averaged
    batch-by-batch.

    :param signal_power_sum: sum of squared pre-compression activation values
    :param noise_power_sum: sum of squared compression noise values
    :return: SQNR in dB, or ``inf`` if there is no noise
    """
    if noise_power_sum == 0:
        return float("inf")
    return 10 * math.log10(signal_power_sum / noise_power_sum)


def cache_pre_compression_output(
    activations: IntermediatesCache,
    batch_idx: int,
    outputs: dict[str, Any],
) -> None:
    """
    Save the main activation tensor from a subgraph's pre-compression output
    into the intermediates cache for later error comparison.

    :param activations: intermediates cache shared across the pipeline
    :param batch_idx: index of the current calibration batch
    :param outputs: subgraph output dict from the calibration (pre-compression) pass
    """
    main_tensor = _get_largest_tensor(outputs)
    if main_tensor is not None:
        activations.update(batch_idx, {_PRE_ERROR_KEY: main_tensor})


def accumulate_batch_power(
    activations: IntermediatesCache,
    batch_idx: int,
    output: dict[str, Any],
) -> tuple[float, float] | None:
    """
    Compute the signal and noise power sums for a single batch by comparing
    the saved pre-compression activation against the post-compression output,
    then remove the cached pre-compression tensor to free memory. Power is
    summed (not averaged) so callers can add it across batches and compute a
    single SQNR from the totals via :func:`compute_sqnr`.

    :param activations: intermediates cache holding the pre-compression tensor
    :param batch_idx: index of the current propagation batch
    :param output: subgraph output dict from the propagation (post-compression) pass
    :return: (signal power sum, noise power sum) for this batch, or None if
        either tensor is unavailable
    """
    pre_data = activations.fetch(batch_idx, [_PRE_ERROR_KEY])
    pre_tensor = pre_data.get(_PRE_ERROR_KEY)
    if pre_tensor is None:
        return None
    # clean up cached tensor; delete only after confirming the key exists,
    # since IntermediatesCache.delete raises KeyError for missing keys
    activations.delete(batch_idx, [_PRE_ERROR_KEY])

    post_tensor = _get_largest_tensor(output)
    if post_tensor is None:
        return None

    noise = post_tensor.float() - pre_tensor.float()
    signal_power_sum = (pre_tensor.float() ** 2).sum().item()
    noise_power_sum = (noise**2).sum().item()
    return signal_power_sum, noise_power_sum
