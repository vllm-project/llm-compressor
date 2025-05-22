from typing import Tuple

import torch

AWQ_PRECISION = torch.float32


def accumulate_mean(
    inp: torch.Tensor, prev_mean_and_count: Tuple[float, int]
) -> Tuple[float, int]:
    prev_mean, prev_count = prev_mean_and_count
    num_added = inp.size(0)
    input_sum = inp.to(AWQ_PRECISION).sum()

    prev_sum = prev_mean * prev_count
    count = prev_count + num_added

    return (prev_sum + input_sum) / count, count
