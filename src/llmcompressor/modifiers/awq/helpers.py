from typing import Tuple, Optional

import torch


def accumulate_mean(
    inp: torch.Tensor,
    prev_mean_and_count: Optional[Tuple[torch.FloatTensor, int]],
) -> Tuple[float, int]:
    sum_added = inp.sum(dim=0)
    num_added = inp.size(0)
    if prev_mean_and_count is None:
        return sum_added, num_added

    prev_mean, prev_count = prev_mean_and_count

    prev_sum = prev_mean * prev_count
    new_count = prev_count + num_added

    return (prev_sum + sum_added) / new_count, new_count
