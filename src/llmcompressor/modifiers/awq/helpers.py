from typing import Tuple

import torch

AWQ_PRECISION = torch.float32


def accumulate_mean(
    inp: torch.Tensor, prev_mean: float, num_samples: int
) -> Tuple[float, int]:
    num_added = inp.size(0)
    input_sum = inp.to(AWQ_PRECISION).sum()

    prev_sum = prev_mean * num_samples
    num_samples += num_added

    return (prev_sum + input_sum) / num_samples, num_samples
