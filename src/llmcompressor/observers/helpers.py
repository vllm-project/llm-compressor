from collections import Counter
from typing import Optional

import torch
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    FloatArgs,
)

__all__ = ["get_observer_token_count", "generate_gparam"]


def get_observer_token_count(module: torch.nn.Module) -> Counter:
    """
    Parse the module and return the number of tokens observed by
    each module's observer.

    :param module: module to parse
    :return: counter with the number of tokens observed by each observer
    """
    token_counts = Counter()
    for name, module in module.named_modules():
        if name.endswith(".input_observer"):
            token_counts[name.replace(".input_observer", "")] = (
                module._num_observed_tokens
            )
    return token_counts


# TODO: we have a similar function in ct already
# consolidate when adding weight global scale
# generation
def generate_gparam(
    updated_min_val: torch.Tensor,
    updated_max_val: torch.Tensor,
    scale_data: Optional[FloatArgs] = FP8_E4M3_DATA,
    quant_data: Optional[FloatArgs] = FP4_E2M1_DATA,
    dtype: Optional[torch.dtype] = torch.float32,
):
    """
    Generate a global scale for an entire tensor (input_tensor).
    Goal of the scale is to ensure that the quantization (local) scale
    falls into the approproiate dtype range.

    E.g. for NVFP4, group (local) scales are in dtype FP8. The global_scale
    attempts to use the entire FP8 dtype range while mapping a per-group max
    to the FP4 max.
    """
    min_vals = torch.min(updated_min_val, torch.zeros_like(updated_min_val))
    max_vals = torch.max(updated_max_val, torch.zeros_like(updated_max_val))
    max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
    global_scale = scale_data.max * quant_data.max / max_val_pos
    return global_scale.to(dtype)
