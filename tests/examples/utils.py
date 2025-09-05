from typing import Union

import pytest


def requires_gpu_count(num_required_gpus: int) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on number of available GPUs. This plays nicely with
    the CUDA_VISIBLE_DEVICES environment variable.
    """
    import torch

    num_gpus = torch.cuda.device_count()
    reason = f"{num_required_gpus} GPUs required, {num_gpus} GPUs detected"
    return pytest.mark.skipif(num_required_gpus > num_gpus, reason=reason)


def requires_gpu_mem(required_amount: Union[int, float]) -> pytest.MarkDecorator:
    """
    Pytest decorator to skip based on total available GPU memory (across all GPUs). This
    plays nicely with the CUDA_VISIBLE_DEVICES environment variable.

    Note: make sure to account for measured memory vs. simple specs. For example, H100
    has '80 GiB' VRAM, however, the actual number, at least per PyTorch, is ~79.2 GiB.

    :param amount: amount of required GPU memory in GiB
    """
    import torch

    vram_bytes = sum(
        torch.cuda.mem_get_info(device_id)[1]
        for device_id in range(torch.cuda.device_count())
    )
    actual_vram = vram_bytes / 1024**3
    reason = (
        f"{required_amount} GiB GPU memory required, "
        f"{actual_vram:.1f} GiB GPU memory found"
    )
    return pytest.mark.skipif(required_amount > actual_vram, reason=reason)
