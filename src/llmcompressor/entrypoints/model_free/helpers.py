import torch
from loguru import logger

__all__ = ["gpu_if_available"]


def gpu_if_available(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    elif torch.accelerator.is_available():
        accel_type = torch.accelerator.current_accelerator().type
        return torch.device(accel_type, 0)

    else:
        logger.warning("No accelerator available! Compressing model on CPU instead")
        return torch.device("cpu")


def build_weights_map(
    weight_map: dict[str, str],
    model_files: dict[str, str],
) -> dict[str, str]:
    """
    Build a mapping of tensor name -> resolved file path from the model's
    weight_map (index.json). This allows any process to locate fused partner
    tensors from other shards without loading entire files.

    :param weight_map: mapping of tensor name -> shard filename (from index.json)
    :param model_files: mapping of shard filename -> resolved absolute path
    :return: mapping of tensor name -> resolved absolute path
    """
    return {
        tensor_name: model_files[shard_name]
        for tensor_name, shard_name in weight_map.items()
        if shard_name in model_files
    }
