import os

import torch
import torch.distributed as dist
from compressed_tensors.offload import init_dist

__all__ = [
    "fix_attention_mask",
    "get_local_gpu_group_size",
    "init_gpu_group_dist",
]


def get_local_gpu_group_size() -> int:
    return int(os.environ.get("LLMCOMPRESSOR_GPUS_PER_GROUP", "1"))


def init_gpu_group_dist(gpus_per_group: int | None = None) -> tuple[int, int, int]:
    """
    Initialize DDP for AutoRound's rank-local GPU grouping.

    Standard DDP binds each rank to ``LOCAL_RANK``. AutoRound can instead use a
    rank-local group of GPUs to tune a single decoding block, so the process group
    must be initialized on ``LOCAL_RANK * gpus_per_group`` from the start.
    """
    gpus_per_group = (
        get_local_gpu_group_size() if gpus_per_group is None else gpus_per_group
    )
    if gpus_per_group < 1:
        raise ValueError(
            f"LLMCOMPRESSOR_GPUS_PER_GROUP must be >= 1, got {gpus_per_group}"
        )

    if gpus_per_group == 1:
        init_dist()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size, int(os.environ["LOCAL_RANK"])

    if "TORCHELASTIC_RUN_ID" not in os.environ:
        raise ValueError(
            "Cannot find distributed environment. "
            "Please make sure you are using `torchrun --nproc-per-node ...`."
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    main_gpu = local_rank * gpus_per_group
    device_count = torch.accelerator.device_count()

    if main_gpu + gpus_per_group > device_count:
        raise ValueError(
            "Requested GPU group exceeds local visible devices: "
            f"main_gpu={main_gpu}, gpus_per_group={gpus_per_group}, "
            f"device_count={device_count}"
        )

    accel_type = torch.accelerator.current_accelerator().type
    if accel_type == "cuda":
        backend = "nccl"
    elif accel_type == "xpu":
        backend = "xccl"
    else:
        backend = "gloo"

    torch.accelerator.set_device_index(main_gpu)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"{accel_type}:{main_gpu}"),
    )
    dist.barrier()
    return rank, world_size, main_gpu


def _collapse_causal_attention_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Reduce causal attention masks to the per-token validity mask AutoRound expects.

    AutoRound uses attention masks only to exclude padded query positions from the
    reconstruction loss, so higher-rank causal masks need to be collapsed to a
    `[batch, seq_len]` mask first.
    """
    if mask.ndim == 4:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(
            "Unsupported causal attention mask shape for AutoRound: "
            f"{tuple(mask.shape)}"
        )

    if mask.dtype == torch.bool:
        return mask.any(dim=-1)

    if mask.numel() == 0 or torch.all(mask == 0):
        raise ValueError(
            "Invalid causal attention mask for AutoRound: all positions are masked"
        )

    global_min = mask.amin()
    return (mask.amax(dim=-1) > global_min).to(torch.long)


def fix_attention_mask(
    mask: torch.Tensor | list[int] | list[list[int]],
) -> torch.Tensor:
    """
    Normalize attention masks for AutoRound custom datasets.

    AutoRound expects at least one masked position when the calibration mask is fully
    dense. When every token is marked valid, set the final position to 0 while
    preserving the original dtype and shape.
    More details can be found here: https://github.com/intel/auto-round/blob/50ee58c9e176e9da2a744dbe6ed220f26e80eccd/auto_round/calibration/llm.py#L315-L355
    """
    normalized_mask = torch.as_tensor(mask).clone()
    if normalized_mask.shape[-1] == 0:
        return normalized_mask

    if (
        normalized_mask.ndim == 4
        and normalized_mask.shape[1] == 1
        and normalized_mask.shape[2] == 1
    ):
        normalized_mask = normalized_mask.squeeze(2).squeeze(1)

    if normalized_mask.ndim in (3, 4):
        normalized_mask = _collapse_causal_attention_mask(normalized_mask)

    if normalized_mask.ndim == 1:
        if torch.all(normalized_mask == 1):
            normalized_mask[-1] = 0
        return normalized_mask

    if normalized_mask.ndim == 2:
        all_ones_rows = torch.all(normalized_mask == 1, dim=1)
        if torch.any(all_ones_rows):
            normalized_mask[all_ones_rows, -1] = 0
        return normalized_mask

    raise ValueError(
        "Unsupported attention mask shape for AutoRound: "
        f"{tuple(normalized_mask.shape)}"
    )
