import torch

__all__ = ["fix_attention_mask"]


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
