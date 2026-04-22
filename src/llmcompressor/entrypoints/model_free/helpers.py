import re
from collections import defaultdict
from typing import Mapping, TypeVar

import torch
from compressed_tensors.utils.match import match_name
from loguru import logger

__all__ = [
    "gpu_if_available",
    "match_names_set_eager",
    "MatchedNamesSet",
    "invert_mapping",
]

KeyType = TypeVar("K")
ValueType = TypeVar("V")
MatchedNamesSet = dict[str, str | None]


def gpu_if_available(device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)

    elif torch.accelerator.is_available():
        accel_type = torch.accelerator.current_accelerator().type
        return torch.device(accel_type, 0)

    else:
        logger.warning("No accelerator available! Compressing model on CPU instead")
        return torch.device("cpu")


def match_names_set_eager(
    names: set[str] | list[str],
    targets: set[str] | list[str],
    return_unmatched: bool = True,
) -> list[MatchedNamesSet] | tuple[list[MatchedNamesSet], MatchedNamesSet]:
    matched_sets = []
    matches = dict.fromkeys(targets, None)

    def natural_key(s: str) -> list[str | int]:
        return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]

    # natural sort for consistent grouping
    names = sorted(names, key=natural_key)

    for name in names:
        # match until we get a full set
        for target in targets:
            if match_name(name, target):
                if matches[target] is None:
                    matches[target] = name
                else:
                    # matched target twice without completing a set
                    raise ValueError(
                        f"Matched a {target} twice before "
                        f"completing set ({matches[target]}, {name})"
                    )

        # once we have a full set, yield and reset
        if all((matches[target] is not None for target in targets)):
            matched_sets.append(matches)
            matches = dict.fromkeys(targets, None)

    unmatched_set = matches if any((v is not None for v in matches.values())) else None

    if return_unmatched:
        return matched_sets, unmatched_set
    else:
        return matched_sets


def invert_mapping(
    mapping: Mapping[KeyType, ValueType],
) -> dict[ValueType, list[KeyType]]:
    inverse = defaultdict(list)

    for key, value in mapping.items():
        inverse[value].append(key)

    return inverse


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
