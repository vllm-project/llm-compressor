import os
from collections import defaultdict
from typing import Mapping, TypeVar

import torch
from compressed_tensors.utils.match import _match_name
from loguru import logger
from transformers.file_utils import CONFIG_NAME

__all__ = [
    "gpu_if_available",
    "find_safetensors_index_path",
    "find_config_path",
    "find_safetensors_index_file",
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

    elif torch.cuda.is_available():
        return torch.device("cuda:0")

    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu:0")

    else:
        logger.warning("CUDA/XPU is not available! Compressing model on CPU instead")
        return torch.device("cpu")


def find_safetensors_index_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name.endswith("safetensors.index.json"):
            return os.path.join(save_directory, file_name)

    return None


def find_config_path(save_directory: str | os.PathLike) -> str | None:
    for file_name in os.listdir(save_directory):
        if file_name in (CONFIG_NAME, "params.json"):
            return os.path.join(save_directory, file_name)

    return None


def find_safetensors_index_file(model_files: dict[str, str]) -> str | None:
    for file_path, resolved_path in model_files.items():
        if file_path.endswith("safetensors.index.json"):
            return resolved_path

    return None


def match_names_set_eager(
    names: set[str] | list[str],
    targets: set[str] | list[str],
    return_unmatched: bool = True,
) -> list[MatchedNamesSet] | tuple[list[MatchedNamesSet], MatchedNamesSet]:
    matched_sets = []
    matches = dict.fromkeys(targets, None)

    for name in names:
        # match until we get a full set
        for target in targets:
            if _match_name(name, target):
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
