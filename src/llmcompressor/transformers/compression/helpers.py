from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from accelerate.accelerator import get_state_dict_offloaded_model
from compressed_tensors.quantization.utils import module_type
from compressed_tensors.utils import align_module_device
from tqdm import tqdm

from llmcompressor.modifiers import Modifier
from llmcompressor.pytorch.utils import get_linear_layers
from llmcompressor.pytorch.utils.helpers import tensor_sparsity

__ALL__ = [
    "tensor_follows_mask_structure",
    "infer_sparsity_structure_from_modifiers",
    "infer_sparsity_structure_from_model",
    "infer_sparse_targets_and_ignores",
    "is_sparse_compression_target",
]


def tensor_follows_mask_structure(tensor: torch.Tensor, mask: str = "2:4") -> bool:
    """
    :param tensor: tensor to check
    :param mask: mask structure to check for, in the format "n:m", also accepts
        "unstructured" as a valid mask structure
    :return: True if the tensor follows the mask structure, False otherwise.
        Note, some weights can incidentally be zero, so we check for
        atleast n zeros in each chunk of size m
    """

    if mask.lower().strip() == "unstructured":
        return True

    n, m = tuple(map(int, mask.split(":")))

    # If n or m is 0, then the tensor follows the mask structure
    if n == 0 or m == 0:
        return True
    # Reshape the tensor into chunks of size m
    tensor = tensor.view(-1, m)

    # Count the number of zeros in each chunk
    zero_counts = (tensor == 0).sum(dim=1)

    # Check if the number of zeros in each chunk atleast n
    # Greater than sign is needed as some weights can incidentally
    # be zero
    return torch.all(zero_counts >= n).item()


def infer_sparsity_structure_from_modifiers(
    modifiers: List[Modifier],  # noqa E501
) -> Optional[str]:
    """
    Determines the sparsity structure, if any exists, given the list of modifiers.

    :param modifiers: List of modifier instances.
    :return: sparsity structure as a string or None.
    """
    for modifier in modifiers:
        if hasattr(modifier, "mask_structure"):
            return modifier.mask_structure
    return None


def infer_sparsity_structure_from_model(model: torch.nn.Module) -> Optional[str]:
    """
    Determines the sparsity structure, if any exists, given the model

    :param model: model to check for sparsity structure
    :return: sparsity structure as a string or None
    """

    # check for the common sparsity structures
    structures = {"2:4"}
    for sparsity_structure in structures:
        linear_modules = get_linear_layers(model)
        offloaded_params = get_state_dict_offloaded_model(model)

        linear_modules_with_sparsity_structure = [
            tensor_follows_mask_structure(offloaded_params[f"{name}.weight"])
            for name in tqdm(
                linear_modules.keys(),
                desc="Checking whether model follows "
                f"{sparsity_structure} sparsity structure",
            )
        ]
        # if the majority of the linear modules follow the sparsity structure
        # we can assume that the model follows the sparsity structure
        # (taking into consideration the fact that some Linear layers like the
        # embedding layer might not be sparse)
        if (
            sum(linear_modules_with_sparsity_structure)
            > len(linear_modules_with_sparsity_structure) * 0.8
        ):
            return sparsity_structure

    return None


def infer_sparse_targets_and_ignores(
    model: torch.nn.Module,
    sparsity_structure: str,
    sparsity_threshold: float,
) -> Tuple[List[str], List[str]]:
    """
    Infers the target and ignore layers in the given model
    to be used for sparsity compression

    :param model: model to check
    :param sparsity_structure: sparsity structure to check against
    :param sparsity_threshold: threshold for sparsity
    :return: tuple of target and ignore layers
    """

    exhaustive_targets, exhaustive_ignore = _get_sparse_targets_ignore_dicts(
        module=model,
        sparsity_structure=sparsity_structure,
        sparsity_threshold=sparsity_threshold,
    )

    return _reduce_targets_and_ignores_into_lists(
        exhaustive_targets=exhaustive_targets,
        exhaustive_ignore=exhaustive_ignore,
    )


def is_sparse_compression_target(
    module: torch.nn.Module, sparsity_threshold: float, sparsity_structure: str
) -> bool:
    """
    :param module: module to check
    :param sparsity_threshold: threshold for sparsity
    :param sparsity_structure: sparsity structure to check against
    :return: whether or not the module is a target for sparsity compression,
        i.e True if it is sparse and follows the sparsity structure, else False
    """
    with align_module_device(module):
        result = (
            hasattr(module, "weight")
            and tensor_sparsity(module.weight) >= sparsity_threshold
            and tensor_follows_mask_structure(
                tensor=module.weight, mask=sparsity_structure
            )
        )

    return result


def _get_sparse_targets_ignore_dicts(
    module: torch.nn.Module, sparsity_structure: str, sparsity_threshold: float
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get sparse targets and ignore dictionaries

    :param module: module to check
    :param sparsity_structure: sparsity structure to check against
    :param sparsity_threshold: threshold for sparsity
    :return: tuple of exhaustive targets and ignore dictionaries
    """
    exhaustive_targets = defaultdict(list)
    exhaustive_ignore = defaultdict(list)

    for name, submodule in module.named_modules():
        submodule_type = module_type(submodule)
        is_target = is_sparse_compression_target(
            module=submodule,
            sparsity_threshold=sparsity_threshold,
            sparsity_structure=sparsity_structure,
        )
        target_dict = exhaustive_targets if is_target else exhaustive_ignore
        target_dict[submodule_type].append(name)
    return exhaustive_targets, exhaustive_ignore


def _reduce_targets_and_ignores_into_lists(
    exhaustive_targets: Dict[str, List[str]], exhaustive_ignore: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    """
    Reduces the targets and ignores dictionaries into lists

    :param exhaustive_targets: dictionary of target layers, must contain all
        targetted layers in the model
    :param exhaustive_ignore: dictionary of ignore layers, must contain all
        ignored layers in the model
    :return: tuple of reduced target and ignore layers
    """

    targets, ignore = [], []
    all_submodule_types = set(exhaustive_targets.keys()).union(
        set(exhaustive_ignore.keys())
    )
    for submodule_type in all_submodule_types:
        curr_targets = exhaustive_targets.get(submodule_type, [])
        curr_ignores = exhaustive_ignore.get(submodule_type, [])

        if len(curr_targets) >= len(curr_ignores):
            targets.append(submodule_type)
            ignore.extend(curr_ignores)
        elif len(curr_targets) > 0:
            # only add ignore layers if
            # they are targetted
            targets.extend(curr_targets)
            ignore.extend(curr_ignores)
    return targets, ignore
