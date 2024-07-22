from typing import List, Optional

import psutil
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from torch.nn.modules import Linear
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from llmcompressor.pytorch.utils import get_linear_layers
from llmcompressor.utils.pytorch import get_layers, get_no_split_params

__ALL__ = [
    "tensor_follows_mask_structure",
    "infer_sparsity_structure_from_stage_modifiers",
    "infer_sparsity_structure_from_model",
    "hessian_memory_requirements",
    "calculate_offload_device_map",
]


def tensor_follows_mask_structure(tensor, mask: str = "2:4") -> bool:
    """
    :param tensor: tensor to check
    :param mask: mask structure to check for, in the format "n:m"
    :return: True if the tensor follows the mask structure, False otherwise.
        Note, some weights can incidentally be zero, so we check for
        atleast n zeros in each chunk of size m
    """

    n, m = tuple(map(int, mask.split(":")))
    # Reshape the tensor into chunks of size m
    tensor = tensor.view(-1, m)

    # Count the number of zeros in each chunk
    zero_counts = (tensor == 0).sum(dim=1)

    # Check if the number of zeros in each chunk atleast n
    # Greater than sign is needed as some weights can incidentally
    # be zero
    return torch.all(zero_counts >= n).item()


def infer_sparsity_structure_from_stage_modifiers(
    stage_modifiers: List["StageModifier"],  # noqa E501
) -> Optional[str]:
    """
    Determines the sparsity structure, if any exists, given the
    list of stage modifiers

    :param stage_modifiers: non-empty list of stage modifiers
    :return: sparsity structure as a string or None
    """
    for stage in stage_modifiers:
        if stage.applied:
            for modifier in stage.modifiers:
                if hasattr(modifier, "mask_structure"):
                    sparsity_structure = modifier.mask_structure
                    return sparsity_structure
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
        linear_modules_with_sparsity_structure = [
            tensor_follows_mask_structure(layer.weight)
            for layer in tqdm(
                linear_modules.values(),
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


def hessian_memory_requirements(model: torch.nn.Module):
    transformer_layers = get_layers(get_no_split_params(model), model)
    single_layer = transformer_layers[list(transformer_layers.keys())[0]]
    total_hessian_elems = 0
    max_column_size = 0
    for _, module in single_layer.named_modules():
        if isinstance(module, Linear):
            for param in module.parameters():
                column_size = param.shape[1]
                total_hessian_elems += column_size * column_size
                if column_size > max_column_size:
                    # max extra memory for inverse calculation
                    max_column_size = column_size

    bytes_per_weight = 32 // 8  # hessians are float32
    inverse_reserved = max_column_size * max_column_size
    return (total_hessian_elems + inverse_reserved) * bytes_per_weight


def calculate_offload_device_map(model_stub: str, reserve_for_hessians=False, num_gpus: int = 1):
    max_cpu_memory = psutil.virtual_memory().available
    max_gpu_memory = list(torch.cuda.mem_get_info())
    available_gpus = len(max_gpu_memory)
    if available_gpus < num_gpus:
        raise ValueError("Requested {num_gpus} GPUs but only {available_gpus} are available.")
    max_gpu_memory = max_gpu_memory[:num_gpus]

    device_map = {}
    with init_empty_weights():
        dummy_model = AutoModelForCausalLM.from_pretrained(
            model_stub, torch_dtype=torch.float16
        )

        reserved_memory = 0
        if reserve_for_hessians:
            reserved_memory = hessian_memory_requirements(dummy_model)
        else:
            reserved_memory = 1e9

        memory_limits = {idx:(max_memory-reserved_memory) for idx,max_memory in enumerate(max_gpu_memory)}
        memory_limits["cpu"] = max_cpu_memory

        device_map = infer_auto_device_map(
            dummy_model,
            max_memory=memory_limits,
            no_split_module_classes=dummy_model._no_split_modules,
        )

    return device_map
