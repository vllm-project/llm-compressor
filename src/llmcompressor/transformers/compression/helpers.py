from typing import Dict, List, Optional, Union

import psutil
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.accelerator import get_state_dict_offloaded_model
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
    "custom_offload_device_map",
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


def hessian_memory_requirements(model: torch.nn.Module) -> int:
    """
    Determines the number of bytes needed to store Hessian data for a single
    transformer layer in model. This is used for reserving memory for GPTQ
    quantization

    :param model: model to calculate requirements for
    :return: number of bytes required to reserve for GPTQ on a single layer
    """
    transformer_layers = get_layers(get_no_split_params(model), model)
    total_hessian_elems = {}
    max_column_size = {}
    for no_split_name, no_split_layer in transformer_layers.items():
        total_hessian_elems[no_split_name] = 0
        max_column_size[no_split_name] = 0
        for name, module in no_split_layer.named_modules():
            if isinstance(module, Linear):
                for param in module.parameters():
                    column_size = param.shape[1]
                    total_hessian_elems[no_split_name] += column_size * column_size
                    if column_size > max_column_size[no_split_name]:
                        # max extra memory for inverse calculation
                        max_column_size[no_split_name] = column_size

    max_total_hessian_elems = max(total_hessian_elems.values())
    overall_max_column_size = max(max_column_size.values())
    bytes_per_weight = 32 // 8  # hessians are float32
    inverse_reserved = overall_max_column_size * overall_max_column_size
    return (max_total_hessian_elems + inverse_reserved) * bytes_per_weight


def quantization_memory_requirement(model: torch.nn.Module) -> int:
    """
    Determines the max number of bytes needed to store quantization scale and zp data

    :param model: model to calculate requirements for
    :return: number of bytes required to reserve for quantization
    """

    total_elements = 0
    for _, module in model.named_modules():
        if isinstance(module, Linear):
            for param in module.parameters():
                # assume the max of group 128 and static scale/zp
                # TODO: base this on the recipe instead instead of assuming max

                # potentially just bias term
                max_quant_shape = param.shape[0] // 128

                if len(param.size()) > 1:  # weights
                    max_quant_shape *= param.shape[1]

                total_elements += max_quant_shape * 4

    bytes_ratio = 32 // 16  # assuming float16
    return total_elements * bytes_ratio


def custom_offload_device_map(
    model_stub: str,
    max_memory_per_gpu: Union[str, int],
    num_gpus: int = 1,
    **model_kwargs,
) -> Dict[Union[int, str], Union[int, str]]:
    """
    Calculates the optimal gpu mappings for model_stub stored as torch_dtype, where
    each GPU is restricted to allocating a specific amount of memory.

    :param model_stub: local path or HF stub to calculate mapping for
    :param max_memory_per_gpu: Max memory to allocate on each GPU, as either a string
        such as "10GB" or an integer number of bytes
    :param num_gpus: number of gpus to utilize
    :param model_kwargs: keyword arguments to pass to model initializer
    :return: memory mapping for layers of model_stub to be passed to from_pretrained()
    """
    max_cpu_memory = psutil.virtual_memory().available
    memory_limits = {device: max_memory_per_gpu for device in range(num_gpus)}
    memory_limits["cpu"] = max_cpu_memory

    device_map = {}
    with init_empty_weights():
        dummy_model = AutoModelForCausalLM.from_pretrained(model_stub, **model_kwargs)
        device_map = infer_auto_device_map(
            dummy_model,
            max_memory=memory_limits,
            no_split_module_classes=dummy_model._no_split_modules,
        )
        del dummy_model

    return device_map


def calculate_offload_device_map(
    model_stub: str,
    reserve_for_hessians=False,
    num_gpus: int = 1,
    torch_dtype: torch.dtype = torch.float16,
    **model_kwargs,
) -> Dict[Union[int, str], Union[int, str]]:
    """
    Calculates the optimal gpu mappings for model_stub stored as torch_dtype. Takes
    into account extra memory required for quantization and (optionally) GPTQ hessians

    :param model_stub: local path or HF stub to calculate mapping for
    :param reserve_for_hessians: whether to reserve memory for GPTQ
    :param num_gpus: number of gpus to utilize
    :param model_kwargs: keyword arguments to pass to model initializer
    :return: memory mapping for layers of model_stub to be passed to from_pretrained()
    """
    max_cpu_memory = psutil.virtual_memory().available
    max_gpu_memory = torch.cuda.mem_get_info(0)[0]
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        raise ValueError(
            f"Requested {num_gpus} GPUs but only {available_gpus} are available."
        )
    max_gpu_memory = [max_gpu_memory] * num_gpus

    device_map = {}
    with init_empty_weights():
        dummy_model = AutoModelForCausalLM.from_pretrained(
            model_stub, torch_dtype=torch_dtype, **model_kwargs
        )

        reserved_memory = 0
        if reserve_for_hessians:
            reserved_memory = hessian_memory_requirements(dummy_model)
        reserved_memory += quantization_memory_requirement(dummy_model)

        memory_limits = {
            idx: (max_memory - reserved_memory)
            for idx, max_memory in enumerate(max_gpu_memory)
        }
        memory_limits["cpu"] = max_cpu_memory

        device_map = infer_auto_device_map(
            dummy_model,
            max_memory=memory_limits,
            no_split_module_classes=dummy_model._no_split_modules,
        )
        del dummy_model

    return device_map
