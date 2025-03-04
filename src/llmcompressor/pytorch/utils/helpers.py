"""
Utility / helper functions
"""

import functools
import inspect
import os
import random
from typing import Any, Dict, Iterable, List, Mapping, OrderedDict, Tuple, Union

import numpy
import torch
from torch import Tensor
from torch.nn import Linear, Module

try:
    quant_err = None
except Exception as _err:
    quant_err = _err
    QuantWrapper = None
    QATLinear = None
    QATConv2d = None


__all__ = [
    "tensors_to_device",
    "tensors_to_precision",
    "tensors_module_forward",
    "tensor_sparsity",
    "get_linear_layers",
    "get_quantized_layers",
    "set_deterministic_seeds",
    "torch_distributed_zero_first",
    "thin_model_from_checkpoint",
    "MEMORY_BOUNDED",
    "memory_aware_threshold",
    "detach",
    "adjust_quantization_for_onnx_export",
    "get_dependency_order",
    "pseudo_quantize_tensor",
    "pseudo_dequantize_linear",
    "tensor_forward_with_input_args",
    "sanitize_kwargs_for_module",
]


##############################
#
# pytorch tensor helper functions
#
##############################


def tensors_to_device(
    tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]], device: str
) -> Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]:
    """
    Default function for putting a tensor or collection of tensors to the proper device.
    Returns the tensor references after being placed on the proper device.

    Supported use cases:
        - single tensor
        - Dictionary of single tensors
        - Dictionary of iterable of tensors
        - Dictionary of dictionary of tensors
        - Iterable of single tensors
        - Iterable of iterable of tensors
        - Iterable of dictionary of tensors

    :param tensors: the tensors or collection of tensors to put onto a device
    :param device: the string representing the device to put the tensors on,
        ex: 'cpu', 'cuda', 'cuda:1'
    :return: the tensors or collection of tensors after being placed on the device
    """
    if isinstance(tensors, Tensor):
        return tensors.to(device)

    if isinstance(tensors, OrderedDict):
        return OrderedDict(
            [(key, tensors_to_device(tens, device)) for key, tens in tensors.items()]
        )

    if isinstance(tensors, Mapping):
        return {key: tensors_to_device(tens, device) for key, tens in tensors.items()}

    if isinstance(tensors, tuple):
        return tuple(tensors_to_device(tens, device) for tens in tensors)

    if isinstance(tensors, Iterable):
        return [tensors_to_device(tens, device) for tens in tensors]

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


def tensors_to_precision(
    tensors: Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]], full_precision: bool
) -> Union[Tensor, Iterable[Tensor], Dict[Any, Tensor]]:
    """
    :param tensors: the tensors to change the precision of
    :param full_precision: True for full precision (float 32) and
        False for half (float 16)
    :return: the tensors converted to the desired precision
    """
    if isinstance(tensors, Tensor):
        return tensors.float() if full_precision else tensors.half()

    if isinstance(tensors, Mapping):
        return {
            key: tensors_to_precision(tens, full_precision)
            for key, tens in tensors.items()
        }

    if isinstance(tensors, tuple):
        return tuple(tensors_to_precision(tens, full_precision) for tens in tensors)

    if isinstance(tensors, Iterable):
        return [tensors_to_precision(tens, full_precision) for tens in tensors]

    raise ValueError(
        "unrecognized type for tensors given of {}".format(tensors.__class__.__name__)
    )


# used by calibration function, TODO: remove with data pipelines
def tensors_module_forward(
    tensors: Union[Tensor, Iterable[Tensor], Mapping[Any, Tensor]],
    module: Module,
    check_feat_lab_inp: bool = True,
) -> Any:
    """
    Default function for calling into a model with data for a forward execution.
    Returns the model result.
    Note, if an iterable the features to be passed into the model are considered
    to be at index 0 and other indices are for labels.

    Supported use cases: single tensor,
    iterable with first tensor taken as the features to pass into the model

    :param tensors: the data to be passed into the model, if an iterable the features
        to be passed into the model are considered to be at index 0 and other indices
        are for labels
    :param module: the module to pass the data into
    :param check_feat_lab_inp: True to check if the incoming tensors looks like
        it's made up of features and labels ie a tuple or list with 2 items
        (typical output from a data loader) and will call into the model with just
        the first element assuming it's the features False to not check
    :return: the result of calling into the model for a forward pass
    """
    if (
        (isinstance(tensors, tuple) or isinstance(tensors, List))
        and len(tensors) == 2
        and check_feat_lab_inp
    ):
        # assume if this is a list or tuple of 2 items that it is made up of
        # (features, labels) pass the features into a recursive call for the model
        return tensors_module_forward(tensors[0], module, check_feat_lab_inp=False)

    if isinstance(tensors, Tensor):
        return module(tensors)

    if isinstance(tensors, Mapping):
        return module(**tensors)

    if isinstance(tensors, Iterable):
        return module(*tensors)

    raise ValueError(
        "unrecognized type for data given of {}".format(tensors.__class__.__name__)
    )


def tensor_sparsity(
    tens: Tensor, dim: Union[None, int, List[int], Tuple[int, ...]] = None
) -> Tensor:
    """
    :param tens: the tensor to calculate the sparsity for
    :param dim: the dimension(s) to split the calculations over;
        ex, can split over batch, channels, or combos
    :return: the sparsity of the input tens, ie the fraction of numbers that are zero
    """
    if dim is None:
        zeros = (tens.cpu() == 0).sum()
        total = tens.numel()

        return zeros.float() / float(total)

    if isinstance(dim, int):
        dim = [dim]

    if max(dim) >= len(tens.shape):
        raise ValueError(
            "Unsupported dim given of {} in {} for tensor shape {}".format(
                max(dim), dim, tens.shape
            )
        )

    sum_dims = [ind for ind in range(len(tens.shape)) if ind not in dim]
    zeros = (tens == 0).sum(dim=sum_dims) if sum_dims else tens == 0
    total = numpy.prod(
        [tens.shape[ind] for ind in range(len(tens.shape)) if ind not in dim]
    )

    permute_order = sorted(
        ((d, len(dim) - i - 1) for i, d in enumerate(dim)), reverse=True
    )
    permute = [d[1] for d in permute_order]

    if permute != [i for i in range(len(permute))]:
        # need to permute to get desired dimensions at the front
        zeros = zeros.permute(*permute).contiguous()

    return zeros.float() / float(total)


def tensor_density(tens: Tensor, dim: Union[None, int, Iterable[int]] = None) -> Tensor:
    """
    :param tens: the tensor to calculate the density for
    :param dim: the dimension(s) to split the calculations over; ex, can split over
        batch, channels, or combos
    :return: the density of the input tens, ie the fraction of numbers that are non zero
    """
    density = (tensor_sparsity(tens, dim) - 1.0) * -1.0

    return density


def tensor_sample(
    tens: Tensor,
    sample_size: int,
    dim: Union[None, int, List[int], Tuple[int, ...]] = None,
) -> Tensor:
    """
    :param tens: the tensor to grab samples from
    :param sample_size: the number of samples to grab overall if dim is not supplied
        or per each dim if it is
    :param dim: the dimension(s) to split the samples over;
        ex, can split over batch, channels, or combos
    :return: the sampled tensor
    """
    if sample_size < 1:
        raise ValueError("improper sample size given of {}".format(sample_size))

    if dim is None:
        indices = tens.new_zeros((sample_size,)).long().random_(0, tens.numel())
        samples = tens.view(-1)[indices]

        return samples

    if isinstance(dim, int):
        dim = [dim]

    if max(dim) >= len(tens.shape):
        raise ValueError(
            "Unsupported dim given of {} in {} for tensor shape {}".format(
                max(dim), dim, tens.shape
            )
        )

    if dim != [ind for ind in range(len(dim))]:
        # put the desired dimension(s) at the front to sample from
        tens = tens.permute(
            *dim, *[ind for ind in range(len(tens.shape)) if ind not in dim]
        )
        dim = [ind for ind in range(len(dim))]

    if not tens.is_contiguous():
        tens = tens.contiguous()

    num_indices = int(numpy.prod([tens.shape[ind] for ind in range(len(dim))]))
    elem_per_ind = int(
        numpy.prod([tens.shape[ind] for ind in range(len(dim), len(tens.shape))])
    )
    # create a new tensor with offsets set for each of our elements that we are indexing
    indices = tens.new_tensor(
        [ind * elem_per_ind for ind in range(num_indices)], dtype=torch.long
    ).unsqueeze(1)
    # now broadcast it across to the total number of elements we should end with
    indices = indices * tens.new_ones((num_indices, sample_size), dtype=torch.long)
    # finally add in a random number within the available range per index
    indices += tens.new_zeros((num_indices, sample_size), dtype=torch.long).random_(
        0, elem_per_ind
    )
    # get our samples
    samples = tens.view(-1)[indices.view(-1)]
    # reshape for the proper dimension
    samples = samples.view(*(tens.shape[ind] for ind in dim), sample_size)

    return samples


def tensor_list_sparsity(tensors: List[Tensor]) -> float:
    """
    :param tensors: the list of tensors to calculate the sparsity for
    :return: the total sparsity of all tensors in the list
    """
    zeros = 0
    numel = 0
    for tensor in tensors:
        zeros += (tensor == 0).sum().item()
        numel += tensor.numel()
    return float(zeros) / float(numel)


def mask_difference(old_mask: Tensor, new_mask: Tensor) -> Tensor:
    """
    :param old_mask: the old mask to compare against for calculating the difference
    :param new_mask: the new mask to compare with for calculating the difference
    :return: a tensor representing the change from the old_mask to the new_mask
             specifically values returned as 1.0 are newly unmasked (0.0 => 1.0)
             values returned as -1.0 are newly masked (1.0 => 0.0)
             values returned as 0.0 had no change in (0.0 => 0.0 or 1.0 => 1.0)
    """
    newly_masked = ((old_mask != new_mask) & (new_mask == 0.0)).type(old_mask.type())
    newly_unmasked = ((old_mask != new_mask) & (new_mask == 1.0)).type(old_mask.type())

    return -1.0 * newly_masked + newly_unmasked


def sanitize_kwargs_for_module(
    kwargs: Dict[str, Any], module: Module
) -> Dict[str, Any]:
    """
    Sanitize the kwargs for a Module by removing any keys that are not
    in the signature of the forward method.
    :param kwargs: the kwargs to sanitize
    :param module: the Module to sanitize the kwargs for
    :return: the sanitized kwargs for the callable object
    """
    if not isinstance(kwargs, dict):
        raise TypeError(f"Expected a dictionary as kwargs, but got {kwargs}")

    allowed_params = inspect.signature(module.forward).parameters
    return {key: value for key, value in kwargs.items() if key in allowed_params}


def tensor_forward_with_input_args(
    module: Module, inputs: Tensor, input_kwargs: Dict[str, Any]
) -> Tensor:
    """
    Forward the given inputs through the given module with the given input_kwargs.
    This function is a wrapper around tensors_module_forward that ensures that the
    input_kwargs are sanitized and passed to the module as keyword arguments during
    the forward pass.
    :param module: the module to forward the inputs through
    :param inputs: the inputs to forward through the module
    :param input_kwargs: the keyword arguments to pass to the
        module during the forward pass
    :return: the output of the module after forwarding the inputs through it
    """
    inputs = inputs.to(next(module.parameters()).device)
    input_kwargs = sanitize_kwargs_for_module(input_kwargs, module)

    return tensors_module_forward(inputs, functools.partial(module, **input_kwargs))


##############################
#
# pytorch module helper functions
#
##############################


def get_linear_layers(module: Module) -> Dict[str, Module]:
    """
    :param module: the module to grab all linear layers for
    :return: a list of all linear layers in the module
    """
    return {
        name: mod for name, mod in module.named_modules() if isinstance(mod, Linear)
    }


def get_quantized_layers(module: Module) -> List[Tuple[str, Module]]:
    """
    :param module: the module to get the quantized layers from
    :return: a list containing the names and modules of the quantized layers
        (Embedding, Linear, Conv2d, Conv3d)
    """

    quantized_layers = []
    for name, mod in module.named_modules():
        if hasattr(mod, "quantization_scheme"):
            weight_scheme = getattr(mod.quantization_scheme, "weights", None)
            if weight_scheme is not None and hasattr(mod, "weight"):
                quantized_layers.append((name, mod))

    return quantized_layers


def set_deterministic_seeds(seed: int = 0):
    """
    Manually seeds the numpy, random, and torch packages.
    Also sets torch.backends.cudnn.deterministic to True
    :param seed: the manual seed to use. Default is 0
    """
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@contextmanager
def torch_distributed_zero_first(local_rank: Optional[int]):
    """
    Decorator to make all processes in distributed training wait for each
    local 0 ranked process to do something.
    :param local_rank: the local rank of this process
    """
    if local_rank is not None and local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def thin_model_from_checkpoint(model: Module, state_dict: Dict[str, Any]):
    """
    Updates any Linear/Conv/BN layers in the given model to match their
    respective shapes in the given state dict. Purpose of compatibility
    when loading weight for a model from a checkpoint of the same architecture
    but with potentially structured thinning applied. Note that this function
    has no guarantees on accuracy, will only resize model parameters for
    loading compatibility. All adjustments done in place

    :param model: model to potentially adjust parameter shapes of
    :param state_dict: state dict to infer parameter shapes from
    """
    first_thinned = True
    for param_name, checkpoint_tens in state_dict.items():
        if not param_name.endswith(".weight"):
            continue  # only deal with weight params of modules
        layer_name = param_name[:-7]
        layer = get_layer(layer_name, model)

        if not hasattr(layer, "weight") or (
            layer.weight.shape == checkpoint_tens.shape
        ):
            continue  # skip if there is no update to shape

        # quick check that target layer is some flavor of FC/Conv/BN
        layer_type = layer.__class__.__name__
        if not (
            "Linear" not in layer_type
            or "Conv" not in layer_type
            or ("BatchNorm" not in layer_type)
        ):
            continue

        orig_shape = layer.weight.shape
        target_shape = checkpoint_tens.shape

        # update weight param + grad
        if len(target_shape) > 1:
            layer.weight.data = layer.weight.data[
                : target_shape[0], : target_shape[1], ...
            ]
            if layer.weight.grad is not None:
                layer.weight.grad = layer.weight.grad[
                    : target_shape[0], : target_shape[1], ...
                ]
        else:
            layer.weight.data = layer.weight.data[: target_shape[0]]
            if layer.weight.grad is not None:
                layer.weight.grad = layer.weight.grad[: target_shape[0]]

        # update bias param + grad
        if hasattr(layer, "bias") and layer.bias is not None:
            # target output channels should be the first dim of target shape
            layer.bias.data = layer.bias.data[: target_shape[0]]
            if layer.bias.grad is not None:
                layer.bias.grad = layer.bias.grad[: target_shape[0]]

        # update layer attributes
        if "BatchNorm" in layer_type:
            if hasattr(layer, "num_features"):
                layer.num_features = layer.weight.size(0)
            # BN running mean and var are not stored as Parameters
            if hasattr(layer, "running_mean"):
                layer.running_mean = torch.zeros_like(layer.running_mean)[
                    : target_shape[0]
                ]
            if hasattr(layer, "running_var"):
                layer.running_var = torch.zeros_like(layer.running_var)[
                    : target_shape[0]
                ]

        if "Linear" in layer_type:
            if hasattr(layer, "out_features"):
                layer.out_features = layer.weight.shape[0]
            if hasattr(layer, "in_features"):
                layer.in_features = layer.weight.shape[1]

        if "Conv" in layer_type:
            if hasattr(layer, "out_channels"):
                layer.out_channels = layer.weight.shape[0]
            if hasattr(layer, "in_channels"):
                layer.in_channels = layer.weight.shape[1]
            if hasattr(layer, "groups") and layer.groups > 1:
                layer.groups = layer.weight.shape[0] // layer.weight.shape[1]

        if first_thinned:
            logger.info(
                "Thinning module layers for compatibility with given state dict:"
            )
            first_thinned = False
        logger.info(
            f"Thinned layer {layer_name} from shape {orig_shape} to "
            f"{layer.weight.shape}"
        )


##############################
#
# misc pytorch helper functions
#
##############################


MEMORY_BOUNDED = "MEMORY_BOUNDED"


def memory_aware_threshold(tensor: torch.Tensor, idx: int) -> Tensor:
    """
    Finds a threshold at the lookup idx in the most efficient way with available
    resources. Will be phased out when GPU-memory overhead of torch.sort reduces,
    or when torch.kthvalue becomes faster than torch.sort.

    :param tensor: A tensor to find a k-th smallest value in, where k=idx+1
    :param idx: A lookup index
    :return: k-th smallest value from the given tensor, where k=idx+1
    """
    try:
        if (
            MEMORY_BOUNDED in os.environ
            and os.environ[MEMORY_BOUNDED].lower() == "true"
        ):
            return torch.kthvalue(tensor.reshape(-1), idx + 1)[0]
        else:
            return torch.sort(tensor.reshape(-1))[0][idx]
    except RuntimeError:
        logger.warning(
            "Finding threshold from sparsity failed due to lack of memory, "
            "will attempt to recover. Consider setting env variable "
            f"{MEMORY_BOUNDED}=True in future runs."
        )
        torch.cuda.empty_cache()
        os.environ[MEMORY_BOUNDED] = "True"
        return torch.kthvalue(tensor.view(-1), idx + 1)[0]


def detach(x: Union[torch.Tensor, List, Tuple]):
    if isinstance(x, torch.Tensor):
        return x.detach()
    elif isinstance(x, List):
        return [detach(e) for e in x]
    elif isinstance(x, Tuple):
        return tuple([detach(e) for e in x])
    else:
        raise ValueError("Unexpected type to detach")


def adjust_quantization_for_onnx_export(module: torch.nn.Module) -> torch.nn.Module:
    # supported pytorch ranges are int8 or uint8
    allowed_ranges = [(0, 127), (0, 255), (-128, 127)]
    fake_quant_modules = [
        m for m in module.modules() if m.__class__.__name__ == "FakeQuantize"
    ]

    if _PARSED_TORCH_VERSION >= version.parse("1.12"):
        for quant in fake_quant_modules:
            # original ranges preserved in quant.quant_min and quant.quant_max
            quant_range = (
                quant.activation_post_process.quant_min,
                quant.activation_post_process.quant_max,
            )
            if quant_range not in allowed_ranges:
                if quant_range[0] < 0:  # convert signed range to int8
                    quant.activation_post_process.quant_min = -128
                    quant.activation_post_process.quant_max = 127
                else:  # convert unsigned range to uint8
                    quant.activation_post_process.quant_min = 0
                    quant.activation_post_process.quant_max = 255
            # don't update observer since ranges are artificially modified
            quant.observer_enabled[0] = 0

    else:  # backwards compatibility for torch <= 1.11
        for quant in fake_quant_modules:
            quant_range = (quant.quant_min, quant.quant_max)
            if quant_range not in allowed_ranges:
                if quant_range[0] < 0:  # convert signed range to int8
                    quant.quant_min = -128
                    quant.quant_max = 127
                else:  # convert unsigned range to uint8
                    quant.quant_min = 0
                    quant.quant_max = 255
            # don't update observer since ranges are artificially modified
            quant.observer_enabled[0] = 0


def get_dependency_order(
    layer: Module, subset: Dict, an_input: Tensor, **kwargs
) -> List[str]:
    """
    Get a list of a subset of modules in layer ordered by execution order, which honors
    the dependencies in the graph

    :param layer: pytorch module to calculate dependencies for
    :param subset: subset of modules in the layer to include in the ordering
    :param an_input: example input to pass through the layer forward pass, used to
        determine execution order

    :return: list of module names in execution order
    """
    order = []

    def exe_input(name):
        def _exe_input(_, inp, out):
            if name in subset:
                order.append(name)

        return _exe_input

    # register a hook for each module of interest, will be triggered in exeuction order
    handles = [subset[name].register_forward_hook(exe_input(name)) for name in subset]
    layer(an_input, **kwargs)
    for h in handles:
        h.remove()
    return order


def swap_modules(
    module: torch.nn.Module, submodule_name: str, submodule_to_replace: torch.nn.Module
) -> torch.nn.Module:
    """
    Iteratively unfold the submodules of the module according to the submodule_name
    to eventually replace the leaf submodule (accessed from the module through the
    submodule_name) with the submodule_to_replace.

    E.g
    ```
    swap_modules(module=Model,
                 module_name="layers.0.sublayer",
                 module_to_replace=ReplaceModule
                 )
    ```
    this will iteratively traverse through the submodules
    'layers' -> '0' -> to eventually replace 'sublayer' with ReplaceModule

    :param module: the module to replace with the module_to_replace
    :param submodule_name: the name of the module to replace
    :param submodule_to_replace: the module to replace the module with
    :return: the replaced module
    """
    parent = module
    sections = submodule_name.split(".")

    for sec in sections[:-1]:
        parent = parent.__getattr__(sec)

    cur = parent.__getattr__(sections[-1])
    parent.__setattr__(sections[-1], submodule_to_replace)

    return cur


def pseudo_quantize_tensor(
    w: torch.Tensor, symmetric: bool = False, bit_width: int = 8, group_size: int = -1
):
    org_w_shape = w.shape
    if group_size > 0:
        assert org_w_shape[-1] % group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({group_size})!"
        w = w.reshape(-1, group_size)
    assert w.dim() == 2
    assert torch.isnan(w).sum() == 0

    # zero point quantization
    if not symmetric:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**bit_width - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
        zeros = (zeros - 2**(bit_width-1)).view(org_w_shape[0], -1) 
    else:
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (bit_width - 1) - 1
        min_int = -(2 ** (bit_width - 1))
        scales = max_val / max_int
        zeros = None
        w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    scales = scales.view(org_w_shape[0], -1)
    w = w.reshape(org_w_shape)

    return w, scales, zeros


def pseudo_dequantize_linear(
    w: torch.Tensor,
    scales: torch.Tensor,
    zeros: Optional[torch.Tensor] = None,
    symmetric: bool = False,
):
    # get repeated count
    repeat_count = w.weight.data.shape[-1] // scales.shape[-1]
    scales = scales.repeat(1, repeat_count).reshape(w.weight.data.shape)

    # dequantize
    if not symmetric:
        zeros = zeros.repeat(1, repeat_count).reshape(w.weight.data.shape)
        w = (w.weight.data - zeros) * scales
    else:
        w = w.weight.data * scales

    return w
