import contextlib
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import tqdm
from compressed_tensors.quantization import find_name_or_class_matches
from compressed_tensors.utils import get_execution_device
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader

from llmcompressor.modifiers.utils.pytorch_helpers import apply_pad_mask_to_batch
from llmcompressor.pipelines.cache import IntermediatesCache
from llmcompressor.pytorch.utils.helpers import tensors_to_device
from llmcompressor.utils.helpers import calibration_forward_context

__all__ = ["match_modules", "capture_first_layer_intermediates", "to_next_layer_kwargs"]


def match_modules(model: Module, target_names: List[str]) -> List[Module]:
    """
    Find all submodules which match the `target_names` and sort them by name

    :param model: model to search for submodules in
    :param target_names: patterns of submodule names to match
    :return: list of submodules
    """
    names_layers = [
        (name, module)
        for name, module in model.named_modules()
        if find_name_or_class_matches(name, module, target_names)
    ]

    names_layers = sorted(names_layers, key=lambda name_layer: name_layer[0])
    return [layer for _name, layer in names_layers]


def capture_first_layer_intermediates(
    model: Module,
    layers: List[Module],
    dataloader: DataLoader,
    mask_padding: bool = True,
) -> IntermediatesCache:
    """
    Captures the intermediate activations directly before the first model layer.
    This is meant to capture any model preprocessing before model layers are executed

    Note that if any modules compressed prior to the execution of the first layer, the
    compression error induced by compressing those modules will not be propagated to
    subsequent activations, as they would be for modules which are compressed within
    a layer

    :param model: model containing layers
    :param layers: list of layer submodules in the model
    :param dataloader: dataloader of calibration inputs
    :param mask_padding: zero out padding tokens if True. This affects modifiers such as
        GPTQ and SparseGPT
    """
    model_device = get_execution_device(model)
    intermediates = IntermediatesCache.empty(len(dataloader), torch.device("cpu"))
    first_layer = layers[0]
    signature = inspect.signature(first_layer.forward)

    with calibration_forward_context(model), early_stop_hook(first_layer):
        desc = "Preparing intermediates cache"
        for batch_index, batch in enumerate(tqdm.tqdm(dataloader, desc=desc)):
            batch = apply_pad_mask_to_batch(batch) if mask_padding else batch
            batch = tensors_to_device(batch, model_device)

            try:
                model(**batch)
            except EarlyStopException as exception:
                layer_args = args_to_kwargs(exception._args, signature)
                assert not set(layer_args.keys()) & set(exception._kwargs.keys())
                layer_args.update(exception._kwargs)

                intermediates.update(batch_index, layer_args)
            else:
                raise ValueError(
                    "Attempted to capture first layer intermediates, but "
                    "EarlyStopException was not raised"
                )

    return intermediates


def to_next_layer_kwargs(args: Tuple[Any, ...], next_layer: Module) -> Dict[str, Any]:
    """
    Convert a list of arguments to a dictionary of keyword arguments which match the
    next layer's function signature

    :param args: list of argument values
    :param next_layer: the next layer whose function signature must be matched
    :return: dictionary mapping function signature keywords to argument values
    """
    signature = inspect.signature(next_layer.forward)
    return args_to_kwargs(args, signature)


def args_to_kwargs(
    args: Tuple[Any, ...], signature: inspect.Signature
) -> Dict[str, Any]:
    return {name: arg for name, arg in zip(signature.parameters.keys(), args)}


@contextlib.contextmanager
def early_stop_hook(module: Module):
    def trigger_early_stop_fn(module, args, kwargs):
        raise EarlyStopException(_args=args, _kwargs=kwargs)

    handle = module.register_forward_pre_hook(trigger_early_stop_fn, with_kwargs=True)

    try:
        yield
    finally:
        handle.remove()


@dataclass
class EarlyStopException(Exception):
    """
    Note: this is exception different from the exception defined in
    llmcompressor.modifiers.utils.pytorch_helpers, and will eventually replace

    Attribute names `args` and `kwargs` are reserved for `dataclass`
    """

    _args: Tuple[Any, ...]
    _kwargs: Dict[str, Any]
