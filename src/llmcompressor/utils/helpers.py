"""
General utility helper functions.
Common functions for interfacing with python primitives and directories/files.
"""

import contextlib
import importlib
import re
from typing import Any, Iterable, Mapping, TypeVar

import torch
from compressed_tensors.offload import disable_onloading
from compressed_tensors.quantization import disable_quantization, enable_quantization
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.sentinel import Sentinel
from llmcompressor.utils import get_embeddings

__all__ = [
    "import_from_path",
    "disable_cache",
    "DisableQuantization",
    "eval_context",
    "disable_hf_kernels",
    "calibration_forward_context",
    "disable_lm_head",
    "numeric_sorted_names",
    "values_by_decoder_layer",
    "hasattr_fallbacks",
    "getattr_fallbacks",
    "hasitem_fallbacks",
    "getitem_fallbacks",
]

_T = TypeVar("_T")

# module names of decoder layers follow the `layers.N` convention, e.g.
# `model.layers.12.mlp.experts`
_DECODER_LAYER_PATTERN = re.compile(r"layers\.(\d+)")


def import_from_path(path: str) -> str:
    """
    Import the module and the name of the function/class separated by :
    Examples:
      path = "/path/to/file.py:func_or_class_name"
      path = "/path/to/file:focn"
      path = "path.to.file:focn"
    :param path: path including the file path and object name
    :return Function or class object
    """
    original_path, class_name = path.split(":")
    _path = original_path

    path = original_path.split(".py")[0]
    path = re.sub(r"/+", ".", path)
    try:
        module = importlib.import_module(path)
    except ImportError:
        raise ImportError(f"Cannot find module with path {_path}")

    try:
        return getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Cannot find {class_name} in {_path}")


@contextlib.contextmanager
def disable_cache(module: torch.nn.Module):
    """
    Temporarily disable the key-value cache for transformer models. Used to prevent
    excess memory use in one-shot cases where the model only performs the prefill
    phase and not the generation phase.

    Example:
    >>> model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    >>> input = torch.randint(0, 32, size=(1, 32))
    >>> with disable_cache(model):
    ...     output = model(input)
    """

    if isinstance(module, PreTrainedModel):
        config = module.config
        config = getattr(config, "text_config", config)
        with patch_attr(config, "use_cache", False):
            yield

    else:
        yield


@contextlib.contextmanager
def DisableQuantization(module: torch.nn.Module):
    """
    Disable quantization during forward passes after applying a quantization config
    """
    try:
        module.apply(disable_quantization)
        yield
    finally:
        module.apply(enable_quantization)


@contextlib.contextmanager
def eval_context(module: torch.nn.Module):
    """
    Disable pytorch training mode for the given module
    """
    restore_value = module.training
    try:
        module.train(False)  # equivalent to eval()
        yield

    finally:
        module.train(restore_value)


@contextlib.contextmanager
def disable_hf_kernels(module: torch.nn.Module):
    """
    In transformers>=4.50.0, some module forward methods may be
    replaced by calls to hf hub kernels. This has the potential
    to bypass hooks added by LLM Compressor
    """
    if isinstance(module, PreTrainedModel):
        with patch_attr(module.config, "disable_custom_kernels", True):
            yield

    else:
        yield


@contextlib.contextmanager
def calibration_forward_context(model: torch.nn.Module, eager_attention: bool = False):
    """
    Context in which all calibration forward passes should occur.

    - Remove gradient calculations
    - Disable the KV cache
    - Disable train mode and enable eval mode
    - Disable hf kernels which could bypass hooks
    - Disable lm head (input and weights can still be calibrated, output will be meta)
    - Optionally force eager attention
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        stack.enter_context(disable_cache(model))
        stack.enter_context(eval_context(model))
        stack.enter_context(disable_hf_kernels(model))
        stack.enter_context(disable_lm_head(model))
        if eager_attention:
            stack.enter_context(use_eager_attention(model))
        yield


@contextlib.contextmanager
def use_eager_attention(model: torch.nn.Module):
    """
    Temporarily force eager attention.

    Keep this active across tracing and calibration when enabled: the attention
    implementation used at trace time must match the implementation used to execute
    the resulting graph.
    """
    if isinstance(model, PreTrainedModel):
        with patch_attr(model.config, "_attn_implementation", "eager"):
            yield
    else:
        yield


@contextlib.contextmanager
def disable_lm_head(model: torch.nn.Module):
    """
    Disable the lm_head of a model by moving it to the meta device. This function
    does not untie parameters and restores the model proper loading upon exit
    """
    _, lm_head = get_embeddings(model)
    if lm_head is None:
        logger.warning(
            f"Attempted to disable lm_head of instance {model.__class__.__name__}, "
            "but was unable to to find lm_head. This may lead to unexpected OOM."
        )
        yield
        return

    elif not isinstance(lm_head, torch.nn.Linear):
        logger.warning(f"Cannot disable LM head of type {lm_head.__class__.__name__}")
        yield
        return

    else:
        # only the weight's shape is needed, so read the offloaded weight rather than
        # onloading all of it to the execution device
        with disable_onloading():
            dummy_weight = lm_head.weight.to("meta")

        def dummy_forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.to("meta") @ dummy_weight.T

        with contextlib.ExitStack() as stack:
            lm_head_forward = dummy_forward.__get__(lm_head)
            stack.enter_context(patch_attr(lm_head, "forward", lm_head_forward))

            if hasattr(model, "_hf_hook"):
                stack.enter_context(patch_attr(model._hf_hook, "io_same_device", False))

            yield


def numeric_sorted_names(names: Iterable[str]) -> list[str]:
    """
    Sort names by their numeric components, so that e.g. `layers.10` sorts
    after `layers.2` rather than before it (as a plain lexicographic sort
    would do).

    MoE layer names must be ordered like the model's layers, since per-layer
    lists (saliency reports, expert-sparsity configs) are matched positionally
    against the model's layer order.

    :param names: an iterable of names, or a mapping keyed by name
    :return: the names in numeric order
    """

    def sort_key(name: str) -> tuple:
        # split into digit and non-digit runs, comparing digit runs as
        # integers; each run is tagged so that ints never compare against
        # strings (which would raise a TypeError)
        return tuple(
            (0, int(run), "") if run.isdigit() else (1, 0, run)
            for run in re.split(r"(\d+)", name)
        )

    return sorted(names, key=sort_key)


def values_by_decoder_layer(
    values: Mapping[str, _T],
    num_layers: int | None = None,
) -> list[_T | None]:
    """
    Position values keyed by module name onto a per-decoder-layer list.

    Each name is matched against ``layers.(\\d+)`` to find the decoder layer
    its module belongs to, and its value is placed at that index of the
    returned list; every other index holds ``None`` (e.g. for dense layers).
    The result is matched positionally against the model's layer order, as
    required by per-layer checkpoint fields such as
    ``ExpertSparsityConfig.num_experts``.

    :param values: mapping from module name to that layer's value, where a
        value of ``None`` marks a layer which was tracked but not modified
    :param num_layers: total number of decoder layers, which sizes the list.
        When ``None``, the list is instead sized to cover the largest layer
        index parsed from ``values`` (with a warning), leaving any trailing
        decoder layers unrepresented
    :return: a list holding each value at its layer index and ``None``
        everywhere else
    :raises ValueError: if a name carries no ``layers.(\\d+)`` component, if a
        layer index falls outside ``[0, num_layers)``, or if two names map to
        the same layer index -- which can happen when e.g. a vision tower
        holds its own MoE layers named ``layers.N``
    """
    # parse each name's decoder layer index up front, so that the list can
    # be sized before values are positioned onto it
    index_by_name: dict[str, int] = {}
    for name in numeric_sorted_names(values):
        match = _DECODER_LAYER_PATTERN.search(name)
        if match is None:
            raise ValueError(
                f"Could not extract a decoder layer index from '{name}'; "
                "expected a module name containing a 'layers.N' component"
            )
        index_by_name[name] = int(match.group(1))

    if num_layers is None:
        logger.warning(
            "num_layers is None; sizing the per-layer list from the largest "
            "decoder layer index parsed from the given module names, so any "
            "decoder layers beyond it will not be represented in the result"
        )
        num_layers = max(index_by_name.values(), default=-1) + 1

    per_layer: list[_T | None] = [None] * num_layers
    first_name_at: dict[int, str] = {}

    for name, index in index_by_name.items():
        if not 0 <= index < num_layers:
            raise ValueError(
                f"Decoder layer index {index} parsed from '{name}' is out of "
                f"range for a model with {num_layers} decoder layers"
            )
        if index in first_name_at:
            raise ValueError(
                f"Modules '{first_name_at[index]}' and '{name}' both map to "
                f"decoder layer {index}, so their values cannot be positioned "
                "unambiguously on a per-layer list. This can happen when e.g. "
                "a vision tower holds its own MoE layers named 'layers.N'"
            )

        first_name_at[index] = name
        per_layer[index] = values[name]

    return per_layer


def hasattr_fallbacks(
    target: object, attrs: list[str], return_attr: bool = False
) -> bool | str:
    for attr in attrs:
        if hasattr(target, attr):
            return attr if return_attr else True

    return False


def getattr_fallbacks(
    target: object, attrs: list[str], default: Any = Sentinel("None")
) -> Any:
    for attr in attrs:
        if hasattr(target, attr):
            return getattr(target, attr)

    if default is not Sentinel("None"):
        return default

    raise AttributeError(f"{target} does not have any of {attrs} attributes")


def hasitem_fallbacks(
    target: dict, keys: list[str], default: Any = Sentinel("None")
) -> Any:
    for key in keys:
        if key in target:
            return key

    if default is not Sentinel("None"):
        return default

    raise AttributeError(f"{target} does not have any of {keys} keys")


def getitem_fallbacks(
    target: dict, keys: list[str], default: Any = Sentinel("None")
) -> Any:
    for key in keys:
        if key in target:
            return target[key]

    if default is not Sentinel("None"):
        return default

    raise AttributeError(f"{target} does not have any of {keys} keys")
