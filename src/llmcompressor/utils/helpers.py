"""
General utility helper functions.
Common functions for interfacing with python primitives and directories/files.
"""

import contextlib
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
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
    "getattr_fallbacks",
    "hasitem_fallbacks",
    "getitem_fallbacks",
]


def _load_module(location: str) -> ModuleType:
    """
    Load a module from either a file system path or a dotted module name.

    :param location: a path to a python file, with or without the `.py` suffix, or an
        importable dotted module name
    :return the loaded module
    """
    for candidate in (Path(location), Path(f"{location}.py")):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(candidate.stem, candidate)
            if spec is None or spec.loader is None:
                break

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    try:
        return importlib.import_module(location)
    except ImportError:
        raise ImportError(f"Cannot find module with path {location}")


def import_from_path(path: str) -> Any:
    """
    Import the module and the name of the function/class separated by :
    Examples:
      path = "/path/to/file.py:func_or_class_name"
      path = "/path/to/file:focn"
      path = "path.to.file:focn"
    :param path: path including the file path and object name
    :return Function or class object
    """
    original_path, class_name = path.rsplit(":", 1)

    module = _load_module(original_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Cannot find {class_name} in {original_path}")


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
def calibration_forward_context(model: torch.nn.Module):
    """
    Context in which all calibration forward passes should occur.

    - Remove gradient calculations
    - Disable the KV cache
    - Disable train mode and enable eval mode
    - Disable hf kernels which could bypass hooks
    - Disable lm head (input and weights can still be calibrated, output will be meta)
    - Force eager attention (see `use_eager_attention`)
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        stack.enter_context(disable_cache(model))
        stack.enter_context(eval_context(model))
        stack.enter_context(disable_hf_kernels(model))
        stack.enter_context(disable_lm_head(model))
        stack.enter_context(use_eager_attention(model))
        yield


@contextlib.contextmanager
def use_eager_attention(model: torch.nn.Module):
    """
    Temporarily force eager attention for the duration of calibration.

    The sequential pipeline traces subgraphs under eager attention, which bakes an
    unconditional causal-mask add into the traced graph. Runtime calibration must match
    so that ``create_causal_mask`` uses the eager path and always returns a mask tensor;
    otherwise (e.g. sdpa) it returns ``None`` for unpadded batches and the baked-in add
    fails. This matters when samples are not padded to a fixed length.
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
        dummy_weight = lm_head.weight.to("meta")

        def dummy_forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.to("meta") @ dummy_weight.T

        with contextlib.ExitStack() as stack:
            lm_head_forward = dummy_forward.__get__(lm_head)
            stack.enter_context(patch_attr(lm_head, "forward", lm_head_forward))

            if hasattr(model, "_hf_hook"):
                stack.enter_context(patch_attr(model._hf_hook, "io_same_device", False))

            yield


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
