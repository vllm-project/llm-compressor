"""
General utility helper functions.
Common functions for interfacing with python primitives and directories/files.
"""

import contextlib
import importlib.metadata
import importlib.util
import re
from typing import Tuple, Union

import torch
from compressed_tensors.quantization import disable_quantization, enable_quantization
from compressed_tensors.utils import patch_attr
from loguru import logger
from transformers import PreTrainedModel

from llmcompressor.utils import get_embeddings

__all__ = [
    "is_package_available",
    "import_from_path",
    "disable_cache",
    "DisableQuantization",
    "eval_context",
    "disable_hf_kernels",
    "calibration_forward_context",
    "disable_lm_head",
    "DISABLE_QAC_MODIFIERS",
]


def is_package_available(
    package_name: str,
    return_version: bool = False,
) -> Union[Tuple[bool, str], bool]:
    """
    A helper function to check if a package is available
    and optionally return its version. This function enforces
    a check that the package is available and is not
    just a directory/file with the same name as the package.

    inspired from:
    https://github.com/huggingface/transformers/blob/965cf677695dd363285831afca8cf479cf0c600c/src/transformers/utils/import_utils.py#L41

    :param package_name: The package name to check for
    :param return_version: True to return the version of
        the package if available
    :return: True if the package is available, False otherwise or a tuple of
        (bool, version) if return_version is True
    """

    package_exists = importlib.util.find_spec(package_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(package_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logger.debug(f"Detected {package_name} version {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


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
def calibration_forward_context(model: torch.nn.Module):
    """
    Context in which all calibration forward passes should occur.

    - Remove gradient calculations
    - Disable the KV cache
    - Disable train mode and enable eval mode
    - Disable hf kernels which could bypass hooks
    - Disable lm head (input and weights can still be calibrated, output will be meta)
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        stack.enter_context(disable_cache(model))
        stack.enter_context(eval_context(model))
        stack.enter_context(disable_hf_kernels(model))
        stack.enter_context(disable_lm_head(model))
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


DISABLE_QAC_MODIFIERS = ["GPTQModifier", "AWQModifier", "AutoRoundModifier"]
