import contextlib
import logging
import os
import tempfile
from typing import Type

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from compressed_tensors.utils import remove_dispatch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

from llmcompressor.utils.helpers import patch_attr

__all__ = [
    "skip_weights_download",
    "patch_transformers_logger_level",
    "dispatch_for_generation",
]


@contextlib.contextmanager
def skip_weights_download(model_class: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager under which models are initialized without having to download
    the model weight files. This differs from `init_empty_weights` in that weights are
    allocated on to assigned devices with random values, as opposed to being on the meta
    device

    :param model_class: class to patch, defaults to `AutoModelForCausalLM`
    """
    original_fn = model_class.from_pretrained
    weights_files = [
        "*.bin",
        "*.safetensors",
        "*.pth",
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
        "*.msgpack",
        "*.pt",
    ]

    @classmethod
    def patched(cls, *args, **kwargs):
        nonlocal tmp_dir

        # intercept model stub
        model_stub = args[0] if args else kwargs.pop("pretrained_model_name_or_path")

        # download files into tmp dir
        os.makedirs(tmp_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_stub, local_dir=tmp_dir, ignore_patterns=weights_files
        )

        # make an empty weights file to avoid errors
        weights_file_path = os.path.join(tmp_dir, "model.safetensors")
        save_file({}, weights_file_path, metadata={"format": "pt"})

        # load from tmp dir
        model = original_fn(tmp_dir, **kwargs)

        # replace model_path
        model.name_or_path = model_stub
        model.config._name_or_path = model_stub

        return model

    with tempfile.TemporaryDirectory() as tmp_dir, patch_attr(
        model_class, "from_pretrained", patched
    ), skip_weights_initialize(), patch_transformers_logger_level():
        yield


@contextlib.contextmanager
def skip_weights_initialize(use_zeros: bool = False):
    """
    Very similar to `transformers.model_utils.no_init_weights`, except that torch.Tensor
    initialization functions are also patched to account for tensors which are
    initialized not on the meta device
    """

    def skip(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if use_zeros:
            return tensor.fill_(0)
        return tensor

    with contextlib.ExitStack() as stack:
        for name in TORCH_INIT_FUNCTIONS.keys():
            stack.enter_context(patch_attr(torch.nn.init, name, skip))
            stack.enter_context(patch_attr(torch.Tensor, name, skip))
        yield


@contextlib.contextmanager
def patch_transformers_logger_level(level: int = logging.ERROR):
    """
    Context under which the transformers logger's level is modified

    This can be used with `skip_weights_download` to squelch warnings related to
    missing parameters in the checkpoint

    :param level: new logging level for transformers logger. Logs whose level is below
        this level will not be logged
    """
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    restore_log_level = transformers_logger.getEffectiveLevel()

    transformers_logger.setLevel(level=level)
    yield
    transformers_logger.setLevel(level=restore_log_level)


def dispatch_for_generation(model: PreTrainedModel) -> PreTrainedModel:
    """
    Dispatch a model autoregressive generation. This means that modules are dispatched
    evenly across avaiable devices and kept onloaded if possible. Removes any HF hooks
    that may have existed previously.

    :param model: model to dispatch
    :return: model which is dispatched
    """
    remove_dispatch(model)

    no_split_module_classes = model._get_no_split_modules("auto")
    max_memory = get_balanced_memory(
        model,
        dtype=model.dtype,
        no_split_module_classes=no_split_module_classes,
    )
    device_map = infer_auto_device_map(
        model,
        dtype=model.dtype,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
    )

    return dispatch_model(model, device_map=device_map)
