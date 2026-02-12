import contextlib
import logging
import os
import tempfile
from functools import wraps
from typing import Type

import torch
from compressed_tensors.offload import dispatch_model
from compressed_tensors.utils import patch_attr
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_utils import TORCH_INIT_FUNCTIONS
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

__all__ = [
    "skip_weights_download",
    "patch_transformers_logger_level",
    "get_main_device",
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

    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch_attr(model_class, "from_pretrained", patched),
        skip_weights_initialize(),
        patch_transformers_logger_level(),
    ):
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


def get_main_device() -> torch.device:
    rank = 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device(f"xpu:{rank}")
    else:
        logger.warning("CUDA/XPU is not available! Compressing model on CPU instead")
        return torch.device("cpu")


@wraps(dispatch_model)
def dispatch_for_generation(*args, **kwargs) -> PreTrainedModel:
    """
    Dispatch a model autoregressive generation. This means that modules are dispatched
    evenly across avaiable devices and kept onloaded if possible.

    :param model: model to dispatch
    :param hint_batch_size: reserve memory for batch size of inputs
    :param hint_batch_seq_len: reserve memory for sequence of length of inputs
    :param hint_model_dtype: reserve memory for model's dtype.
        Will be inferred from model if none is provided
    :param hint_extra_memory: extra memory reserved for model serving
    :param no_split_modules: names of module classes which should not be split
        across multiple devices
    :return: dispatched model
    """
    return dispatch_model(*args, **kwargs)
