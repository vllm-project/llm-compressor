import contextlib
import os
import tempfile
from typing import Type
import logging

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

from llmcompressor.utils import patch_attr


@contextlib.contextmanager
def skip_weights_download(model_class: Type[PreTrainedModel] = AutoModelForCausalLM):
    """
    Context manager under which models are initialized without having to download
    the model weight files

    :param model_class: class to patch, `AutoModelForCausalLM`
    """
    original_fn = model_class.from_pretrained
    weights_files = [
        "*.bin",
        "*.safetensors",
        "*.pth",
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
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
        patch_transformers_logger_level(),
    ):
        yield


@contextlib.contextmanager
def skip_weights_initialize(use_zeros: bool = False):
    def skip(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if use_zeros:
            return tensor.fill_(0)
        return tensor

    with (
        patch_attr(torch.nn.init, "kaiming_uniform_", skip),
        patch_attr(torch.nn.init, "uniform", skip),
        patch_attr(torch.nn.init, "normal_", skip),
        patch_attr(torch.Tensor, "kaiming_uniform_", skip),
        patch_attr(torch.Tensor, "uniform", skip),
        patch_attr(torch.Tensor, "normal_", skip),
    ):
        yield


@contextlib.contextmanager
def patch_transformers_logger_level(level: int = logging.ERROR):
    transformers_logger = logging.getLogger("transformers.modeling_utils")
    restore_log_level = transformers_logger.getEffectiveLevel()
    transformers_logger.setLevel(level=level)

    yield

    transformers_logger.setLevel(level=restore_log_level)