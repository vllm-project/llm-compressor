from typing import Optional, Union

import torch
from loguru import logger
from torch.nn import Module

from llmcompressor.core import active_session

__all__ = [
    "copy_python_files_from_model_cache",
    "parse_dtype",
    "get_session_model",
]


def parse_dtype(dtype_arg: Union[str, torch.dtype]) -> torch.dtype:
    """
    :param dtype_arg: dtype or string to parse
    :return: torch.dtype parsed from input string
    """
    dtype_arg = str(dtype_arg)
    dtype = "auto"  # get precision from model by default
    if dtype_arg in ("half", "float16", "torch.float16"):
        dtype = torch.float16
    elif dtype_arg in ("torch.bfloat16", "bfloat16"):
        dtype = torch.bfloat16
    elif dtype_arg in ("full", "float32", "torch.float32"):
        dtype = torch.float32

    return dtype


def get_session_model() -> Optional[Module]:
    """
    :return: pytorch module stored by the active CompressionSession,
        or None if no session is active
    """
    session = active_session()
    if not session:
        return None

    active_model = session.state.model
    return active_model


def copy_python_files_from_model_cache(model, save_path: str):
    config = model.config
    cache_path = None
    if hasattr(config, "_name_or_path") and len(config._name_or_path.strip()) > 0:
        import os
        import shutil

        from huggingface_hub import hf_hub_download
        from transformers.utils import http_user_agent

        cache_path = config._name_or_path
        if not os.path.exists(cache_path):
            user_agent = http_user_agent()
            # Use cache_dir=None to respect HF_HOME, HF_HUB_CACHE, and other
            # environment variables for cache location
            config_file_path = hf_hub_download(
                repo_id=cache_path,
                filename="config.json",
                cache_dir=None,
                force_download=False,
                user_agent=user_agent,
            )
            cache_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

        for file in os.listdir(cache_path):
            full_file_name = os.path.join(cache_path, file)
            if file.endswith(".py") and os.path.isfile(full_file_name):
                logger.debug(f"Transferring {full_file_name} to {save_path}")
                shutil.copy(full_file_name, save_path)
