import json
import os
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from safetensors import safe_open
from torch.nn import Module
from transformers import PreTrainedModel

from llmcompressor.core import active_session
from llmcompressor.typing import Processor

COMPLETED_STAGES_FILENAME = "completed_stages.json"

__all__ = [
    "copy_python_files_from_model_cache",
    "parse_dtype",
    "get_session_model",
    "get_completed_stages",
    "save_completed_stages",
    "save_checkpoint",
]


def save_checkpoint(
    save_path: str,
    model: PreTrainedModel,
    processor: Optional[Processor] = None,
    save_safetensors: bool = True,
    save_compressed: bool = True,
    skip_sparsity_compression_stats: bool = False,
):
    """
    Save a model, processor, and recipe

    :param save_path: Path used to save model and processor
    :param model: model to save
    :param processor: processor to save
    :param save_safetensors: save model checkpoint using safetensors file type
    :param save_compressed: save model checkpoint using compressed-tensors format
    """
    from llmcompressor.transformers.compression.compressed_tensors_utils import (
        get_model_compressor,  # avoid circular import
    )

    # used for decompression
    # unfortunately, if skip_sparsity_compression_stats==True, sparsity stats
    # are computed twice. In the future, track sparsity from recipe or
    # share recipe between compression and decompression
    compressor = get_model_compressor(
        model=model,
        save_compressed=save_compressed,
        skip_sparsity_compression_stats=skip_sparsity_compression_stats,
    )

    # saving the model also saves the recipe
    model.save_pretrained(
        save_path,
        save_safetensors=save_safetensors,
        save_compressed=save_compressed,
        skip_sparsity_compression_stats=skip_sparsity_compression_stats,
    )
    if processor is not None:
        processor.save_pretrained(save_path)

    # decompression: saving the model modifies the model strcuture
    # as this is only a checkpoint, decompress model to enable future training/oneshot
    if compressor is not None:
        compressor.decompress_model(model)


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


def get_completed_stages(checkpoint_dir: Any) -> List[str]:
    """
    Given a checkpoint directory for a staged run, get the list of stages that
    have completed in a prior run if the checkpoint_dir is a string

    :param checkpoint_dir: path to staged checkpoint
    :return: list of completed stage names
    """
    if isinstance(checkpoint_dir, str):
        stage_path = os.path.join(checkpoint_dir, COMPLETED_STAGES_FILENAME)
        if os.path.exists(stage_path):
            with open(stage_path) as stage_file:
                stage_data = json.load(stage_file)
                return stage_data["completed"]

    return []


def save_completed_stages(checkpoint_dir: str, completed_stages: List[str]):
    """
    Save a list of completed stages to a checkpoint directory

    :param checkpoint_dir: model checkpoint directory to save stages to
    :param completed_stages: list of stage names that have been run
    """
    stage_path = os.path.join(checkpoint_dir, COMPLETED_STAGES_FILENAME)
    with open(stage_path, "w") as out_file:
        json.dump({"completed": completed_stages}, out_file)


def load_safetensors_state_dict(file_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a safetensors file from disk

    :param file_path: path to the safetensors file
    :return: dictionary of safetensors data
    """
    with safe_open(file_path, framework="pt", device="cpu") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def copy_python_files_from_model_cache(model, save_path: str):
    config = model.config
    cache_path = None
    if hasattr(config, "_name_or_path"):
        import os
        import shutil

        from huggingface_hub import hf_hub_download
        from transformers import TRANSFORMERS_CACHE
        from transformers.utils import http_user_agent

        cache_path = config._name_or_path
        if not os.path.exists(cache_path):
            user_agent = http_user_agent()
            config_file_path = hf_hub_download(
                repo_id=cache_path,
                filename="config.json",
                cache_dir=TRANSFORMERS_CACHE,
                force_download=False,
                user_agent=user_agent,
            )
            cache_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

        for file in os.listdir(cache_path):
            full_file_name = os.path.join(cache_path, file)
            if file.endswith(".py") and os.path.isfile(full_file_name):
                logger.debug(f"Transferring {full_file_name} to {save_path}")
                shutil.copy(full_file_name, save_path)
