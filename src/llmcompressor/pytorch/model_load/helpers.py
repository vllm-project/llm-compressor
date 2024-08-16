import json
import os
from typing import Any, Dict, List, Optional

import torch
from loguru import logger
from safetensors import safe_open
from torch.nn import Module

from llmcompressor.core import active_session, create_session, pre_initialize_structure
from llmcompressor.pytorch.utils import ModuleSparsificationInfo

COMPLETED_STAGES_FILENAME = "completed_stages.json"

__all__ = [
    "log_model_load",
    "initialize_recipe",
    "save_model_and_recipe",
    "fallback_to_cpu",
    "parse_dtype",
    "get_session_model",
    "get_completed_stages",
    "save_completed_stages",
]

RECIPE_FILE_NAME = "recipe.yaml"


def log_model_load(
    model: Module, model_name_or_path: str, model_type: str, delayed_load: bool
):
    """
    Log the state of a loaded model including sparsity and
    prunable params information.

    :param model: the loaded model
    :param model_name_or_path: the original name of or path to the model that loaded
    :param model_type: specify the type of model loaded for logging;
        ex one of [model, student, teacher]
    :param delayed_load: True if this model load was delayed until after
        recipe instantiation due to QAT or other architectural state changes
    """
    if delayed_load:
        logger.info(
            f"Delayed load of model {model_name_or_path} detected. "
            f"Will print out model information once LLMCompressor recipes have loaded"
        )
        return

    sparsification_info = ModuleSparsificationInfo(model)

    logger.info(
        f"Loaded {model_type} from {model_name_or_path} "
        f"with {sparsification_info.params_total} total params. "
        f"Of those there are {sparsification_info.params_prunable_total} prunable "
        f"params which have {sparsification_info.params_prunable_sparse_percent} "
        "avg sparsity."
    )
    model_type = (
        "sparse" if sparsification_info.params_prunable_sparse_percent > 5 else "dense"
    )
    logger.info(
        f"{model_type} model detected, "
        f"all sparsification info: {sparsification_info}"
    )


def initialize_recipe(model: Module, recipe_path: str):
    """
    Initializes a recipe that has been previously applied to the model

    :param model: PyTorch model to apply structure to
    :param recipe_path: path to recipe to apply to the model
    """
    if not active_session():
        create_session()
    pre_initialize_structure(model=model, recipe=recipe_path)

    # no need to reload if no recipe was applied
    if recipe_path is None:
        return

    session = active_session()
    num_stages = len(session.lifecycle.recipe_container.compiled_recipe.stages)
    msg = (
        "an unstaged recipe"
        if num_stages == 1
        else f"a staged recipe with {num_stages} stages"
    )
    logger.info(f"Applied {msg} to the model")


def save_model_and_recipe(
    model: Module,
    save_path: str,
    tokenizer: Optional[Any] = None,
    save_safetensors: bool = False,
    save_compressed: bool = False,
):
    """
    Save a model, tokenizer and the currently loaded recipe to file

    :param model: pytorch model to save
    :param save_path: path to save output to
    :param tokenizer: model tokenizer to save
    :param save_safetensors: whether to save as safetensors or pickle (bin)
    :param save_compressed: whether to compress sparse weights on disk
    """

    model.save_pretrained(
        save_path, save_compressed=save_compressed, safe_serialization=save_safetensors
    )

    if tokenizer is not None:
        tokenizer.save_pretrained(save_path)

    logger.info("Saving output to {}".format(os.path.abspath(save_path)))

    recipe_path = os.path.join(save_path, RECIPE_FILE_NAME)
    session = active_session()
    recipe_yaml_str = session.get_serialized_recipe()
    with open(recipe_path, "w") as fp:
        fp.write(recipe_yaml_str)

    # copy python files from cache dir to save_path if any
    _copy_python_files_from_model_cache(model, save_path)


def fallback_to_cpu(device: str) -> str:
    """
    Takes in a device string and forces it to cpu if cuda is not available

    :param device: device id to check
    :return: device modified for CUDA status
    """
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning(
            f"Requested {device} but CUDA is not available, falling back to CPU"
        )
        return "cpu"

    return device


def parse_dtype(dtype_arg: str) -> torch.dtype:
    """
    :param dtype_arg: dtype string to parse
    :return: torch.dtype parsed from input string
    """
    dtype = "auto"  # get precision from model by default
    if dtype_arg == "half" or dtype_arg == "float16":
        dtype = torch.float16
    elif dtype_arg == "bfloat16":
        dtype = torch.bfloat16
    elif dtype_arg == "full" or dtype_arg == "float32":
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


def _copy_python_files_from_model_cache(model: Module, save_path: str):
    config = model.config
    cache_dir = None
    if hasattr(config, "_name_or_path"):
        import os
        import shutil

        cache_dir = config._name_or_path
        for file in os.listdir(cache_dir):
            full_file_name = os.path.join(cache_dir, file)
            if file.endswith(".py") and os.path.isfile(full_file_name):
                logger.debug(f"Transferring {full_file_name} to {save_path}")
                shutil.copy(full_file_name, save_path)
