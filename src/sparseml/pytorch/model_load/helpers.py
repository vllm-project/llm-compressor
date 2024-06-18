# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.nn import Module

import sparseml
from safetensors import safe_open
from sparseml.core.framework import Framework
from sparseml.pytorch.utils import ModuleSparsificationInfo


COMPLETED_STAGES_FILENAME = "completed_stages.json"

__all__ = [
    "log_model_load",
    "initialize_recipe",
    "reload_model_state",
    "reload_model_from_checkpoint",
    "save_model_and_recipe",
    "fallback_to_cpu",
    "parse_dtype",
    "get_session_model",
    "get_completed_stages",
    "save_completed_stages"
]

_LOGGER = logging.getLogger(__name__)

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
        _LOGGER.info(
            f"Delayed load of model {model_name_or_path} detected. "
            f"Will print out model information once SparseML recipes have loaded"
        )
        return

    sparsification_info = ModuleSparsificationInfo(model)

    _LOGGER.info(
        f"Loaded {model_type} from {model_name_or_path} "
        f"with {sparsification_info.params_total} total params. "
        f"Of those there are {sparsification_info.params_prunable_total} prunable "
        f"params which have {sparsification_info.params_prunable_sparse_percent} "
        "avg sparsity."
    )
    model_type = (
        "sparse" if sparsification_info.params_prunable_sparse_percent > 5 else "dense"
    )
    _LOGGER.info(
        f"{model_type} model detected, "
        f"all sparsification info: {sparsification_info}"
    )


def initialize_recipe(model: Module, recipe_path: str):
    """
    Initializes a recipe that has been previously applied to the model

    :param model: PyTorch model to apply structure to
    :param recipe_path: path to recipe to apply to the model
    """
    if not sparseml.active_session():
        sparseml.create_session()
    sparseml.pre_initialize_structure(
        model=model, recipe=recipe_path, framework=Framework.pytorch
    )

    # no need to reload if no recipe was applied
    if recipe_path is None:
        return

    session = sparseml.active_session()
    num_stages = len(session.lifecycle.recipe_container.compiled_recipe.stages)
    msg = (
        "an unstaged recipe"
        if num_stages == 1
        else f"a staged recipe with {num_stages} stages"
    )
    _LOGGER.info(f"Applied {msg} to the model")

def reload_model_from_checkpoint(model: Module, checkpoint: Optional[str] = None):
    """
    Reload the model state dict from a specified checkpoint if provided

    :model: loaded pytorch module
    :checkpoint: path to checkpoint file to load
    """
    if checkpoint is None:
        return

    orig_state_dict = model.state_dict()

    # reload the state dict for the model from the checkpoint
    if reload_model_state(model, checkpoint, orig_state_dict):
        _LOGGER.info(f"Reloaded model state from checkpoint {checkpoint}")


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

    _LOGGER.info("Saving output to {}".format(os.path.abspath(save_path)))

    recipe_path = os.path.join(save_path, RECIPE_FILE_NAME)
    session = sparseml.active_session()
    recipe_yaml_str = session.get_serialized_recipe()
    with open(recipe_path, "w") as fp:
        fp.write(recipe_yaml_str)


def fallback_to_cpu(device: str) -> str:
    """
    Takes in a device string and forces it to cpu if cuda is not available

    :param device: device id to check
    :return: device modified for CUDA status
    """
    if "cuda" in device and not torch.cuda.is_available():
        _LOGGER.warning(
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


def get_session_model() -> Module:
    """
    :return: pytorch module stored by the active SparseSession, or None if no session
    is active
    """
    session = sparseml.active_session()
    if not session:
        return None

    active_model = session.state.model.model
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
