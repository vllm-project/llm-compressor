import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args.dataset_arguments import DatasetArguments
from llmcompressor.args.model_arguments import ModelArguments
from llmcompressor.entrypoints.utils import (
    _warn_tied_embeddings,
    initialize_processor_from_path,
)
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.factory import ModifierFactory
from llmcompressor.pytorch.model_load.helpers import parse_dtype
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    patch_tied_tensors_bug,
    untie_weights,
)
from llmcompressor.transformers.utils.helpers import is_model_ct_quantized_from_path
from llmcompressor.typing import RecipeInput

""" llmcompressor.Args """


@dataclass
class LCModelArguments(ModelArguments):
    recipe: RecipeInput = field(
        default=None
    )  # dummy default to avoid inheritance issue


@dataclass
class LCDatasetArguments(DatasetArguments):
    split: Optional[str] = field(default=None)


""" llmcompressor.recipe """


def get_modifiers_from_recipe(
    recipe: Union[str, List[Modifier], Modifier],
) -> List[Modifier]:
    # trivial cases
    if isinstance(recipe, Modifier):
        return [recipe]
    if isinstance(recipe, List):
        return recipe

    # load yaml as dict
    if os.path.exists(recipe):
        with open(recipe, "r") as file:
            recipe = yaml.safe_load(file)
    else:
        recipe_dict = yaml.safe_load(recipe)

    if not isinstance(recipe_dict, dict):
        raise ValueError("Cannot parse yaml")

    breakpoint()
    if not ModifierFactory._loaded:
        ModifierFactory.refresh()

    return [
        ModifierFactory.create(
            modifier_type,
            allow_registered=True,
            allow_experimental=True,
            **args,
        )
        for modifier_type, args in get_modifiers_args_from_dict(recipe_dict)
    ]


def get_modifiers_args_from_dict(values: Dict) -> List[Dict[str, Any]]:
    modifiers = []
    remove_keys = []

    if "modifiers" in values and values["modifiers"]:
        remove_keys.append("modifiers")
        for mod_key, mod_value in values["stages"].items():
            modifier = {mod_key: mod_value}
            modifier["group"] = "default"
            modifiers.append(modifier)

    for key, value in list(values.items()):
        if key.endswith("_modifiers"):
            remove_keys.append(key)
            group = key.rsplit("_modifiers", 1)[0]
            for mod_key, mod_value in value.items():
                modifier = {mod_key: mod_value}
                modifier["group"] = group
                modifiers.append(modifier)

    for key in remove_keys:
        del values[key]

    return modifiers


""" llmcompressor.model """


def prepare_models(model_args: ModelArguments):
    # Initialize model
    if isinstance(model_args.model, str):
        model_args.model = initialize_model_from_path(model_args.model, model_args)

    # Initialize teacher
    if isinstance(model_args.distill_teacher, str):
        model_args.distill_teacher = initialize_model_from_path(
            model_args.distill_teacher, model_args
        )

    # Initialize processor
    if isinstance(model_args.processor, (str, type(None))):
        model_args.processor = initialize_processor_from_path(
            model_args, model_args.model
        )

    # warnings and patches
    _warn_tied_embeddings(model_args.tie_word_embeddings)
    patch_tied_tensors_bug(model_args.model)  # untie tie_word_embeddings weights
    if model_args.tie_word_embeddings:
        untie_weights(model_args.model)

    # potentially attach this compressor to the model?

    return model_args.model, model_args.distill_teacher, model_args.processor


def initialize_model_from_path(
    model_path: str, model_args: ModelArguments
) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code_model,
    )

    # TODO: seems to be redundancy between config and model kwargs
    model_kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "torch_dtype": parse_dtype(model_args.precision),
        "device_map": model_args.oneshot_device or "auto",
        "trust_remote_code": model_args.trust_remote_code_model,
    }

    # for convenience, decompress any CT compressed models
    if is_model_ct_quantized_from_path(model_path):
        logger.warning("Decompressing model")
        model_kwargs["quantization_config"] = CompressedTensorsConfig(
            run_compressed=False
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    if "sequence_length" in model_kwargs:
        model.seqlen = model_kwargs[
            "sequence_length"
        ]  # TODO: Pretty sure the seqlen attribute is never used/ doesn't exist

    return model
