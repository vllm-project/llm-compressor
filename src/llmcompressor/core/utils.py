import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from torch.utils.data.dataloader import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.utils.quantization_config import CompressedTensorsConfig

from llmcompressor.args import ModelArguments
from llmcompressor.pytorch.model_load.helpers import parse_dtype
from llmcompressor.transformers.sparsification.compressed_tensors_utils import (
    patch_tied_tensors_bug,
    untie_weights,
)
from llmcompressor.transformers.utils.helpers import is_model_ct_quantized_from_path
from llmcompressor.utils import resolve_modifier_quantization_config

if TYPE_CHECKING:
    from llmcompressor.modifiers import Modifier


""" llmcompressor.recipe """


def get_modifiers_from_recipe(
    recipe: Union[str, List["Modifier"], "Modifier"],
) -> List["Modifier"]:
    # avoid circular import
    from llmcompressor.modifiers import Modifier, ModifierFactory

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

    # validate yaml
    if not isinstance(recipe_dict, dict):
        raise ValueError("Cannot parse yaml")

    # load modifiers
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
    # TODO: circular import
    from llmcompressor.entrypoints.utils import initialize_processor_from_path

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

    # warnings
    if model_args.tie_word_embeddings:
        logger.debug(
            "The tie_word_embeddings flag is by default set to False. "
            "This guarantees that the one-shot algorithm saves the final "
            "weights without errors. Detected tie_word_embeddings=True. "
            "This may cause issues with the one-shot algorithm on save."
        )

    # patch tied weights
    patch_tied_tensors_bug(model_args.model)  # untie tie_word_embeddings weights
    if model_args.tie_word_embeddings:
        untie_weights(model_args.model)
    if model_args.distill_teacher is not None:
        patch_tied_tensors_bug(model_args.distill_teacher)
        if model_args.tie_word_embeddings:
            untie_weights(model_args.distill_teacher)

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
        "device_map": "auto",  # default to load on cpu
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


""" llmcompressor.data """


def error_if_requires_calibration_data(
    modifiers: List["Modifier"], calibration_loader: Optional[DataLoader]
):
    requires_data = False
    for modifier in modifiers:
        if hasattr(modifier, "scheme"):
            config = resolve_modifier_quantization_config(modifier)
            if config.requires_calibration_data():
                requires_data = True
                break

    if requires_data and calibration_loader is None:
        raise ValueError(
            "Recipe requries calibration data, but none was provided. Please call "
            "LLMCompressor.set_calibration_dataset with a calibration dataset"
        )
