import json
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.recipe.utils import (
    _load_json_or_yaml_string,
    _parse_recipe_from_md,
    append_recipe_dict,
    filter_dict,
    get_yaml_serializable_dict,
)

__all__ = [
    "Recipe",
    "RecipeInput",
    "RecipeStageInput",
    "RecipeArgsInput",
]


def _validate_modifier_ordering(modifiers: List[Modifier]) -> None:
    """
    Validate modifier ordering for AWQModifier.

    This function validates that:
    1. AWQModifier does not appear after a quantization modifier
    2. AWQModifier config_groups matches subsequent quantization modifier's config

    Note: Warning for standalone AWQ is handled by AWQModifier._warn_if_standalone
    during on_finalize, which has access to the session state.

    :param modifiers: list of modifiers to validate
    :raises ValueError: if AWQModifier appears after a quantization modifier
    :raises ValueError: if AWQModifier config_groups mismatch with subsequent modifier
    """
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor.modifiers.quantization.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization.quantization import QuantizationModifier

    # Explicit type checking for quantization modifiers
    quant_modifier_types = (QuantizationModifier, GPTQModifier)

    # Find all AWQ modifiers and quantization modifiers
    awq_indices = []
    quant_indices = []

    for idx, modifier in enumerate(modifiers):
        if isinstance(modifier, AWQModifier):
            awq_indices.append(idx)
        elif isinstance(modifier, quant_modifier_types):
            quant_indices.append(idx)

    for awq_idx in awq_indices:
        # Check if AWQ appears after any quantization modifier
        for quant_idx in quant_indices:
            if quant_idx < awq_idx:
                raise ValueError(
                    f"AWQModifier at position {awq_idx} appears after quantization "
                    f"modifier at position {quant_idx}. AWQModifier must come before "
                    "quantization modifiers in the recipe. "
                    "Example: [AWQModifier(...), QuantizationModifier(...)]"
                )

        # Validate config_groups match if there's a subsequent quantization modifier
        subsequent_quants = [q for q in quant_indices if q > awq_idx]
        if subsequent_quants:
            awq_modifier = modifiers[awq_idx]
            next_quant_idx = min(subsequent_quants)
            quant_modifier = modifiers[next_quant_idx]

            _validate_awq_config_match(
                awq_modifier, quant_modifier, awq_idx, next_quant_idx
            )


def _validate_awq_config_match(
    awq_modifier, quant_modifier, awq_idx: int, quant_idx: int
) -> None:
    """
    Validate that AWQModifier config_groups matches the subsequent quantization
    modifier's config_groups for weight quantization parameters.

    :param awq_modifier: AWQModifier instance
    :param quant_modifier: subsequent quantization modifier
    :param awq_idx: index of AWQModifier in recipe
    :param quant_idx: index of quantization modifier in recipe
    :raises ValueError: if config_groups mismatch
    """
    awq_config = awq_modifier.config_groups
    quant_config = getattr(quant_modifier, "config_groups", None)

    if awq_config is None or quant_config is None:
        return  # Nothing to validate

    # Extract weight parameters from each config for comparison
    awq_weight_params = _extract_weight_params(awq_config)
    quant_weight_params = _extract_weight_params(quant_config)

    if awq_weight_params and quant_weight_params:
        mismatches = _compare_weight_params(awq_weight_params, quant_weight_params)
        if mismatches:
            raise ValueError(
                f"AWQModifier at position {awq_idx} has config_groups that do not "
                f"match the subsequent quantization modifier at position {quant_idx}. "
                f"Mismatching weight parameters: {mismatches}. "
                "AWQModifier's config_groups must match the quantization modifier's "
                "config_groups to ensure consistent quantization during grid search."
            )


def _extract_weight_params(config_groups: dict) -> dict:
    """
    Extract weight parameters from config_groups.

    :param config_groups: config_groups dictionary
    :return: dictionary of weight parameters
    """
    weight_params = {}
    for group_name, group_config in config_groups.items():
        if isinstance(group_config, dict) and "weights" in group_config:
            weights = group_config["weights"]
            if isinstance(weights, dict):
                weight_params[group_name] = {
                    k: v
                    for k, v in weights.items()
                    if k in ["num_bits", "type", "symmetric", "strategy", "group_size"]
                }
    return weight_params


def _compare_weight_params(awq_params: dict, quant_params: dict) -> List[str]:
    """
    Compare weight parameters between AWQ and quantization modifier configs.

    :param awq_params: AWQ weight parameters
    :param quant_params: quantization modifier weight parameters
    :return: list of mismatching parameter names
    """
    mismatches = []

    # Compare all matching groups by name
    for group_name in awq_params:
        if group_name not in quant_params:
            continue

        awq_weights = awq_params[group_name]
        quant_weights = quant_params[group_name]

        all_keys = set(awq_weights.keys()) | set(quant_weights.keys())
        for key in all_keys:
            awq_val = awq_weights.get(key)
            quant_val = quant_weights.get(key)
            if awq_val is not None and quant_val is not None and awq_val != quant_val:
                mismatches.append(f"group {group_name}: {key}: AWQ={awq_val}, quant={quant_val}")

    return mismatches


class Recipe(BaseModel):
    """
    A class to represent a recipe for a model.
    Recipes encode the instructions needed for modifying
    the model and/or training process as a list of modifiers.

    Recipes can be created from a file, string, or HuggingFace stub.
    Acceptable file formats include both json and yaml, however,
    when serializing a recipe, yaml will be used by default.
    """

    args: Dict[str, Any] = Field(default_factory=dict)
    stage: str = "default"
    modifiers: List[Modifier] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_modifiers(
        cls,
        modifiers: Union[Modifier, List[Modifier]],
        modifier_group_name: Optional[str] = None,
    ) -> "Recipe":
        """
        Create a recipe instance from a list of modifiers

        (Note: all modifiers are wrapped into a single stage
        with the modifier_group_name as the stage name. If modifier_group_name is None,
        the default run type is `oneshot`)

        Lfecycle:
        | - Validate Modifiers
        | - Create recipe string from modifiers
        | - Create recipe instance from recipe string

        :param modifiers: The list of RecipeModifier instances
        :param modifier_group_name: The stage_name of the recipe,
            if `oneshot` or `train` the run_type of the recipe will be
            inferred from the modifier_group_name, if None, a dummy default
            group_name will be assigned.
        :return: The Recipe instance created from the modifiers
        """
        logger.info("Creating recipe from modifiers")

        if isinstance(modifiers, Modifier):
            modifiers = [modifiers]

        if any(not isinstance(modifier, Modifier) for modifier in modifiers):
            raise ValueError("modifiers must be a list of Modifier instances")

        # Validate modifier ordering (e.g., AWQ before quantization)
        _validate_modifier_ordering(modifiers)

        group_name = modifier_group_name or "default"

        # assume one stage for modifier instances
        recipe = cls()
        recipe.stage = group_name
        recipe.modifiers = modifiers
        return recipe

    @classmethod
    def create_instance(
        cls,
        path_or_modifiers: Union[str, Modifier, List[Modifier], "Recipe"],
        modifier_group_name: Optional[str] = None,
        target_stage: Optional[str] = None,
    ) -> "Recipe":
        """
        Create a recipe instance from a file, string, or RecipeModifier objects


        Using a recipe string or file is supported:
        >>> recipe_str = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 0.0
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe = Recipe.create_instance(recipe_str, target_stage="test_stage")

        :param path_or_modifiers: The path to the recipe file or
            or the recipe string (must be a valid
            json/yaml file or a valid json/yaml string). Can also
            accept a RecipeModifier instance, or a list of
            RecipeModifiers
        :param modifier_group_name: The stage_name of the recipe,
            if `oneshot` or `train` the run_type of the recipe will be
            inferred from the modifier_group_name, if None, a dummy default
            group_name will be assigned. This argument is only used
            when creating a recipe from a Modifier/list of Modifier(s)
            instance, else it's ignored.
        :return: The Recipe instance created from the path or modifiers,
            or a valid recipe string in yaml/json format
        """
        if isinstance(path_or_modifiers, Recipe):
            # already a recipe
            return path_or_modifiers

        if isinstance(path_or_modifiers, (Modifier, list)):
            return cls.from_modifiers(
                modifiers=path_or_modifiers, modifier_group_name=modifier_group_name
            )

        if not os.path.isfile(path_or_modifiers):
            # not a local file
            # assume it's a string
            logger.debug(
                "Could not initialize recipe as a file path or zoo stub, "
                "attempting to process as a string."
            )
            logger.debug(f"Input string: {path_or_modifiers}")
            obj = _load_json_or_yaml_string(path_or_modifiers)
            return cls.from_dict(filter_dict(obj, target_stage=target_stage))
        else:
            logger.info(f"Loading recipe from file {path_or_modifiers}")

        with open(path_or_modifiers, "r") as file:
            content = file.read().strip()
            if path_or_modifiers.lower().endswith(".md"):
                content = _parse_recipe_from_md(path_or_modifiers, content)

            if path_or_modifiers.lower().endswith(".json"):
                obj = json.loads(content)
            elif path_or_modifiers.lower().endswith(
                ".yaml"
            ) or path_or_modifiers.lower().endswith(".yml"):
                obj = yaml.safe_load(content)
            else:
                try:
                    obj = _load_json_or_yaml_string(content)
                except ValueError:
                    raise ValueError(
                        f"Could not parse recipe from path {path_or_modifiers}"
                    )
            return cls.from_dict(filter_dict(obj, target_stage=target_stage))

    @classmethod
    def from_dict(cls, recipe_dict: Dict[str, Any]) -> "Recipe":
        """
        Parses a dictionary representing a recipe and returns a Recipe instance.
        Ensures all modifier entries are instantiated Modifier objects.

        :param recipe_dict: Dictionary containing the recipe structure.
        :return: Recipe instance with instantiated Modifier objects.
        """
        args = recipe_dict.get("args", {})
        modifiers: List[Modifier] = []
        stage = "default"

        if not ModifierFactory._loaded:
            ModifierFactory.refresh()

        for stage_key, stage_val in recipe_dict.items():
            if stage_key.endswith("_stage") and isinstance(stage_val, dict):
                stage = stage_key.replace("_stage", "")
                for group_key, group_val in stage_val.items():
                    if group_key.endswith("_modifiers") and isinstance(group_val, dict):
                        inferred_group = group_key.replace("_modifiers", "")
                        for mod_type, mod_args in group_val.items():
                            group = mod_args.get("group", inferred_group)
                            modifier = ModifierFactory.create(
                                mod_type,
                                group=group,
                                allow_registered=True,
                                allow_experimental=True,
                                **mod_args,
                            )
                            modifiers.append(modifier)

        # Validate modifier ordering (e.g., AWQ before quantization)
        _validate_modifier_ordering(modifiers)

        return Recipe(
            args=args,
            stage=stage,
            modifiers=modifiers,
        )

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        :return: A dictionary representation of the recipe
        """

        return get_yaml_serializable_dict(modifiers=self.modifiers, stage=self.stage)

    def yaml(
        self,
        file_path: Optional[str] = None,
        existing_recipe_path: Optional[str] = None,
    ) -> str:
        """
        Return a YAML string representation of the recipe,
        optionally merging with another YAML file.

        :param file_path: Optional path to save YAML
        :param existing_recipe_path: Optional path to another recipe.yaml file
        :return: Combined YAML string
        """
        # Load the other recipe from file, if given
        existing_dict = {}
        if existing_recipe_path:
            with open(existing_recipe_path, "r") as f:
                existing_recipe_str = f.read()
            existing_dict = _load_json_or_yaml_string(existing_recipe_str)

        # Serialize current recipe
        self_dict = get_yaml_serializable_dict(
            modifiers=self.modifiers,
            stage=self.stage,
        )

        # Deep merge — keep both recipe contents
        merged_dict = append_recipe_dict(existing_dict, self_dict)

        # Dump YAML
        file_stream = None if file_path is None else open(file_path, "w")
        yaml_str = yaml.dump(
            merged_dict,
            stream=file_stream,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=None,
            width=88,
        )

        if file_stream:
            file_stream.close()

        return yaml_str


RecipeInput = Union[str, List[str], Recipe, List[Recipe], Modifier, List[Modifier]]
RecipeStageInput = Union[str, List[str], List[List[str]]]
RecipeArgsInput = Union[Dict[str, Any], List[Dict[str, Any]]]
