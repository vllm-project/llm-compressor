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
