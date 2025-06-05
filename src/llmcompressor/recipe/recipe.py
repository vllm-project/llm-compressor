import json
import os
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from llmcompressor.modifiers import Modifier, ModifierFactory
from llmcompressor.recipe.utils import (
    _load_json_or_yaml_string,
    _parse_recipe_from_md,
    deep_merge_dicts,
    deep_merge_dicts,
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

    version: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    stage: str = "default"
    modifiers: List[Dict[str, Any]] = Field(default_factory=list)

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
        >>> recipe = Recipe.create_instance(recipe_str)

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
            return Recipe.model_validate(obj)
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
            return Recipe.model_validate(obj)

    @model_validator(mode="before")
    @classmethod
    def parse_from_dict(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        version = values.get("version")
        args = values.get("args", {})
        modifiers: List[Dict[str, Any]] = []
        stage = "default"

        for key, value in values.items():
            if key.endswith("_stage") and isinstance(value, dict):
                stage = key.replace("_stage", "")
                for mod_group_key, mod_defs in value.items():
                    if mod_group_key.endswith("_modifiers") and isinstance(
                        mod_defs, dict
                    ):
                        for mod_type, mod_args in mod_defs.items():
                            modifiers.append(
                                {
                                    "type": mod_type,
                                    "group": stage,
                                    "args": mod_args,
                                }
                            )
        return {
            "version": version,
            "args": args,
            "stage": stage,
            "modifiers": modifiers,
        }

    @staticmethod
    def simplify_recipe(
        recipe: Optional["RecipeInput"] = None,
        target_stage: Optional["RecipeStageInput"] = None,
        override_args: Optional["RecipeArgsInput"] = None,
    ) -> "Recipe":
        """
        Simplify a Recipe by removing stages that are not in the target_stages
        and updating args if overrides are provided

        :param recipe: The Recipe instance to simplify
        :param target_stages: The stages to target when simplifying the recipe
        :param override_args: The arguments used to override existing recipe args
        :return: The simplified Recipe instance
        """
        if recipe is None or (isinstance(recipe, list) and len(recipe) == 0):
            return Recipe()

        # prepare recipe
        else:
            recipe = Recipe.create_instance(recipe)

        # Filter modifiers based on target_stage
        target_modifiers = []
        if target_stage:
            recipe.stage = target_stage
            for modifier in recipe.modifiers:
                if modifier["group"] in target_stage or modifier["group"] == "default":
                    target_modifiers.append(modifier)
            recipe.modifiers = target_modifiers
        # Apply argument overrides if provided
        if override_args:
            recipe.args = {**recipe.args, **override_args}
        return recipe

    def create_modifier(self) -> List[Modifier]:
        """
        Create and return a list of Modifiers for the recipe

        >>> recipe_str = '''
        ... test_stage:
        ...     pruning_modifiers:
        ...         ConstantPruningModifier:
        ...             start: 0.0
        ...             end: 2.0
        ...             targets: ['re:.*weight']
        ... '''
        >>> recipe = Recipe.create_instance(recipe_str)
        >>> modifiers = recipe.create_modifier()
        >>> len(modifiers) == 1
        True

        :return: A list of Modifiers for the recipe
        """
        if not ModifierFactory._loaded:
            ModifierFactory.refresh()
        self.modifiers = [
            modifier
            if isinstance(modifier, Modifier)
            else ModifierFactory.create(
                modifier["type"],
                group=modifier.get("group"),
                allow_registered=True,
                allow_experimental=True,
                **modifier["args"],
            )
            for modifier in self.modifiers
        ]
        return self.modifiers

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
        merged_dict = deep_merge_dicts(existing_dict, self_dict)

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
