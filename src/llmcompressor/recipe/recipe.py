import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import Field, model_validator, BaseModel, ConfigDict

from llmcompressor.modifiers import Modifier, ModifierFactory

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

    @staticmethod
    def simplify_combine_recipes(
        recipes: List[Union[str, "Recipe"]],
    ) -> "Recipe":
        """
        A method to combine multiple recipes into one recipe
        Automatically calculates the start and end of the combined recipe
        and shifts the start and end of the recipes accordingly

        :param recipes: The list of Recipe instances to combine
        :return: The combined Recipe instance
        """

        combined = Recipe()
        for recipe in recipes:
            simplified = Recipe.simplify_recipe(
                recipe=recipe,
            )
            combined.version = simplified.version
            combined.modifiers.extend(simplified.modifiers)
            combined.args.update(simplified.args)

        return combined

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
        >>> len(stage_modifiers) == 1
        True
        >>> len(stage_modifiers[0].modifiers) == 1
        True

        :return: A list of Modifiers for the recipe
        """
        if not ModifierFactory._loaded:
            ModifierFactory.refresh()
        return [
            modifier if isinstance(modifier, Modifier)
            else ModifierFactory.create(
                modifier["type"],
                allow_registered=True,
                allow_experimental=True,
                **modifier["args"],
            )
            for modifier in self.modifiers
        ]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        :return: A dictionary representation of the recipe
        """
        dict_ = super().model_dump(*args, **kwargs)
        stages = {}

        for stage in dict_["stages"]:
            name = f"{stage['group']}_stage"
            del stage["group"]

            if name not in stages:
                stages[name] = []

            stages[name].append(stage)

        dict_["stages"] = stages

        return dict_

    def yaml(self, file_path: Optional[str] = None) -> str:
        """
        Return a yaml string representation of the recipe.

        :param file_path: optional file path to save yaml to
        :return: The yaml string representation of the recipe
        """
        file_stream = None if file_path is None else open(file_path, "w")
        yaml_dict = get_yaml_serializable_dict(modifiers=self.modifiers)

        ret = yaml.dump(
            yaml_dict,
            stream=file_stream,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=None,
            width=88,
        )

        if file_stream is not None:
            file_stream.close()

        return ret


RecipeInput = Union[str, List[str], Recipe, List[Recipe], Modifier, List[Modifier]]
RecipeStageInput = Union[str, List[str], List[List[str]]]
RecipeArgsInput = Union[Dict[str, Any], List[Dict[str, Any]]]


def _load_json_or_yaml_string(content: str) -> Dict[str, Any]:
    # try loading as json first, then yaml
    # if both fail, raise a ValueError
    try:
        ret = json.loads(content)
    except json.JSONDecodeError:
        try:
            ret = yaml.safe_load(content)
        except yaml.YAMLError as err:
            raise ValueError(f"Could not parse recipe from string {content}") from err

    if not isinstance(ret, dict):
        raise ValueError(
            f"Could not parse recipe from string {content}. If you meant load from "
            "a file, please make sure that the specified file path exists"
        )
    return ret


def _parse_recipe_from_md(file_path, yaml_str):
    """
    extract YAML front matter from markdown recipe card. Copied from
    llmcompressor.optim.helpers:_load_yaml_str_from_file
    :param file_path: path to recipe file
    :param yaml_str: string read from file_path
    :return: parsed yaml_str with README info removed
    """
    # extract YAML front matter from markdown recipe card
    # adapted from
    # https://github.com/jonbeebe/frontmatter/blob/master/frontmatter
    yaml_delim = r"(?:---|\+\+\+)"
    yaml = r"(.*?)"
    re_pattern = r"^\s*" + yaml_delim + yaml + yaml_delim
    regex = re.compile(re_pattern, re.S | re.M)
    result = regex.search(yaml_str)

    if result:
        yaml_str = result.group(1)
    else:
        # fail if we know whe should have extracted front matter out
        raise RuntimeError(
            "Could not extract YAML front matter from recipe card:" " {}".format(
                file_path
            )
        )
    return yaml_str


def get_yaml_serializable_dict(modifiers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    This function is used to convert a list of modifiers into a dictionary
    where the keys are the group names and the values are the modifiers
    which in turn are dictionaries with the modifier type as the key and
    the modifier args as the value.
    This is needed to conform to our recipe structure during yaml serialization
    where each stage, modifier_groups, and modifiers are represented as
    valid yaml dictionaries.

    Note: This function assumes that modifier groups do not contain the same
    modifier type more than once in a group. This assumption is also held by
    Recipe.create_instance(...) method.

    :param modifiers: A list of dictionaries where each dictionary
        holds all information about a modifier
    :return: A dictionary where the keys are the group names and the values
        are the modifiers which in turn are dictionaries with the modifier
        type as the key and the modifier args as the value.
    """
    stage_dict = {}
    for modifier in modifiers:
        # Handle dict-style modifier
        if isinstance(modifier, dict):
            group = modifier["group"]
            modifier_type = modifier["type"]
            args = modifier["args"]
        # Handle object-style modifier
        else:
            group = getattr(modifier, "group", "default")
            modifier_type = modifier.__class__.__name__
            args = {
                k: v for k, v in modifier.__dict__.items()
                if not k.endswith("_") and not k.startswith("__")
            }
        if group not in stage_dict:
            stage_dict[group] = {}
        stage_dict[group][modifier_type] = args
    return stage_dict
