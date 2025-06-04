import json
import re
from typing import Any, Dict, List

import yaml

from llmcompressor.modifiers import Modifier


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


def get_yaml_serializable_dict(modifiers: List[Modifier], stage: str) -> Dict[str, Any]:
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
    stage_name = stage + "_stage"
    stage_dict[stage_name] = {}
    for modifier in modifiers:
        group = getattr(modifier, "group", stage) or stage
        group_name = f"{group}_modifiers"
        modifier_type = modifier.__class__.__name__

        args = {
            k: v
            for k, v in modifier.model_dump().items()
            if v is not None and not k.endswith("_")
        }

        if group_name not in stage_dict[stage_name]:
            stage_dict[stage_name][group_name] = {}

        stage_dict[stage_name][group_name][modifier_type] = args

    return stage_dict
