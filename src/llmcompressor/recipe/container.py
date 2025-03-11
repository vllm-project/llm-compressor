from dataclasses import dataclass, field
from typing import List, Optional

from llmcompressor.modifiers import Modifier
from llmcompressor.recipe.recipe import (
    Recipe,
    RecipeArgsInput,
    RecipeInput,
    RecipeStageInput,
    RecipeTuple,
)

__all__ = ["RecipeContainer"]


@dataclass
class RecipeContainer:
    """
    A container for recipes to be used in a session. Provides utilities
    to update the recipes and compile them into a single recipe.

    :param compiled_recipe: the compiled recipe from the recipes list
    :param recipes: the list of RecipeTuple instances to be compiled
    :param applied_stages: list of recipe stages that have already been applied
    """

    compiled_recipe: Optional[Recipe] = None
    recipes: List[RecipeTuple] = field(default_factory=list)
    applied_stages: List[str] = field(default_factory=list)

    def prepend(
        self,
        recipe: Optional[RecipeInput] = None,
        recipe_stage: Optional[RecipeStageInput] = None,
        recipe_args: Optional[RecipeArgsInput] = None,
    ):
        recipe_tuples = self._prepare_tuples(recipe, recipe_stage, recipe_args)
        self.recipes = recipe_tuples + self.recipes
        self._check_compile_recipe()

    def append(
        self,
        recipe: Optional[RecipeInput] = None,
        recipe_stage: Optional[RecipeStageInput] = None,
        recipe_args: Optional[RecipeArgsInput] = None,
    ):
        recipe_tuples = self._prepare_tuples(recipe, recipe_stage, recipe_args)
        self.recipes = self.recipes + recipe_tuples
        self._check_compile_recipe()

    def get_modifiers(self) -> List[Modifier]:
        if self.compiled_recipe is None:
            return []

        return self.compiled_recipe.create_modifier()

    def _prepare_tuples(
        self,
        recipe: Optional[RecipeInput] = None,
        recipe_stage: Optional[RecipeStageInput] = None,
        recipe_args: Optional[RecipeArgsInput] = None,
    ) -> List[RecipeTuple]:
        if recipe is None or (isinstance(recipe, list) and len(recipe) == 0):
            return []

        # prepare recipe
        if isinstance(recipe, Modifier) or (
            isinstance(recipe, list)
            and all(isinstance(mod, Modifier) for mod in recipe)
        ):
            recipe = Recipe.create_instance(recipe)

        if not isinstance(recipe, list):
            recipe = [recipe]

        recipe = [
            Recipe.create_instance(rec) if isinstance(rec, str) else rec
            for rec in recipe
        ]

        # prepare stage
        if recipe_stage is None:
            recipe_stage = [None] * len(recipe)
        else:
            if not isinstance(recipe_stage, list):
                recipe_stage = [[recipe_stage]] * len(recipe)
            if not isinstance(recipe_stage[0], list):
                recipe_stage = [recipe_stage] * len(recipe)

        # prepare args
        if recipe_args is None:
            recipe_args = [{}] * len(recipe)
        elif not isinstance(recipe_args, list):
            recipe_args = [recipe_args] * len(recipe)

        # validation
        if len(recipe) != len(recipe_stage) or len(recipe) != len(recipe_args):
            raise ValueError(
                "recipe, recipe_stage, and recipe_args must be the same length"
            )

        # create tuples
        return [
            RecipeTuple(rec, stage, args)
            for rec, stage, args in zip(recipe, recipe_stage, recipe_args)
        ]

    def update_applied_stages(self, new_stages: List[str]):
        """
        Updates the applied_stages list with new stages, indicating their structure
        has already been applied

        :param new_stages: new stage names to add
        """
        for stage in new_stages:
            if stage not in self.applied_stages:
                self.applied_stages.append(stage)

    def _check_compile_recipe(self):
        """
        Check if the recipes need to be compiled into a single recipe and
        compile them if they do.

        :return: True if the recipes were compiled, False otherwise
        """
        if self.recipes:
            self.compiled_recipe = Recipe.simplify_combine_recipes(self.recipes)

    def check_any_recipe_exists(self) -> bool:
        """
        Checks if any recipes have been added to the container, compiled or not

        :return: True if any recipes exist in the container, False otherwise
        """
        if self.compiled_recipe is not None:
            return True
        if len(self.recipes) > 0:
            return True

        return False
