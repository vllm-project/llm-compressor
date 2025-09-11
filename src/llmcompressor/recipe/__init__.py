"""
Recipe system for defining and managing compression workflows.

Provides the recipe framework for specifying compression
configurations, including metadata tracking, recipe parsing, and
workflow orchestration. Supports stage-based execution and flexible
parameter management for complex compression pipelines.
"""

from .metadata import DatasetMetaData, LayerMetaData, ModelMetaData, ParamMetaData
from .recipe import Recipe, RecipeArgsInput, RecipeInput, RecipeStageInput

__all__ = [
    "DatasetMetaData",
    "ParamMetaData",
    "LayerMetaData",
    "ModelMetaData",
    "Recipe",
    "RecipeInput",
    "RecipeStageInput",
    "RecipeArgsInput",
]
