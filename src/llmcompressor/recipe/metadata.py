"""
Metadata classes for recipe and model information tracking.

This module defines Pydantic models for capturing and validating metadata about
datasets, parameters, layers, and models used in compression recipes. Provides
structured data containers for recipe configuration and execution tracking.
"""

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "DatasetMetaData",
    "ParamMetaData",
    "LayerMetaData",
    "ModelMetaData",
]


class DatasetMetaData(BaseModel):
    name: str = None
    version: str = None
    hash: str = None
    shape: list[int] = Field(default_factory=list)
    num_classes: int = None
    num_train_samples: int = None
    num_val_samples: int = None
    num_test_samples: int = None


class ParamMetaData(BaseModel):
    name: str = None
    shape: list[int] = None
    weight_hash: str = None


class LayerMetaData(BaseModel):
    name: str = None
    type: str = None
    index: int = None
    attributes: dict[str, Any] = None
    input_shapes: list[list[int]] = None
    output_shapes: list[list[int]] = None
    params: dict[str, ParamMetaData] = None


class ModelMetaData(BaseModel):
    architecture: str = None
    sub_architecture: str = None
    input_shapes: list[list[int]] = None
    output_shapes: list[list[int]] = None
    layers: list[LayerMetaData] = Field(default_factory=list)
    layer_prefix: str | None = None
