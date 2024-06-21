from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

__all__ = [
    "DatasetMetaData",
    "ParamMetaData",
    "LayerMetaData",
    "ModelMetaData",
    "RecipeMetaData",
]


class DatasetMetaData(BaseModel):
    name: str = None
    version: str = None
    hash: str = None
    shape: List[int] = Field(default_factory=list)
    num_classes: int = None
    num_train_samples: int = None
    num_val_samples: int = None
    num_test_samples: int = None


class ParamMetaData(BaseModel):
    name: str = None
    shape: List[int] = None
    weight_hash: str = None


class LayerMetaData(BaseModel):
    name: str = None
    type: str = None
    index: int = None
    attributes: Dict[str, Any] = None
    input_shapes: List[List[int]] = None
    output_shapes: List[List[int]] = None
    params: Dict[str, ParamMetaData] = None


class ModelMetaData(BaseModel):
    architecture: str = None
    sub_architecture: str = None
    input_shapes: List[List[int]] = None
    output_shapes: List[List[int]] = None
    layers: List[LayerMetaData] = Field(default_factory=list)
    layer_prefix: Optional[str] = None


class RecipeMetaData(BaseModel):
    domain: str = None
    task: str = None
    requirements: List[str] = None
    tags: List[str] = None
    target_dataset: DatasetMetaData = None
    target_model: ModelMetaData = None

    def update_missing_metadata(self, other: "RecipeMetaData"):
        """
        Update recipe metadata with missing values from another
        recipe metadata instance

        :param other: the recipe metadata to update with
        """
        self.domain = self.domain or other.domain
        self.task = self.task or other.task
        self.requirements = self.requirements or other.requirements
        self.tags = self.tags or other.tags
        self.target_dataset = self.target_dataset or other.target_dataset
        self.target_model = self.target_model or other.target_model
