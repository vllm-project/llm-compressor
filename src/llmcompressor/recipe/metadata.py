from typing import Any, Dict, List, Optional

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
