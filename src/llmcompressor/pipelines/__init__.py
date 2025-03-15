from typing import Callable, Any

from .basic import run_pipeline as run_basic
from .sequential import run_pipeline as run_sequential
from .layer_sequential import run_pipeline as run_layer_sequential


def get_pipeline(pipeline: str) -> Callable[[Any], Any]:
    if pipeline == "basic":
        return run_basic
    
    if pipeline == "sequential":
        return run_sequential
    
    if pipeline == "layer_sequential":
        return run_layer_sequential
    
    raise ValueError(f"Unknown pipeline {pipeline}")