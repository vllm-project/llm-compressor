from typing import Any, Callable

from .basic import run_pipeline as run_basic
from .layer_sequential import run_pipeline as run_layer_sequential
from .sequential import run_pipeline as run_sequential


def get_pipeline(pipeline: str) -> Callable[[Any], Any]:
    if pipeline == "basic":
        return run_basic

    if pipeline == "sequential":
        return run_sequential

    if pipeline == "layer_sequential":
        return run_layer_sequential

    raise ValueError(f"Unknown pipeline {pipeline}")
