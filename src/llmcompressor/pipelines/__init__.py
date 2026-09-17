# ruff: noqa

"""
Compression pipelines for orchestrating different compression strategies.

Provides various compression pipelines including basic, sequential,
independent, layer-sequential, and data-free approaches. Each pipeline
coordinates different compression techniques and workflows for optimal
model optimization based on specific requirements and constraints.
"""

# Populate the registry by importing the concrete pipeline modules directly.
from .basic.pipeline import *
from .data_free.pipeline import *
from .independent.pipeline import *
from .registry import *
from .sequential.pipeline import *
