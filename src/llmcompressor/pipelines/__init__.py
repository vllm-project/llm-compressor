# ruff: noqa

"""
Compression pipelines for orchestrating different compression strategies.

Provides various compression pipelines including basic, sequential,
independent, layer-sequential, and data-free approaches. Each pipeline
coordinates different compression techniques and workflows for optimal
model optimization based on specific requirements and constraints.
"""

# populate registry
from .basic import *
from .data_free import *
from .independent import *
from .layer_sequential import *
from .registry import *
from .sequential import *
