# ruff: noqa

"""
Provides entry points for model compression workflows.

Includes oneshot compression, training, and pre and post-processing utilities
for model optimization tasks.
"""

from .oneshot import Oneshot, oneshot
from .train import train
from .model_free import model_free_ptq
from .utils import post_process, pre_process
