# ruff: noqa

"""
Provides entry points for model compression workflows.

Includes oneshot compression, training, and pre and post-processing utilities
for model optimization tasks.
"""

from .oneshot import Oneshot, oneshot
from .train import train
from .utils import post_process, pre_process
