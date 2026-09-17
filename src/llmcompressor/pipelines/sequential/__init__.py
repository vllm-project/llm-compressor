"""Sequential pipeline helpers.

The concrete pipeline is imported by ``llmcompressor.pipelines`` to populate the
pipeline registry. Keeping this package initializer lightweight avoids importing
the pipeline while lower-level modules import sequential offloading utilities.
"""

# ruff: noqa
from .helpers import *
