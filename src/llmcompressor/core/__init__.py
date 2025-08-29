"""
Provides the core compression framework for LLM Compressor.

The core API manages compression sessions, tracks state changes, handles events
during compression, and Provides lifecycle hooks for the compression
process.
"""

from llmcompressor.core.events import Event, EventType
from llmcompressor.core.lifecycle import CompressionLifecycle
from llmcompressor.core.model_layer import ModelParameterizedLayer
from llmcompressor.core.session import CompressionSession
from llmcompressor.core.session_functions import (
    LifecycleCallbacks,
    active_session,
    callbacks,
    create_session,
    reset_session,
)
from llmcompressor.core.state import Data, Hardware, ModifiedState, State

__all__ = [
    "Event",
    "EventType",
    "State",
    "Data",
    "Hardware",
    "ModifiedState",
    "ModelParameterizedLayer",
    "CompressionLifecycle",
    "CompressionSession",
    "create_session",
    "active_session",
    "reset_session",
    "apply",
    "callbacks",
    "LifecycleCallbacks",
]
