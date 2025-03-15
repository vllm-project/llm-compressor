"""
LLM Compressor Core Events Package

This package provides the core components and lifecycle management for events
used in the LLM Compressor framework. It includes definitions for various
event types and lifecycles that are critical for managing the state and
execution flow of the model compression and training processes.
"""

from .event import Event, EventType

__all__ = ["Event", "EventType"]
