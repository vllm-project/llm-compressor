"""
Helper functions and utilities for pruning modifiers.

Provides scheduling functions, factory classes, and configuration
utilities for pruning operations. Includes linear and cubic schedulers,
custom scheduler creation, and settings management for structured and
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict

from llmcompressor.core import Event, State

__all__ = [
    "PruningCreateSettings",
    "SchedulerCalculationType",
    "CreateSchedulerType",
    "PruningSchedulerFactory",
    "create_custom_scheduler",
    "linear_scheduler",
    "cubic_scheduler",
    "polynomial_decay_scheduler",
    "polynomial_scheduler",
    "multi_step_scheduler",
]


@dataclass
class PruningCreateSettings:
    start: float
    end: float
    update: float
    init_sparsity: float
    final_sparsity: float
    args: Dict[str, Any]


SchedulerCalculationType = Callable[[Event, State], float]
CreateSchedulerType = Callable[[PruningCreateSettings], SchedulerCalculationType]


class PruningSchedulerFactory:
    registry = {}  # type: Dict[str, CreateSchedulerType]

    @staticmethod
    def register(name: str, func: CreateSchedulerType):
        PruningSchedulerFactory.registry[name] = func

    @staticmethod
    def register_decorator(name: str):
        def inner(func: CreateSchedulerType):
            PruningSchedulerFactory.registry[name] = func
            return func

        return inner

    @staticmethod
    def create_scheduler(
        scheduler_type: str, settings: PruningCreateSettings
    ) -> SchedulerCalculationType:
        if scheduler_type in PruningSchedulerFactory.registry:
            return PruningSchedulerFactory.registry[scheduler_type](settings)
        elif scheduler_type.startswith("calc(") and scheduler_type.endswith(")"):
            return create_custom_scheduler(scheduler_type, settings)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_custom_scheduler(
    scheduler_type: str, settings: PruningCreateSettings
) -> SchedulerCalculationType:
    pattern = re.compile(r"calc\(([^()]*)\)")
    match = pattern.search(scheduler_type)

    if not match:
        raise ValueError(f"invalid calc string {scheduler_type}")

    inner_expr = match.group(1)

    def _schedule(event: Event, state: State):
        return eval(
            inner_expr,
            {"math": math},
            {
                "start": settings.start,
                "end": settings.end,
                "update": settings.update,
                "init_sparsity": settings.init_sparsity,
                "final_sparsity": settings.final_sparsity,
                **(settings.args if settings.args else {}),
                "index": event.current_index,
            },
        )

    return _schedule


@PruningSchedulerFactory.register_decorator("linear")
def linear_scheduler(settings: PruningCreateSettings) -> SchedulerCalculationType:
    def _schedule(event: Event, state: State) -> float:
        per_complete = (event.current_index - settings.start) / (
            settings.end - settings.start
        )

        return (
            settings.init_sparsity
            + (settings.final_sparsity - settings.init_sparsity) * per_complete
        )

    return _schedule


@PruningSchedulerFactory.register_decorator("cubic")
def cubic_scheduler(settings: PruningCreateSettings) -> SchedulerCalculationType:
    settings.args = {"exponent": 3}

    return polynomial_decay_scheduler(settings)


@PruningSchedulerFactory.register_decorator("polynomial_decay")
def polynomial_decay_scheduler(
    settings: PruningCreateSettings,
) -> SchedulerCalculationType:
    args = settings.args if settings.args else {}
    exponent = args.get("exponent", 2)

    def _schedule(event: Event, state: State) -> float:
        per_complete = (event.current_index - settings.start) / (
            settings.end - settings.start
        )

        scaled_complete = pow(per_complete - 1, exponent) + 1

        return (
            settings.init_sparsity
            + (settings.final_sparsity - settings.init_sparsity) * scaled_complete
        )

    return _schedule


@PruningSchedulerFactory.register_decorator("polynomial")
def polynomial_scheduler(settings: PruningCreateSettings) -> SchedulerCalculationType:
    args = settings.args if settings.args else {}
    exponent = args.get("exponent", 2)

    def _schedule(event: Event, state: State) -> float:
        per_complete = (event.current_index - settings.start) / (
            settings.end - settings.start
        )
        scaled_complete = per_complete**exponent

        return (
            settings.init_sparsity
            + (settings.final_sparsity - settings.init_sparsity) * scaled_complete
        )

    return _schedule


@PruningSchedulerFactory.register_decorator("multi_step")
def multi_step_scheduler(settings: PruningCreateSettings) -> SchedulerCalculationType:
    args = settings.args if settings.args else {}
    steps = args.get("steps", [])
    steps = sorted(steps, key=lambda x: x[0])

    def _schedule(event: Event, state: State) -> float:
        current_sparsity = settings.init_sparsity

        for index, sparsity in steps:
            if event.current_index >= index:
                current_sparsity = sparsity

        return current_sparsity

    return _schedule
