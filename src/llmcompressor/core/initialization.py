"""Pipeline-owned initialization hooks for compression modifiers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from llmcompressor.core.state import State
    from llmcompressor.modifiers import Modifier

__all__ = [
    "ModuleScopedInitializationFunction",
    "PipelineInitializationContext",
    "PipelineInitializationFunction",
    "PipelineInitializationPlan",
]


@dataclass(frozen=True)
class PipelineInitializationContext:
    """Inputs available when a pipeline runs one modifier initializer."""

    state: State
    model: torch.nn.Module
    modules: tuple[torch.nn.Module, ...] | None = None
    pipeline: str | None = None
    initialization_kwargs: Mapping[str, Any] = field(default_factory=dict)


PipelineInitializationFunction = Callable[[PipelineInitializationContext], None]


@dataclass(frozen=True)
class ModuleScopedInitializationFunction:
    """An initialization function that runs once for each pipeline module scope."""

    function: PipelineInitializationFunction


PipelineInitializationStep = (
    PipelineInitializationFunction | ModuleScopedInitializationFunction
)


class PipelineInitializationPlan:
    """Ordered modifier initialization functions for a calibration pipeline.

    Model-wide functions run once, while module-scoped functions run once for each
    newly encountered module scope. Each modifier's initial start event follows its
    setup functions, preserving recipe order. The selected pipeline decides when and
    with what module scope these functions run.
    """

    def __init__(
        self,
        functions: Iterable[PipelineInitializationStep] = (),
        *,
        initialization_kwargs: Mapping[str, Any] | None = None,
        requires_eager_initialization: bool = False,
    ):
        self.functions: tuple[PipelineInitializationStep, ...] = tuple(functions)
        self.initialization_kwargs = dict(initialization_kwargs or {})
        self.requires_eager_initialization = requires_eager_initialization
        self._initialized_functions: set[int] = set()
        self._initialized_scoped_modules: dict[int, set[torch.nn.Module]] = {}

    @classmethod
    def from_modifiers(
        cls,
        modifiers: Iterable[Modifier],
        state: State,
        **kwargs,
    ) -> PipelineInitializationPlan:
        functions = []
        requires_eager_initialization = False
        for modifier in modifiers:
            functions.extend(modifier.get_initialization_functions(state, **kwargs))
            functions.extend(
                ModuleScopedInitializationFunction(function)
                for function in modifier.get_module_initialization_functions(
                    state, **kwargs
                )
            )
            functions.append(
                ModuleScopedInitializationFunction(modifier.start_pipeline)
            )
            requires_eager_initialization |= modifier.requires_eager_initialization
        return cls(
            functions,
            initialization_kwargs=kwargs,
            requires_eager_initialization=requires_eager_initialization,
        )

    def initialize(
        self,
        state: State,
        model: torch.nn.Module,
        *,
        modules: Iterable[torch.nn.Module] | None = None,
        pipeline: str | None = None,
    ) -> None:
        """Run initialization functions in recipe order for the given scope."""
        context = PipelineInitializationContext(
            state=state,
            model=model,
            modules=None if modules is None else tuple(modules),
            pipeline=pipeline,
            initialization_kwargs=self.initialization_kwargs,
        )
        for initialization_function in self.functions:
            if isinstance(initialization_function, ModuleScopedInitializationFunction):
                function = initialization_function.function
                function_id = id(function)
                if function_id in self._initialized_functions:
                    continue
                if context.modules is None:
                    function(context)
                    self._initialized_functions.add(function_id)
                    continue

                initialized_modules = self._initialized_scoped_modules.setdefault(
                    function_id, set()
                )
                new_modules = tuple(
                    module
                    for module in context.modules
                    if module not in initialized_modules
                )
                if not new_modules:
                    continue
                function(replace(context, modules=new_modules))
                initialized_modules.update(new_modules)
                continue

            function_id = id(initialization_function)
            if function_id in self._initialized_functions:
                continue
            initialization_function(replace(context, modules=None))
            self._initialized_functions.add(function_id)
