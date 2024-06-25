from typing import Any, Generator, Tuple

import torch
from pydantic import Field

from llmcompressor.pytorch.utils.sparsification_info.configs import (
    SparsificationInfo,
    SparsificationPruning,
    SparsificationQuantization,
    SparsificationSummaries,
)


class ModuleSparsificationInfo(SparsificationInfo):
    """
    Pydantic model for storing sparsification information of a torch module.
    """

    summary_info: SparsificationSummaries = Field(
        description="Model that holds the sparsification summary info of the module"
    )
    pruning_info: SparsificationPruning = Field(
        description="Model that holds the pruning info of the module"
    )
    quantization_info: SparsificationQuantization = Field(
        description="Model that holds the quantization info of the module"
    )

    @classmethod
    def from_module(cls, module: torch.nn.Module) -> "ModuleSparsificationInfo":
        """
        Factory method to create a ModuleSparsificationInfo object from a torch module.

        :param module: the module to create the ModuleSparsificationInfo object from
        :return: the ModuleSparsificationInfo object created from the module
        """
        if not isinstance(module, torch.nn.Module):
            raise ValueError(
                "Module must be a torch.nn.Module, not {}".format(type(module))
            )

        return cls(
            summary_info=SparsificationSummaries.from_module(module),
            pruning_info=SparsificationPruning.from_module(module),
            quantization_info=SparsificationQuantization.from_module(module),
        )

    def loggable_items(self, **kwargs) -> Generator[Tuple[str, Any], None, None]:
        """
        A generator that yields the loggable items of
        the ModuleSparsificationInfo object.

        :param kwargs: additional kwargs to pass to the loggable items
        :return a generator that yields a tuple of:
            - the name of the loggable item
            - the value of the loggable item
        """
        for info in [self.summary_info, self.pruning_info, self.quantization_info]:
            yield from info.loggable_items(**kwargs)
