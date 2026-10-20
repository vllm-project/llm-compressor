from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.modifiers import Modifier

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["CalibrationPipeline"]


class CalibrationPipeline(ABC, RegistryMixin):
    @staticmethod
    @abstractmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        raise NotImplementedError()

    @classmethod
    def from_modifiers(
        cls, modifiers: list[Modifier], user: str | None = None
    ) -> "CalibrationPipeline":
        """
        Infer which calibration pipeline to use based on the available modifiers and
        any user specifications

        :param modifiers: modifiers to apply to model
        :param user: pipeline name passed by user
        :return: CalibrationPipeline instance to be called with data (if not datafree)
        """
        user = standardize_lookup_name(user) if user else None
        independent = standardize_lookup_name("independent")

        if user == independent:
            return cls.load_from_registry(independent)

        required = {
            modifier.required_pipeline
            for modifier in modifiers
            if modifier.required_pipeline is not None
        }
        if len(required) > 1:
            raise ValueError(
                f"Recipe modifiers require incompatible pipelines: {sorted(required)}. "
                "Run them in separate stages or use pipeline='independent'."
            )
        if required and user is not None and user not in required:
            raise ValueError(
                f"This recipe requires pipeline={next(iter(required))!r}, got {user!r}."
            )

        inferred = next(iter(required)) if required else cls._infer_pipeline(modifiers)
        inferred = standardize_lookup_name(inferred)

        if user is not None and user != inferred:
            logger.warning(
                f"Calibration pipeline is set to `{user}`, but it is recommended to "
                f"use `{inferred}`"
            )

        pipeline = user or inferred
        return cls.load_from_registry(pipeline)

    @staticmethod
    def _infer_pipeline(modifiers: list[Modifier]) -> str:
        if any(modifier.requires_calibration_data for modifier in modifiers):
            return "sequential"
        else:
            return "datafree"
