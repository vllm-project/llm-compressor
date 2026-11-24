from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.modifiers import Modifier

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments
    from llmcompressor.pipelines.sequential.restructure import RestructureHandler

__all__ = ["CalibrationPipeline"]


class CalibrationPipeline(ABC, RegistryMixin):
    @staticmethod
    @abstractmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
        restructure_handler: "RestructureHandler",
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
        inferred = standardize_lookup_name(cls._infer_pipeline(modifiers))
        independent = standardize_lookup_name("independent")
        deferred_transform = cls._deferred_transform_precedes_quantization(modifiers)

        if deferred_transform is not None or user == independent:
            inferred = independent

        if user is not None and user != inferred:
            if deferred_transform is not None:
                logger.warning(
                    f"User requested pipeline {user} but when {deferred_transform} "
                    "is followed by a quantization modifier, the independent "
                    "calibration pipeline is recommended. Change or remove the "
                    "pipeline argument in oneshot if you'd like to use the "
                    "recommended pipeline"
                )
            else:
                logger.warning(
                    f"Calibration pipeline is set to `{user}`, but it is recommended "
                    f"to use `{inferred}`"
                )

        pipeline = user or inferred
        return cls.load_from_registry(pipeline)

    @staticmethod
    def _infer_pipeline(modifiers: list[Modifier]) -> str:
        if any(modifier.requires_calibration_data for modifier in modifiers):
            return "sequential"
        else:
            return "datafree"

    @staticmethod
    def _deferred_transform_precedes_quantization(
        modifiers: list[Modifier],
    ) -> str | None:
        """Name a late-applied scaling transform followed by a quantizer."""
        from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
        from llmcompressor.modifiers.transform.awq import AWQModifier
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

        transforms = []
        for modifier in modifiers:
            if isinstance(modifier, (AWQModifier, SmoothQuantModifier)):
                transforms.append(type(modifier).__name__)
            elif transforms and isinstance(modifier, QuantizationMixin):
                return ", ".join(transforms)

        return None
