from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.pruning.sparsegpt.sgpt_base import SparsityModifierBase
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

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
        cls, modifiers: List[Modifier], user: Optional[str] = None
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

        if user == independent:
            inferred = independent

        if user is not None and user != inferred:
            logger.warning(
                f"Calibration pipeline is set to `{user}`, but it is recommended to "
                f"use `{inferred}`"
            )

        pipeline = user or inferred
        return cls.load_from_registry(pipeline)

    @staticmethod
    def _infer_pipeline(modifiers: List[Modifier]) -> str:
        def _modifier_requires_calibration(modifier: Modifier):
            if isinstance(
                modifier,
                (SmoothQuantModifier, SparsityModifierBase, GPTQModifier, AWQModifier),
            ):
                return True
            elif isinstance(modifier, QuantizationModifier):
                config = modifier.resolve_quantization_config()
                return config.requires_calibration_data()
            else:
                return False

        if any(_modifier_requires_calibration(modifier) for modifier in modifiers):
            return "sequential"
        else:
            return "datafree"
