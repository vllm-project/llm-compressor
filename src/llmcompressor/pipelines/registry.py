from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.obcq.sgpt_mixin import SparsityModifierMixin
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationMixin
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["CalibrationPipeline"]

SEQUENTIAL_MODIFIERS = (GPTQModifier, SparsityModifierMixin)


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
        inferred = standardize_lookup_name(cls._validate_infer_pipeline(modifiers))
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
    def _validate_infer_pipeline(modifiers: List[Modifier]) -> str:
        if any(isinstance(modifier, AWQModifier) for modifier in modifiers):
            if len(modifiers) > 1:
                logger.warning(
                    "AWQ does not currently support sharing a data pipeline with other "
                    "modifiers. Inferring `independent` calibration pipeline"
                )
                return "independent"
            return "datafree"

        if any(isinstance(modifier, SEQUENTIAL_MODIFIERS) for modifier in modifiers):
            return "sequential"

        active_qmods = _get_active_quant_modifiers(modifiers)
        if len(active_qmods) > 1:
            raise ValueError(
                f"Recipe contains more than one active quantization config "
                f"({active_qmods}). These configs may be conflicting, Please modify "
                "your recipe to use at most one quantization config"
            )

        if len(active_qmods) == 1:
            quant_modifier = active_qmods[0]
            config = quant_modifier.resolve_quantization_config()
            if config.requires_calibration_data():
                return "basic"
            else:
                return "datafree"

        if any(isinstance(modifier, SmoothQuantModifier) for modifier in modifiers):
            return "basic"

        return "datafree"


def _get_active_quant_modifiers(modifiers: List[Modifier]) -> List[QuantizationMixin]:
    return [
        modifier
        for modifier in modifiers
        if isinstance(modifier, QuantizationMixin) and modifier.has_config()
    ]
