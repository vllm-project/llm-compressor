from typing import Dict, Optional

from compressed_tensors import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.quantization.utils import is_model_quantized
from torch import Tensor
from torch.nn import Module

from llmcompressor.core import active_session
from llmcompressor.pytorch.utils import ModuleSparsificationInfo
from llmcompressor.transformers.compression.helpers import (
    infer_sparsity_structure_from_model,
    infer_sparsity_structure_from_stage_modifiers,
)


class SparsityConfigMetadata:
    """
    Class of helper functions for filling out a SparsityCompressionConfig with readable
    metadata from the model
    """

    @staticmethod
    def infer_global_sparsity(
        model: Module, state_dict: Optional[Dict[str, Tensor]] = None
    ) -> float:
        """
        Calculates the global percentage of sparse zero weights in the model

        :param model: pytorch model to infer sparsity of
        :param state_dict: optional state_dict to replace that in model, used for
        gathering global FSDP model info
        :return: global sparsity of model
        """

        info = ModuleSparsificationInfo(model, state_dict=state_dict)
        global_sparsity = info.params_sparse_percent / 100.0  # convert % to float
        return global_sparsity

    @staticmethod
    def infer_sparsity_structure(model: Optional[Module] = None) -> str:
        """
        Determines what sparsity structure, if any, was applied.

        First, there is an attempt to dedue the sparsity structure
        from the currently active sparse session.

        If that fails, the sparsity structure is inferred from the
        model (if provided)

        Finally, if both fail, the sparsity structure is set to
        "unstructured"

        :return: sparsity structure as a string
        """
        sparsity_structure = None

        current_session = active_session()
        stage_modifiers = current_session.lifecycle.modifiers
        if stage_modifiers:
            sparsity_structure = infer_sparsity_structure_from_stage_modifiers(
                stage_modifiers
            )

        if model and sparsity_structure is None:
            sparsity_structure = infer_sparsity_structure_from_model(model)

        return sparsity_structure or "unstructured"

    @staticmethod
    def from_pretrained(
        model: Module,
        state_dict: Optional[Dict[str, Tensor]] = None,
        compress: bool = False,
    ) -> Optional["SparsityCompressionConfig"]:
        """
        Determines compression type and informational parameters for a given model

        :param model: pytorch model to calculate sparsity config for
        :param state_dict: optional state_dict to replace that in model, used for
        gathering global FSDP model info
        :param compress: whether or not to compress the model on disk
        :return: compression config inferred from the model
        """

        global_sparsity = SparsityConfigMetadata.infer_global_sparsity(
            model, state_dict=state_dict
        )

        if global_sparsity < 0.05:
            return None

        sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure(
            model=model
        )
        if is_model_quantized(model):
            # compressing a sparse quantized model is not supported yet
            format = CompressionFormat.dense.value
        elif compress:
            format = CompressionFormat.sparse_bitmask.value
        else:
            format = CompressionFormat.dense.value

        return SparsityCompressionConfig.load_from_registry(
            format,
            global_sparsity=global_sparsity,
            sparsity_structure=sparsity_structure,
        )

    @staticmethod
    def fill_config_details(
        config: SparsityCompressionConfig,
        model: Module,
        state_dict: Optional[Dict[str, Tensor]] = None,
    ):
        """
        Fills in informational sparsity parameters from a given model

        :param config: sparsity config to fill in
        :param model: pytorch model to infer config parameters from
        :param state_dict: optional state_dict to replace that in model, used for
        gathering global FSDP model info
        """
        config.global_sparsity = SparsityConfigMetadata.infer_global_sparsity(
            model, state_dict=state_dict
        )
        config.sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure()
