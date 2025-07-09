from typing import Dict, List, Optional

from compressed_tensors import CompressionFormat, SparsityCompressionConfig
from compressed_tensors.config import SparsityStructure
from compressed_tensors.quantization import QuantizationType
from compressed_tensors.quantization.utils import (
    is_model_quantized,
    is_module_quantized,
)
from loguru import logger
from torch import Tensor
from torch.nn import Module

from llmcompressor.core import active_session
from llmcompressor.pytorch.utils import ModuleSparsificationInfo
from llmcompressor.transformers.compression.helpers import (
    infer_sparse_targets_and_ignores,
    infer_sparsity_structure_from_model,
    infer_sparsity_structure_from_modifiers,
)


class SparsityConfigMetadata:
    """
    Class of helper functions for filling out a SparsityCompressionConfig with readable
    metadata from the model
    """

    SPARSITY_THRESHOLD: float = 0.49

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
    def infer_sparsity_structure(
        model: Optional[Module] = None, check_only_modifiers: Optional[bool] = False
    ) -> str:
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
        stage_modifiers = current_session.lifecycle.recipe.modifiers
        if stage_modifiers:
            sparsity_structure = infer_sparsity_structure_from_modifiers(
                stage_modifiers
            )

        if check_only_modifiers:
            return sparsity_structure

        if model and sparsity_structure is None:
            sparsity_structure = infer_sparsity_structure_from_model(model)

        return SparsityStructure(sparsity_structure).value

    @staticmethod
    def from_pretrained(
        model: Module,
        state_dict: Optional[Dict[str, Tensor]] = None,
        compress: bool = False,
        quantization_format: Optional[CompressionFormat] = None,
        disable_sparse_compression: bool = False,
        sparsity_structure: Optional[str] = None,
    ) -> Optional["SparsityCompressionConfig"]:
        """
        Determines compression type and informational parameters for a given model

        :param model: pytorch model to calculate sparsity config for
        :param state_dict: optional state_dict to replace that in model, used for
        gathering global FSDP model info
        :param compress: whether or not to compress the model on disk
        :param quantization_format: the quantization compression format being used
            for the model
        :param disable_sparse_compression: whether or not to compress the model with
            sparse compressors, If True, the sparse compression format will
            be dense, default is False.
        :param sparsity_structure: sparsity structure for the model. Providing it as
            input will skip the step to infer it from the model directly
        :return: compression config inferred from the model
        """
        # TODO: can we remove this? Do we need the state dict?
        global_sparsity = SparsityConfigMetadata.infer_global_sparsity(
            model, state_dict=state_dict
        )

        if sparsity_structure is None:
            sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure(
                model=model
            )

        if (
            disable_sparse_compression
            or quantization_format == CompressionFormat.marlin_24
        ):
            # sparse compressor should be dense
            # when no_sparse_compression is True
            # or when marlin_24 is used
            format = CompressionFormat.dense.value
        elif compress and SparsityConfigMetadata.is_sparse24_bitmask_supported(
            model, sparsity_structure
        ):
            format = CompressionFormat.sparse_24_bitmask.value
        else:
            format = CompressionFormat.dense.value

        # TODO: eventually should be done similar to quantization
        # so we do not have to infer
        targets, ignores = infer_sparse_targets_and_ignores(
            model,
            sparsity_structure=sparsity_structure,
            sparsity_threshold=SparsityConfigMetadata.SPARSITY_THRESHOLD,
        )

        if not (targets or ignores):
            # no sparsity config
            # needed if targets/ignores are empty
            return None

        return SparsityCompressionConfig.load_from_registry(
            format,
            global_sparsity=global_sparsity,
            sparsity_structure=sparsity_structure,
            targets=targets,
            ignore=ignores,
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

    @staticmethod
    def is_sparse24_bitmask_supported(
        model: Module,
        sparsity_structure: Optional[str] = None,
    ) -> bool:
        """
        Determines if sparse 24 bitmask sparse compressor is supported for a given model
        and its sparsity structure in vLLM

        :param model: pytorch model to check for sparse 24 bit sparsity support
        :param sparsity_structure: sparsity structure of the model, if
            not supplied it will be inferred
        :return: whether or not sparse 24 bitmask compression is supported
            in vLLM for the given model
        """
        if sparsity_structure is None:
            sparsity_structure = SparsityConfigMetadata.infer_sparsity_structure(model)

        if sparsity_structure != SparsityStructure.TWO_FOUR.value:
            # only supported for 2:4 sparsity
            return False

        if not is_model_quantized(model):
            logger.warning(
                "Compressed Sparse-only 2:4 models are not supported in vLLM<=0.7.0, "
                "consider saving with `disable_sparse_compression` set, "
                "`model.save_pretrained(..., disable_sparse_compression=True)`"
            )
            return True

        # when model is quantized, and has 2:4 sparsity

        supported_scheme_types: List[str] = [
            QuantizationType.INT.value,
            QuantizationType.FLOAT.value,
        ]

        for submodule in model.modules():
            if not is_module_quantized(submodule):
                continue

            weight_scheme = submodule.quantization_scheme.weights
            input_scheme = submodule.quantization_scheme.input_activations

            if weight_scheme and input_scheme:
                # weight and activation quantization
                # check schemes are supported
                for scheme in [weight_scheme, input_scheme]:
                    scheme_supported = (
                        scheme.num_bits == 8 and scheme.type in supported_scheme_types
                    )
                    if not scheme_supported:
                        logger.info(
                            "Quantization scheme not supported,"
                            " turning off sparse 24 compression."
                            f" Invalid Scheme: {scheme}"
                        )
                        return False

            elif weight_scheme or input_scheme:
                # weight only quantization
                logger.info(
                    "Weight only quantization detected, "
                    "turning off sparse 24 compression."
                )
                return False

        return True
