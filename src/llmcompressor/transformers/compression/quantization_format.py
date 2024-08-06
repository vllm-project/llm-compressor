from typing import Optional

from compressed_tensors import CompressionFormat
from compressed_tensors.config import SparsityCompressionConfig
from compressed_tensors.quantization import QuantizationStrategy, QuantizationType
from compressed_tensors.quantization.utils import (
    is_model_quantized,
    is_module_quantized,
    iter_named_leaf_modules,
)

__all__ = ["infer_quantization_format"]


def infer_quantization_format(
    model,
    quantization_format: Optional[str] = None,
    save_compressed: bool = False,
    sparsity_config: Optional[SparsityCompressionConfig] = None,
) -> str:
    """
    Infers a quantization format based on model state and compression args

    :param model: model to check for quantization, if the model is not quantized no
        quantization format is returned
    :param quantization_format: user provided quantization format, supercedes any
        inferred quantization format
    :param save_compressed: used to infer a quantization format if None is provided
    :return compression format appropriate for model
    """
    if not is_model_quantized(model):
        return None

    if quantization_format is not None:
        return quantization_format

    if save_compressed:
        weight_args, input_args = _get_unique_quant_args(model)
        is_24_structure = (
            sparsity_config and sparsity_config.sparsity_structure == "2:4"
        )
        is_weight_only = len(input_args) == 0 and len(weight_args) > 0

        if is_weight_only:  # w4a16 and w8a16
            is_valid_pack = (
                len(weight_args) == 1
                and weight_args[0].num_bits in [4, 8]
                and weight_args[0].type == QuantizationType.INT.value
            )
            if not is_valid_pack:  # packing only valid for int4 and int 8
                return CompressionFormat.naive_quantized
            if is_24_structure:
                for arg in weight_args:
                    if (
                        arg.strategy is not QuantizationStrategy.CHANNEL.value
                        and arg.strategy is not QuantizationStrategy.GROUP.value
                    ):
                        # marlin24 kernel only applicable for channel/group quantization
                        return CompressionFormat.pack_quantized
                return CompressionFormat.marlin_24
            return CompressionFormat.pack_quantized
        else:  # w8a8 float and int
            if len(weight_args) == 1:
                if (
                    weight_args[0].type == QuantizationType.FLOAT.value
                    and weight_args[0].num_bits == 8
                ):
                    return CompressionFormat.float_quantized
                if weight_args[0].type == QuantizationType.INT.value:
                    return CompressionFormat.int_quantized

            return CompressionFormat.naive_quantized
    else:
        # format will be inferred from config
        return None


def _get_unique_quant_args(model):
    """
    Gets a list of all the unique quantization settings present in model
    """
    quant_info_weight = []
    quant_info_inputs = []
    for _, submodule in iter_named_leaf_modules(model):
        if is_module_quantized(submodule):
            weight_scheme = submodule.quantization_scheme.weights
            input_scheme = submodule.quantization_scheme.input_activations
            if weight_scheme is not None:
                if weight_scheme not in quant_info_weight:
                    quant_info_weight.append(weight_scheme)
            if input_scheme is not None:
                if input_scheme not in quant_info_inputs:
                    quant_info_inputs.append(input_scheme)

    return quant_info_weight, quant_info_inputs
