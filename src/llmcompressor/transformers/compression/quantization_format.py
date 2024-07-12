from typing import Optional

from compressed_tensors import CompressionFormat
from compressed_tensors.config import SparsityCompressionConfig
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
        weight_schemes, input_schemes = _get_unique_quant_args(model)
        is_24_structure = (
            sparsity_config and sparsity_config.sparsity_structure == "2:4"
        )
        is_weight_only = len(input_schemes) == 0 and len(weight_schemes) > 0

        if is_weight_only:  # w4a16 and w8a16
            if is_24_structure:
                return CompressionFormat.marlin_24
            return CompressionFormat.pack_quantized
        else:  # w8a8 float and int
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
