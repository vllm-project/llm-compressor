from typing import Optional

from compressed_tensors import CompressionFormat
from compressed_tensors.quantization.utils import (
    is_model_quantized,
    is_module_quantized,
    iter_named_leaf_modules,
)

__all__ = ["infer_quantization_format"]


def infer_quantization_format(
    model, quantization_format: Optional[str] = None, save_compressed: bool = False
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
        quant_types = _get_quant_types(model)
        if quant_types == ["int4"]:  # save packed if everything is int4
            return CompressionFormat.pack_quantized
        elif quant_types == ["float8"]:
            return CompressionFormat.float_quantized

        # otherwise just quantize to int8
        return CompressionFormat.int_quantized
    else:
        # format will be inferred from config
        return None


def _get_quant_types(model):
    """
    Gets a list of all the quantized types present in model
    """
    quant_info = []
    for _, submodule in iter_named_leaf_modules(model):
        if is_module_quantized(submodule):
            weight_scheme = submodule.quantization_scheme.weights
            if weight_scheme is not None:
                weight_bit_depth = weight_scheme.num_bits
                weight_type = weight_scheme.type
                weight_info = f"{weight_type}{weight_bit_depth}"
                if weight_info not in quant_info:
                    quant_info.append(weight_info)

    return quant_info
