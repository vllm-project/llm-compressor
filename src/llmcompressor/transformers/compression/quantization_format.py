from typing import List, Optional

from compressed_tensors import CompressionFormat
from compressed_tensors.config import SparsityStructure
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils import is_module_quantized
from loguru import logger

__all__ = ["infer_and_set_per_module_quantization_format"]


def _get_quant_compression_format(
    input_args: QuantizationArgs,
    weight_args: QuantizationArgs,
    sparsity_structure: Optional[str] = None,
):
    is_24_structure = (
        SparsityStructure(sparsity_structure) == SparsityStructure.TWO_FOUR
    )
    is_weight_only = weight_args is not None and input_args is None

    if weight_args.num_bits == 4 and weight_args.type == QuantizationType.FLOAT.value:
        return CompressionFormat.nvfp4_pack_quantized

    if is_weight_only:  # w4a16 and w8a16
        is_valid_pack = (
            weight_args.num_bits in [4, 8]
            and weight_args.type == QuantizationType.INT.value
        )
        if not is_valid_pack:  # packing only valid for int4 and int 8
            return CompressionFormat.naive_quantized
        if is_24_structure:
            if (
                weight_args.strategy is not QuantizationStrategy.CHANNEL.value
                and weight_args.strategy is not QuantizationStrategy.GROUP.value
            ):
                # marlin24 kernel only applicable for channel/group quantization
                return CompressionFormat.pack_quantized
            return CompressionFormat.marlin_24
        return CompressionFormat.pack_quantized

    else:  # w8a8 float and int
        if (
            weight_args.type == QuantizationType.FLOAT.value
            and weight_args.num_bits == 8
        ):
            return CompressionFormat.float_quantized
        if weight_args.type == QuantizationType.INT.value:
            return CompressionFormat.int_quantized

        return CompressionFormat.naive_quantized


def infer_and_set_per_module_quantization_format(
    model,
    quantization_format: Optional[str] = None,
    save_compressed: bool = False,
    sparsity_structure: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Infers the quantization format for a model based on its state and provided
    compression arguments. Also updates thhe quantization_scheme.format value
    based on the inferred format. Returns the unique list of formats in the model
    or None if empty list

    For a summary of the formats, see `docs/guides/compression_formats.md`.

    :param model: model to check for quantization, if the model is not quantized no
        quantization format is returned
    :param quantization_format: user provided quantization format, supercedes any
        inferred quantization format
    :param save_compressed: used to infer a quantization format if None is provided
    :return compression format appropriate for model
    """

    if not save_compressed:
        return None

    if quantization_format:
        return [quantization_format]

    unique_formats = []
    for submodule in model.modules():
        if is_module_quantized(submodule):
            weight_scheme = submodule.quantization_scheme.weights
            input_scheme = submodule.quantization_scheme.input_activations
            if weight_scheme is None:
                continue  # no weight quant - nothing to compress
            compression_format = _get_quant_compression_format(
                input_scheme, weight_scheme, sparsity_structure
            )

            # If set, we check if it matches our inferred one
            if submodule.quantization_scheme.format is not None:
                # If it does not, warn the user
                if submodule.quantization_scheme.format != compression_format.value:
                    logger.warning(
                        "The provided format for the module does not match the "
                        "inferred format. Compression may fail "
                    )
            else:
                # If not set, we set ours
                submodule.quantization_scheme.format = compression_format.value

            if submodule.quantization_scheme.format not in unique_formats:
                unique_formats.append(submodule.quantization_scheme.format)

    if len(unique_formats) > 0:
        return unique_formats
    return None
