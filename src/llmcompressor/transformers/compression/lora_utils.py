"""
Utilities for making compressed models LoRA-compatible.

This module provides functions to unpack INT4 quantized weights back to floating-point
tensors, enabling LoRA adapter injection in frameworks like vLLM.
"""

from typing import Optional

import torch
from loguru import logger

__all__ = ["unpack_int4_for_lora", "materialize_weights_for_lora", "get_lora_metadata"]


def unpack_int4_weights(
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    group_size: Optional[int] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Unpack INT4 quantized weights to floating-point format.

    INT4 weights are typically stored with 2 values per byte in a uint8
    tensor. This function unpacks them and dequantizes using the provided
    scales and zero points.

    Args:
        packed_weights: Packed INT4 weights stored as uint8,
            shape [out_features, in_features // 2]
        scales: Quantization scales for dequantization
            - For per-tensor: shape [1]
            - For per-channel: shape [out_features]
            - For grouped: shape [out_features, num_groups]
        zero_points: Optional zero points for asymmetric quantization,
            same shape as scales
        group_size: Group size for grouped quantization (e.g., 128).
            Required if using grouped scales.
        output_dtype: Output dtype for unpacked weights
            (default: torch.float16)

    Returns:
        Unpacked and dequantized weights with shape [out_features, in_features]

    Example:
        >>> # Grouped INT4 quantization with group_size=128
        >>> packed = torch.randint(0, 255, (4096, 2048), dtype=torch.uint8)
        >>> # 4096 // 128 = 32 groups
        >>> scales = torch.randn(4096, 32, dtype=torch.float16)
        >>> unpacked = unpack_int4_weights(packed, scales, group_size=128)
        >>> unpacked.shape
        torch.Size([4096, 4096])
    """
    # Validate inputs
    if packed_weights.dtype != torch.uint8:
        raise ValueError(f"packed_weights must be uint8, got {packed_weights.dtype}")

    # Unpack: extract two INT4 values from each uint8 byte
    # Lower 4 bits: value & 0x0F
    # Upper 4 bits: (value >> 4) & 0x0F
    out_features, packed_in_features = packed_weights.shape
    in_features = packed_in_features * 2

    # Unpack to INT4 values (0-15 range)
    unpacked = torch.zeros(
        (out_features, in_features), dtype=torch.uint8, device=packed_weights.device
    )
    unpacked[:, 0::2] = packed_weights & 0x0F  # Lower 4 bits (even indices)
    unpacked[:, 1::2] = (packed_weights >> 4) & 0x0F  # Upper 4 bits (odd indices)

    # Convert to signed INT4 range: [0, 15] -> [-8, 7]
    # For symmetric quantization, values are in range [-7, 7] typically
    unpacked_signed = unpacked.to(torch.int8) - 8

    # Dequantize: w_float = (w_int - zero_point) * scale
    unpacked_fp = unpacked_signed.to(output_dtype)

    # Handle zero points (for asymmetric quantization)
    if zero_points is not None:
        if zero_points.numel() == 1:
            # Per-tensor zero point
            unpacked_fp = unpacked_fp - zero_points.to(output_dtype)
        elif zero_points.shape[0] == out_features and zero_points.ndim == 1:
            # Per-channel zero point
            unpacked_fp = unpacked_fp - zero_points.view(-1, 1).to(output_dtype)
        elif zero_points.ndim == 2:
            # Grouped zero point
            if group_size is None:
                raise ValueError("group_size must be provided for grouped zero points")
            # Reshape and broadcast zero points
            zp_expanded = zero_points.unsqueeze(2).repeat(1, 1, group_size)
            zp_flat = zp_expanded.view(out_features, -1)[:, :in_features].to(
                output_dtype
            )
            unpacked_fp = unpacked_fp - zp_flat

    # Apply scales
    if scales.numel() == 1:
        # Per-tensor scale
        unpacked_fp = unpacked_fp * scales.to(output_dtype)
    elif scales.shape[0] == out_features and scales.ndim == 1:
        # Per-channel scale
        unpacked_fp = unpacked_fp * scales.view(-1, 1).to(output_dtype)
    elif scales.ndim == 2:
        # Grouped scale
        if group_size is None:
            raise ValueError("group_size must be provided for grouped quantization")
        # Reshape and broadcast scales:
        # [out_features, num_groups] -> [out_features, in_features]
        scales_expanded = scales.unsqueeze(2).repeat(1, 1, group_size)
        scales_flat = scales_expanded.view(out_features, -1)[:, :in_features].to(
            output_dtype
        )
        unpacked_fp = unpacked_fp * scales_flat
    else:
        raise ValueError(f"Unsupported scales shape: {scales.shape}")

    return unpacked_fp


def unpack_int4_for_lora(
    module: torch.nn.Module,
    output_dtype: torch.dtype = torch.float16,
) -> Optional[torch.Tensor]:
    """
    Unpack INT4 weights from a quantized module for LoRA compatibility.

    This function looks for packed weights and quantization parameters in a module
    and unpacks them to floating-point format suitable for LoRA adapter injection.

    Args:
        module: PyTorch module containing quantized weights
        output_dtype: Desired output dtype (default: torch.float16)

    Returns:
        Unpacked weights as a floating-point tensor, or None if module is not quantized

    Example:
        >>> # After loading a compressed INT4 model
        >>> for name, module in model.named_modules():
        ...     if hasattr(module, 'weight_packed'):
        ...         fp_weight = unpack_int4_for_lora(module)
        ...         # Now fp_weight can be used for LoRA injection
    """
    # Check for packed weight buffer (common naming in compressed-tensors)
    if not hasattr(module, "weight_packed") and not hasattr(module, "weight"):
        logger.warning(f"Module {module} has no weight or weight_packed attribute")
        return None

    # Try to find packed weights
    packed_weights = getattr(module, "weight_packed", None)
    if packed_weights is None:
        # Check if regular weight is actually packed (uint8 dtype is a hint)
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dtype == torch.uint8:
            packed_weights = weight
        else:
            logger.debug(f"Module {module} does not appear to have packed INT4 weights")
            return None

    # Get quantization parameters
    scales = getattr(module, "weight_scale", None)
    zero_points = getattr(module, "weight_zero_point", None)

    if scales is None:
        logger.warning(f"Module {module} missing weight_scale for dequantization")
        return None

    # Infer group size from scales shape
    group_size = None
    if scales.ndim == 2:
        out_features, num_groups = scales.shape
        in_features = packed_weights.shape[1] * 2  # Packed dimension * 2
        group_size = in_features // num_groups
        logger.debug(
            f"Inferred group_size={group_size} from scales shape {scales.shape}"
        )

    try:
        unpacked = unpack_int4_weights(
            packed_weights=packed_weights,
            scales=scales,
            zero_points=zero_points,
            group_size=group_size,
            output_dtype=output_dtype,
        )
        logger.debug(
            f"Successfully unpacked weights from shape {packed_weights.shape} "
            f"to {unpacked.shape} with dtype {unpacked.dtype}"
        )
        return unpacked
    except Exception as e:
        logger.error(f"Failed to unpack INT4 weights for module {module}: {e}")
        return None


def materialize_weights_for_lora(
    model: torch.nn.Module,
    target_modules: Optional[list] = None,
    output_dtype: torch.dtype = torch.float16,
    inplace: bool = False,
) -> dict:
    """
    Materialize floating-point weights for all quantized modules in a model.

    This prepares a compressed INT4 model for LoRA injection by unpacking
    all quantized weights to floating-point format. The unpacked weights
    can be stored alongside the packed weights (if inplace=False) or replace them.

    Args:
        model: PyTorch model with quantized weights
        target_modules: Optional list of module name patterns to target
            (e.g., ["q_proj", "v_proj"]). If None, all quantized modules
            will be processed
        output_dtype: Desired output dtype for unpacked weights
            (default: torch.float16)
        inplace: If True, replace packed weights with unpacked weights.
            If False (default), store unpacked weights in a separate
            attribute.

    Returns:
        Dictionary mapping module names to their unpacked weights

    Example:
        >>> # Load INT4 quantized model
        >>> model = AutoModelForCausalLM.from_pretrained("model_int4")
        >>> # Materialize FP16 weights for LoRA
        >>> unpacked_weights = materialize_weights_for_lora(
        ...     model, target_modules=["q_proj", "v_proj"]
        ... )
        >>> # Now model is ready for LoRA adapter injection
    """
    unpacked_weights = {}

    for name, module in model.named_modules():
        # Check if this module should be processed
        if target_modules is not None:
            if not any(target in name for target in target_modules):
                continue

        # Try to unpack weights
        unpacked = unpack_int4_for_lora(module, output_dtype=output_dtype)

        if unpacked is not None:
            unpacked_weights[name] = unpacked

            if inplace:
                # Replace packed weights with unpacked weights
                if hasattr(module, "weight_packed"):
                    delattr(module, "weight_packed")
                module.register_buffer("weight", unpacked)
                logger.info(f"Replaced packed weights with unpacked weights for {name}")
            else:
                # Store unpacked weights alongside packed weights
                module.register_buffer("weight_lora", unpacked)
                logger.info(f"Materialized LoRA-compatible weights for {name}")

    logger.info(f"Materialized {len(unpacked_weights)} modules for LoRA compatibility")
    return unpacked_weights


def get_lora_metadata(model: torch.nn.Module) -> dict:
    """
    Extract LoRA-relevant metadata from a compressed model.

    This function collects information about quantized modules that will be
    useful for LoRA injection, including original shapes, quantization parameters,
    and module types.

    Args:
        model: PyTorch model with quantized weights

    Returns:
        Dictionary containing LoRA metadata for the model

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("model_int4")
        >>> metadata = get_lora_metadata(model)
        >>> metadata["quantized_modules"]  # List of module names with INT4 weights
        >>> metadata["lora_target_modules"]  # Suggested modules for LoRA
    """
    metadata = {
        "quantized_modules": [],
        "lora_target_modules": [],
        "quantization_info": {},
        "lora_compatible": True,
    }

    common_lora_targets = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    for name, module in model.named_modules():
        # Check if module has packed weights
        has_packed = hasattr(module, "weight_packed") or (
            hasattr(module, "weight")
            and hasattr(module.weight, "dtype")
            and module.weight.dtype == torch.uint8
        )

        if has_packed:
            metadata["quantized_modules"].append(name)

            # Collect quantization info
            quant_info = {}
            if hasattr(module, "weight_scale"):
                quant_info["scale_shape"] = tuple(module.weight_scale.shape)
            if hasattr(module, "weight_zero_point"):
                quant_info["has_zero_point"] = True
            if hasattr(module, "weight_packed"):
                quant_info["packed_shape"] = tuple(module.weight_packed.shape)
            elif hasattr(module, "weight"):
                quant_info["packed_shape"] = tuple(module.weight.shape)

            metadata["quantization_info"][name] = quant_info

            # Check if this is a common LoRA target
            if any(target in name for target in common_lora_targets):
                metadata["lora_target_modules"].append(name)

    # Deduplicate target module patterns
    target_patterns = set()
    for name in metadata["lora_target_modules"]:
        for pattern in common_lora_targets:
            if pattern in name:
                target_patterns.add(pattern)

    metadata["suggested_lora_targets"] = sorted(target_patterns)
    metadata["num_quantized_modules"] = len(metadata["quantized_modules"])

    logger.info(
        f"Found {metadata['num_quantized_modules']} quantized modules, "
        f"{len(metadata['lora_target_modules'])} are common LoRA targets"
    )

    return metadata
