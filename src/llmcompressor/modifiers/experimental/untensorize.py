"""
Convert models with TensorizedLinear/BlockTensorizedLinear layers back to dense nn.Linear.

This reconstructs the full weight matrix from the tensor network factorization and replaces
the tensorized layers with standard PyTorch Linear layers.
"""

import torch.nn as nn

from llmcompressor.modifiers.experimental.tensorized_linear import TensorizedLinear
from llmcompressor.modifiers.experimental.block_tensorized_linear import (
    BlockTensorizedLinear,
)

__all__ = ["untensorize_model"]


def untensorize_model(model: nn.Module) -> nn.Module:
    """
    Replace all TensorizedLinear and BlockTensorizedLinear layers with dense nn.Linear.

    Args:
        model: PyTorch model potentially containing tensorized layers

    Returns:
        Model with tensorized layers replaced by dense Linear layers
    """
    # Track replacements for logging
    replacements = []

    def replace_tensorized_recursive(module: nn.Module, name: str = ""):
        """Recursively replace tensorized layers with dense ones."""
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child_module, (TensorizedLinear, BlockTensorizedLinear)):
                # Convert tensorized layer to dense using its to_dense() method
                dense_layer = child_module.to_dense()
                setattr(module, child_name, dense_layer)
                replacements.append(
                    (
                        full_name,
                        type(child_module).__name__,
                        child_module.num_params,
                        dense_layer.weight.numel()
                        + (
                            dense_layer.bias.numel()
                            if dense_layer.bias is not None
                            else 0
                        ),
                    )
                )
            else:
                # Recurse into child modules
                replace_tensorized_recursive(child_module, full_name)

    # Perform replacement
    replace_tensorized_recursive(model)

    # Log results
    if replacements:
        print(f"Converted {len(replacements)} tensorized layers to dense:")
        for name, layer_type, tensor_params, dense_params in replacements:
            compression = tensor_params / dense_params
            print(
                f"  {name} ({layer_type}): {tensor_params:,} → {dense_params:,} params "
                f"(compression: {compression:.2%})"
            )
    else:
        print("No tensorized layers found in model")

    return model
