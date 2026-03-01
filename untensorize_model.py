#!/usr/bin/env python3
"""
CLI utility to convert a model with TensorizedLinear/BlockTensorizedLinear layers back to dense nn.Linear.

This is a convenience wrapper around the untensorize module.
"""

import torch
from llmcompressor.modifiers.experimental.untensorize import untensorize_model


def main():
    """Example usage."""
    print("Usage example:")
    print()
    print("Convert entire model:")
    print("  from llmcompressor.modifiers.experimental import untensorize_model")
    print("  model = torch.load('tensorized_model.pt')")
    print("  dense_model = untensorize_model(model)")
    print("  torch.save(dense_model, 'dense_model.pt')")
    print()
    print("Or convert a single layer:")
    print("  dense_layer = tensorized_layer.to_dense()")
    print()
    print("Note: untensorize_model() modifies the model in-place and returns it")


if __name__ == "__main__":
    main()
