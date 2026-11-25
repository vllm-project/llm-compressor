"""Restore compressed input modules before initializing new compression recipes."""

import torch
from compressed_tensors.compressors import decompress_module
from compressed_tensors.quantization import QuantizationMetadata, QuantizationStatus
from torch.nn import Embedding, Linear, Module, Parameter

from llmcompressor.modeling.module_restructure import (
    ModuleRestructure,
)


class DecompressionRestructure(ModuleRestructure):
    """Decompress input weights on meta, then repeat on their retained real sources."""

    def __init__(self, *, enabled: bool = False):
        self.enabled = enabled

    def apply(
        self,
        model: Module,
        mapping: dict[Module, dict] | None = None,
    ) -> Module:
        for name, module in model.named_modules():
            if getattr(module, "quantization_status", None) not in (
                QuantizationStatus.COMPRESSED,
                QuantizationStatus.DECOMPRESSED,
            ):
                continue
            if not self.enabled:
                raise ValueError(
                    f"Module {name or '<root>'!r} requires decompression. "
                    "Set layerwise_decompression=True to allow it."
                )
            if mapping is not None:
                mapping[module]["restructures"].append(self)
            self._decompress(module)
        return model

    def restructure_sources(self, model, targets, device, offload_kwargs):
        for entry in targets.values():
            self._decompress(entry["source"])

    @staticmethod
    def _decompress(module: Module) -> None:
        if module.quantization_status == QuantizationStatus.COMPRESSED:
            decompress_module(module, leave_decompressed=False)
            # Packed meta tensors can only approximate unpadded dimensions.
            # Linear/embedding modules retain their exact logical shape.
            shape = None
            if isinstance(module, Linear):
                shape = (module.out_features, module.in_features)
            elif isinstance(module, Embedding):
                shape = (module.num_embeddings, module.embedding_dim)
            if shape is not None and module.weight.is_meta:
                weight = module.weight
                module.weight = Parameter(
                    torch.empty(shape, device="meta", dtype=weight.dtype),
                    requires_grad=weight.requires_grad,
                )
        else:
            # Loaders may have already unpacked the weights.
            QuantizationMetadata.clear_quantization(module)
            del module.quantization_status
