"""Restore compressed input modules before initializing new compression recipes."""

import torch
from compressed_tensors.compressors import decompress_module
from compressed_tensors.quantization import QuantizationMetadata, QuantizationStatus
from torch.nn import Embedding, Linear, Module, Parameter

from llmcompressor.modeling.module_restructure import ModuleRestructure


class DecompressionRestructure(ModuleRestructure):
    """Decompress input weights on meta, then repeat on their retained real sources."""

    def targets(self, model: Module):
        for name, module in list(model.named_modules()):
            if getattr(module, "quantization_status", None) in (
                QuantizationStatus.COMPRESSED,
                QuantizationStatus.DECOMPRESSED,
            ):
                yield name, module, None

    def restructure(self, source: Module, context) -> Module:
        for _, module, _ in self.targets(source):
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
                # Loaders may have already unpacked the weights. Clear their old
                # qparams/forward without attempting a second dequantization.
                QuantizationMetadata.clear_quantization(module)
                del module.quantization_status
        return source
